from TestModel import test_model
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def load_and_standardize_data(path):
    df = pd.read_csv(path, sep=',')
    df = df.fillna(-99)
    df = df.values.reshape(-1, df.shape[1]).astype('float32')
    X_train, X_test = train_test_split(df, test_size=0.3, random_state=42)
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)   
    return X_train, X_test, scaler

class DataBuilder(Dataset):
    def __init__(self, path, train=True):
        self.X_train, self.X_test, self.standardizer = load_and_standardize_data(path)
        if train:
            self.x = torch.from_numpy(self.X_train)
            self.len=self.x.shape[0]
        else:
            self.x = torch.from_numpy(self.X_test)
            self.len=self.x.shape[0]
        del self.X_train
        del self.X_test 
    def __getitem__(self,index):      
        return self.x[index]
    def __len__(self):
        return self.len

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
    
    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD

class Autoencoder(nn.Module):
    def __init__(self,D_in,H=50,H2=12,latent_dim=3):
        #Encoder
        super(Autoencoder,self).__init__()
        self.linear1=nn.Linear(D_in,H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2=nn.Linear(H,H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3=nn.Linear(H2,H2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)
        
        # Latent vectors
        self.fc1 = nn.Linear(H2, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

        # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H2)
        self.fc_bn4 = nn.BatchNorm1d(H2)
        
        # Decoder
        self.linear4=nn.Linear(H2,H2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5=nn.Linear(H2,H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6=nn.Linear(H,D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)
        
        self.relu = nn.ReLU()
        
    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)
        
        return r1, r2
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def generate_fake(mu, logvar, no_samples, scaler, model):
    #With trained model, generate some data
    sigma = torch.exp(logvar/2)
    q = torch.distributions.Normal(mu.mean(axis=0), sigma.mean(axis=0))
    z = q.rsample(sample_shape=torch.Size([no_samples]))
    with torch.no_grad():
        pred = model.decode(z).cpu().numpy()
    fake_data = scaler.inverse_transform(pred)
    return fake_data

# When you have all the code in place to generate synthetic data, uncomment the code below to run the model and the tests. 
def main():

    # Original dataset path
    ORIGINAL_DATA_PATH = 'data/loan_continuous.csv'

    # Load original dataset
    data = pd.read_csv(ORIGINAL_DATA_PATH)

    # Split data with Loan Status = 1
    denied_loans = data[data['Loan Status'] == 1]

    # Save denied loans temporarily
    denied_loans.to_csv(
        'data/denied_loans.csv',
        index=False
    )

    # Create DataLoaders
    trainloader = DataLoader(
        DataBuilder(
            'data/denied_loans.csv',
            train=True
        ),
        batch_size=64,
        shuffle=True
    )

    valloader = DataLoader(
        DataBuilder(
            'data/denied_loans.csv',
            train=False
        ),
        batch_size=64,
        shuffle=False
    )

    # Get input dimensions
    D_in = denied_loans.shape[1]

    # Initialize model
    model = Autoencoder(D_in)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    # Loss function
    loss_mse = CustomLoss()

    epochs = 500

    # Training loop
    for epoch in range(epochs):

        model.train()

        train_loss = 0

        for data_batch in trainloader:

            data_batch = data_batch.float()

            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data_batch)

            loss = loss_mse(
                recon_batch,
                data_batch,
                mu,
                logvar
            )

            loss.backward()

            train_loss += loss.item()

            optimizer.step()

        # Validation
        model.eval()

        val_loss = 0

        with torch.no_grad():

            for val_batch in valloader:

                val_batch = val_batch.float()

                recon_batch, mu, logvar = model(val_batch)

                loss = loss_mse(
                    recon_batch,
                    val_batch,
                    mu,
                    logvar
                )

                val_loss += loss.item()

        if epoch % 50 == 0:

            print(
                f"Epoch {epoch} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

    # Generate synthetic data
    scaler = trainloader.dataset.standardizer

    fake_data = generate_fake(
        mu,
        logvar,
        50000,
        scaler,
        model
    )

    # Convert synthetic data to dataframe
    fake_df = pd.DataFrame(
        fake_data,
        columns=data.columns
    )

    # Combine with original dataset
    augmented_data = pd.concat(
        [data, fake_df],
        ignore_index=True
    )

    # Save augmented dataset
    augmented_data.to_csv(
        'data/loan_continuous_expanded.csv',
        index=False
    )

    print("Synthetic data generated successfully!")

    # Test model on augmented dataset
    DATA_PATH = 'data/loan_continuous_expanded.csv'

    test_model(DATA_PATH)

if __name__ == '__main__':
    main()
    print("done")