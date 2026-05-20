import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")


def load_xy(df):

    # Replace missing values
    df = df.fillna(-99)

    target = "Loan Status"

    x_df = df.drop(target, axis=1)

    y_df = df[target]

    y_df = LabelEncoder().fit_transform(y_df)

    return x_df, y_df


def run_test(x_df, y_df):

    # Fast classifier
    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42
    )

    # Train model
    model.fit(x_df, y_df)

    # Predictions
    y_pred = model.predict(x_df)

    # Results
    print("\nResults on the dataset:\n")

    print(classification_report(y_df, y_pred))


def test_model(datapath):

    print(f"\nTesting dataset: {datapath}\n")

    # Load dataset
    df_base = pd.read_csv(datapath, sep=',')

    # Prepare X and y
    x_df, y_df = load_xy(df_base)

    # Run model
    run_test(x_df, y_df)