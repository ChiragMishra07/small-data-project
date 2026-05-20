import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np

def imshow(inp, title=None):

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)

    if title is not None:
        plt.title(title)

    plt.pause(2.0)


def test_model(test_loader, trained_model, class_names):

    print('\n========== SHOWING TEST RESULTS ==========\n')

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    trained_model = trained_model.to(device)

    trained_model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for i, (inputs, labels) in enumerate(test_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = trained_model(inputs)

            _, preds = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (preds == labels).sum().item()

            # Show prediction images
            for j in range(len(inputs)):

                inp = inputs.cpu().data[j]

                actual = class_names[labels.cpu()[j]]

                predicted = class_names[preds.cpu()[j]]

                imshow(
                    inp,
                    title=f"Actual: {actual} | Predicted: {predicted}"
                )

    accuracy = 100 * correct / total

    print("\n===================================")
    print(f"FINAL TEST ACCURACY: {accuracy:.2f}%")
    print("===================================\n")