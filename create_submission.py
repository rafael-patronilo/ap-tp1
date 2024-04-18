import CustomImageDataset as CID
import torch
import numpy as np
import torch.optim as optim
import sys
import os
from torcheval.metrics import MulticlassF1Score
from torch.utils.data import SubsetRandomSampler
import torchvision
from torch import nn
import pandas as pd
import itertools

device = "cuda" if torch.cuda.is_available() else "cpu"

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0]
    ),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load the test set
val_dataset = CID.CustomImageDataset(
    annotations_file="./data/images/images/test.csv",
    img_dir="./data/images/images/test/",
    transform=preprocess
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64)


def create_submission(model):
    #model = torch.load("training_mlp_0.pth")
    model.eval()
    model.to(device)
    # pred = torch.tensor([])
    pred = np.array([])

    for X, Y in val_loader:
        print(".", end="")
        sys.stdout.flush()
        X = X.to(device)
        Y = Y.to(device)
        pred = np.concatenate((pred, model(X).cpu().detach().numpy().argmax(axis=1)))
    print()
    # pred = pred.cpu().detach().numpy().argmax(axis=1)
    print(pred)
    submission = pd.DataFrame({"Id": val_dataset.img_labels.iloc[:, 0], "main_type": pred})
    submission.to_csv("./submission_last.csv", index=False)

if __name__ == "__main__":
    model = torch.load(sys.argv[1], map_location=device)
    create_submission(model)
    print("Submission created.")