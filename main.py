import CustomImageDataset as CID
import torch
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
from torch import nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# Load the dataset
train_dataset = CID.CustomImageDataset(annotations_file='./data/images/images/train.csv', img_dir='./data/images/images/train/')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

# Load the test set
val_dataset = CID.CustomImageDataset(annotations_file='./data/images/images/test.csv', img_dir='./data/images/images/test/')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Flatten(),
    nn.LazyLinear(32),
    nn.ReLU(),
    nn.LazyLinear(18),
    nn.Softmax(dim=1) # [0, 0, 0, 1, 0, ..., 0]   3
)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
epoch = 50
for epoch in range(50):
    train(train_loader,model,loss_fn,optimizer)


model.eval()
size = len(train_loader.dataset)
for batch, (X, y) in enumerate(train_dataset):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
#test_dataset = None
#y_pred_tensor = None

#submission = pd.DataFrame({'Id': test_dataset.img_labels.iloc[:, 0], 'main_type': y_pred_tensor})
#submission.to_csv('./submission.csv', index=False)