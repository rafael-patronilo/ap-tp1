import CustomImageDataset as CID
import torch
import numpy as np
import torch.optim as optim
import os
from torcheval.metrics import MulticlassF1Score
from torch.utils.data import SubsetRandomSampler
from torch import nn
import pandas as pd
import itertools

device = "cuda" if torch.cuda.is_available() else "cpu"


train_dataset = CID.CustomImageDataset(
    annotations_file="./data/images/images/train.csv",
    img_dir="./data/images/images/train/",
    # transform=preprocess
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Load the test set
val_dataset = CID.CustomImageDataset(
    annotations_file="./data/images/images/test.csv",
    img_dir="./data/images/images/test/",
    # transform=preprocess
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)


def make_model():
    # TODO: Implement a function that creates a model with the given layer sizes
    model = nn.Sequential(
        nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.LazyConv2d(256, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # nn.LazyConv2d(384, kernel_size=3, padding=1),
        # nn.ReLU(),
        # nn.LazyConv2d(384, kernel_size=3, padding=1),
        # nn.ReLU(),
        # nn.LazyConv2d(256, kernel_size=3, padding=1),
        # nn.ReLU(),
        # nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        nn.LazyLinear(18),
    )

    return model

def make_vgg(vgg_blocks, linear_layers):
    layers = []
    for (num_convs, out_channels) in vgg_blocks:
        if num_convs > 0:
            for _ in range(num_convs):
                layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    layers.append(nn.Flatten())
    for out_features in linear_layers:
        layers.append(nn.LazyLinear(out_features))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout())
    layers.append(nn.LazyLinear(18))
    return nn.Sequential(*layers)

def test_vgg(vgg_blocks, linear_layers):
    model = make_vgg(vgg_blocks, linear_layers)
    optimizer = optim.Adam(
        model.parameters(),
    )
    loss_fn = nn.CrossEntropyLoss()
    epochs = 15
    model.to(device)
    loss = None
    accuracy = None
    f_score = None
    for epoch in range(epochs):
        print(f"Epoch: {epoch} for {vgg_blocks} VGG blocks and {linear_layers} linear layers")
        loss, accuracy, f_score = train(
            train_loader, test_loader, model, loss_fn, optimizer
        )
    return model, loss, accuracy, f_score

def test_vgg_architectures():
    print("-" * 196)
    out_channels = [64, 128, 256, 512, 512]
    all_conv_numbers = [x for x in itertools.product(range(1, 3), repeat=len(out_channels)) if sum(x) <= 8]
    all_conv_numbers.extend([x for x in itertools.product(range(1, 3), repeat=len(out_channels)-1) if sum(x) <= 6])
    linear_layer_sizes = [256, 1024, 4096]
    best_f_score = None
    all_linear_layers = zip(linear_layer_sizes, linear_layer_sizes)
    for i, (conv_numbers, linear_layers) in enumerate(itertools.product(all_conv_numbers, all_linear_layers)):
        vgg_blocks = tuple(zip(conv_numbers, out_channels))
        model, loss, accuracy, f_score = test_vgg(vgg_blocks, linear_layers)
        if best_f_score is None or f_score > best_f_score:
            best_f_score = f_score
            print(f"New best model found:{vgg_blocks} {linear_layers}")
            save_last_n(model, "best_vgg", 3)
        print("-" * 196)
        

def train(train_loader, test_loader, model, loss_fn, optimizer):
    size = int(len(train_loader.dataset) * 0.7)

    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        # y = nn.functional.one_hot(y, num_classes=18)
        # y = torch.tensor(y.clone().detach(),dtype=torch.float32)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        # loss, current = loss.item(), ((batch )*64+ len(X) )if not len(X)== 64 else (batch+1)*len(X)
        # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    test_loss, accuracy, f_score = evaluate(
        model, loss_fn, train_loader, int(len(test_loader.dataset) * 0.7)
    )
    print(
        f"Train Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f}, F1-score: {f_score:>8f} \n"
    )
    test_loss, accuracy, f_score = evaluate(
        model, loss_fn, test_loader, int(len(test_loader.dataset) * 0.3)
    )
    print(
        f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f}, F1-score: {f_score:>8f} \n"
    )
    return test_loss, accuracy, f_score


def evaluate(model, loss_fn, loader, total_size):
    with torch.no_grad():
        model.eval()
        test_loss, correct = 0, 0
        f_score = MulticlassF1Score(device=device)

        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            f_score.update(pred, y)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= len(loader)
        correct /= total_size
        accuracy = 100 * correct
        f_score = f_score.compute()
    return test_loss, accuracy, f_score


total_size = len(train_loader.dataset)
# indices = list(range(total_size))
split = int(0.7 * total_size)
indices = np.arange(total_size)
train_indices = indices[:split]
test_indices = indices[split:]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(
    train_loader.dataset, batch_size=64, sampler=train_sampler
)
test_loader = torch.utils.data.DataLoader(
    train_loader.dataset, batch_size=64, sampler=test_sampler
)


def save_last_n(model, name, n):
    file = f"{name}_{n-1}.pth"
    if os.path.isfile(file):
        os.remove(file)
    for i in range(1, n):
        old_file = f"{name}_{i-1}.pth"
        file = f"{name}_{i}.pth"
        if os.path.isfile(file):
            os.rename(old_file, file)
    torch.save(model, f"{name}_0.pth")


def test_architecture():
    model = make_model()
    optimizer = optim.Adam(
        model.parameters(),
    )
    loss_fn = nn.CrossEntropyLoss()
    epochs = 15
    model.to(device)
    loss = None
    accuracy = None
    f_score = None
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        loss, accuracy, f_score = train(
            train_loader, test_loader, model, loss_fn, optimizer
        )
    return model, loss, accuracy, f_score


def test_mlp_architectures():
    best_f_score = None
    model, loss, accuracy, f_score = test_architecture()
    if best_f_score is None or f_score > best_f_score:
        best_f_score = f_score
        save_last_n(model, "best_mlp", 3)


def train_indefinitely(model):
    epoch = 0
    optimizer = optim.Adam(
        model.parameters(),
    )
    loss_fn = nn.CrossEntropyLoss()
    best_f_score = None
    try:
        while True:
            print(f"Epoch: {epoch}")
            loss, accuracy, f_score = train(
                train_loader, test_loader, model, loss_fn, optimizer
            )
            if epoch % 25 == 0:
                print("Saving model")
                save_last_n(model, "training_mlp", 3)
                if best_f_score is None or f_score > best_f_score:
                    best_f_score = f_score
                    print("New best model found")
                    save_last_n(model, "training_best_mlp", 1)
            epoch += 1
    except KeyboardInterrupt:
        print("Training stopped, saving current model")
        save_last_n(model, "training_mlp", 4)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Saving current model")
        save_last_n(model, "training_mlp", 4)


# test_mlp_architectures()
# train_indefinitely(torch.load("best_mlp_0.pth"))
model = torch.load("best_mlp_0.pth")
model.eval()
model.to(device)
# pred = torch.tensor([])
pred = np.array([])

for X, Y in val_loader:
    X = X.to(device)
    Y = Y.to(device)
    pred = np.concatenate((pred, model(X).cpu().detach().numpy().argmax(axis=1)))
# pred = pred.cpu().detach().numpy().argmax(axis=1)
print(pred)
submission = pd.DataFrame({"Id": val_dataset.img_labels.iloc[:, 0], "main_type": pred})
submission.to_csv("./submission_best.csv", index=False)


model = torch.load("training_mlp_0.pth")
model.eval()
model.to(device)
# pred = torch.tensor([])
pred = np.array([])

for X, Y in val_loader:
    X = X.to(device)
    Y = Y.to(device)
    pred = np.concatenate((pred, model(X).cpu().detach().numpy().argmax(axis=1)))
# pred = pred.cpu().detach().numpy().argmax(axis=1)
print(pred)
submission = pd.DataFrame({"Id": val_dataset.img_labels.iloc[:, 0], "main_type": pred})
submission.to_csv("./submission_last.csv", index=False)
