import CustomImageDataset as CID
import torch
import numpy as np
import torch.optim as optim
import os
from torcheval.metrics import MulticlassF1Score
from torch.utils.data import SubsetRandomSampler
from torch import nn
import itertools


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


train_dataset = CID.CustomImageDataset(
    annotations_file="./data/images/images/train.csv",
    img_dir="./data/images/images/train/",
    # transform=preprocess
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=10)

# Load the test set
val_dataset = CID.CustomImageDataset(
    annotations_file="./data/images/images/test.csv",
    img_dir="./data/images/images/test/",
    # transform=preprocess
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=10)


def make_model(layer_sizes):
    layers = []
    in_features = 3 * 300 * 400
    layers.append(nn.Flatten())
    layers.append(nn.LayerNorm(in_features))
    for size in layer_sizes:
        layers.append(nn.Linear(in_features, size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout())
        in_features = size
    layers.append(nn.Linear(in_features, 18))
    return nn.Sequential(*layers)


def train(train_loader, test_loader, model, loss_fn, optimizer):
    size = int(len(train_loader.dataset) * 0.7)

    model.train()
    for batch, (X, y) in enumerate(train_loader):
        print(".", end="")
        sys.stdout.flush()
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

    print()
    test_loss, accuracy, f_score = evaluate(model, loss_fn, train_loader)
    print(
        f"Train Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f}, F1-score: {f_score:>8f} \n"
    )
    test_loss, accuracy, f_score = evaluate(model, loss_fn, test_loader)
    print(
        f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f}, F1-score: {f_score:>8f} \n"
    )
    return test_loss, accuracy, f_score

def evaluate(model, loss_fn, loader):
    total_size = len(loader.dataset)
    with torch.no_grad():
        model.eval()
        test_loss, correct = 0, 0
        f_score = MulticlassF1Score(device=device, mean='weighted')

        for X, y in loader:
            print(".", end="")
            sys.stdout.flush()
            X, y = X.to(device), y.to(device)
            pred = model(X)
            f_score.update(pred, y)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        print()
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


def test_architecture(layer_sizes):
    model = make_model(layer_sizes)
    optimizer = optim.Adam(
        model.parameters(),
    )
    loss_fn = nn.CrossEntropyLoss()
    epochs = 2
    model.to(device)
    loss = None
    accuracy = None
    f_score = None
    for epoch in range(epochs):
        print(f"Epoch: {epoch} for layers {layer_sizes}")
        loss, accuracy, f_score = train(
            train_loader, test_loader, model, loss_fn, optimizer
        )
    return model, loss, accuracy, f_score


def test_mlp_architectures():
    print("-" * 196)
    best_f_score = None
    already_tested = set()
    layers_size = [256, 128, 64, 32]
    layers_size = [x for x in layers_size for _ in range(1, 3)]
    for i in range(1, 5):
        combinations = list(itertools.combinations(layers_size, i))

        for x in combinations:
            if x not in already_tested:
                print(f"Using layers {list(x)}")
                already_tested.add(x)
                model, loss, accuracy, f_score = test_architecture(list(x))
                if best_f_score is None or f_score > best_f_score:
                    best_f_score = f_score
                    print(f"New best model found:{list(x)}")
                    save_last_n(model, "best_mlp", 3)
                print("-" * 196)


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
                train_loader, val_loader, model, loss_fn, optimizer
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

test_mlp_architectures()