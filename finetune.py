import CustomImageDataset as CID
import torch
import numpy as np
import torch.optim as optim
import sys
import os
from torcheval.metrics import MulticlassF1Score
from torch.utils.data import SubsetRandomSampler
from torch import nn
import torchvision
import pandas as pd
import itertools
import traceback
from torchsummary import summary
import gc

EPOCHS_PER_MODEL = 50
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

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

orig_train_dataset = CID.CustomImageDataset(
    annotations_file="./data/images/images/train.csv",
    img_dir="./data/images/images/train/",
    transform=preprocess
)

# Load the test set
val_dataset = CID.CustomImageDataset(
    annotations_file="./data/images/images/test.csv",
    img_dir="./data/images/images/test/",
    transform=preprocess
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)





def split_training_set(seed_source=57473):
    global train_dataset, test_dataset, train_loader, test_loader
    seed = hash(seed_source)
    print(f"Splitting training set using seed {seed} from {seed_source}")
    train_dataset, test_dataset = torch.utils.data.random_split(
        orig_train_dataset, 
        [0.7, 0.3], 
        generator=torch.Generator().manual_seed(seed)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=True
    )

def prepare_pretrained_model(model):
    model.fc = nn.Linear(model.fc.in_features, 18)
    nn.init.xavier_uniform_(model.fc.weight)

def train(train_loader, test_loader, model, loss_fn, optimizer):
    size = len(train_loader.dataset)

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
        del loss
        gc.collect()
    print()
    print("Train Error:")
    test_loss, accuracy, f_score = evaluate(
        model, loss_fn, train_loader
    )
    print(
        f"Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f}, F1-score: {f_score:>8f} \n"
    )
    print("Test Error:")
    test_loss, accuracy, f_score = evaluate(
        model, loss_fn, test_loader
    )
    print(
        f"Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f}, F1-score: {f_score:>8f} \n"
    )
    return test_loss, accuracy, f_score

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

def evaluate(model, loss_fn, loader):
    total_size = len(loader.dataset)
    with torch.no_grad():
        model.eval()
        test_loss, correct = 0, 0
        f_score = MulticlassF1Score(device=device)

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

def train_fine_tuning(name, model, learning_rate,
                      param_group=True):
    split_training_set(name)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = None
    if param_group:
        params_1x = [param for name, param in model.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        optimizer = torch.optim.SGD([{'params': params_1x},
                                   {'params': model.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    #epoch = 0
    best_f_score = None
    try:
        for epoch in range(EPOCHS_PER_MODEL):
            print(f"Epoch: {epoch}")
            loss, accuracy, f_score = train(
                train_loader, test_loader, model, loss_fn, optimizer
            )
            if epoch % 25 == 0:
                print("Saving model")
                save_last_n(model, f"training_{name}", 1)
            if best_f_score is None or f_score > best_f_score:
                best_f_score = f_score
                print("New best model found")
                save_last_n(model, f"training_best_{name}", 1)
        print("Finished training {name}")
        print("Saving model")
        save_last_n(model, f"training_{name}", 1)
    except KeyboardInterrupt:
        print("Training stopped, saving current model")
        save_last_n(model, f"training_{name}", 2)
    except Exception as e:
        print("Error during training:")
        print(traceback.format_exc())
        print("Saving model")
        save_last_n(model, f"training_{name}", 2)



models = [
    ("resnet18", lambda : torchvision.models.resnet18(pretrained=True)),
]

for name, builder in models:
    print("Training model", name)
    try:
        model = builder()
        prepare_pretrained_model(model)
        model.to(device)
        print(summary(model, (3, 300, 400)))
        train_fine_tuning(name, model, 0.001, param_group=True)
    except Exception as e:
        print("Error during building model:")
        print(traceback.format_exc())
        print("Skipping")