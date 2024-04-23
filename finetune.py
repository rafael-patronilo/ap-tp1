import CustomImageDataset as CID
import torch
import numpy as np
import sys
import os
from torcheval.metrics import MulticlassF1Score
from torch import nn
import torchvision
import traceback
import time

# from torchsummary import summary
best_f_score = 0.0
EPOCHS_PER_MODEL = 30
EPOCHS_PER_TRIAL = 25
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

preprocess = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomHorizontalFlip(p=0.33),
        torchvision.transforms.RandomVerticalFlip(p=0.33),
        torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation((180,180))],  p=0.33),
        torchvision.transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]
        ),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

orig_train_dataset = CID.CustomImageDataset(
    annotations_file="./data/images/images/train.csv",
    img_dir="./data/images/images/train/",
    transform=preprocess,
)

# Load the test set
val_dataset = CID.CustomImageDataset(
    annotations_file="./data/images/images/test.csv",
    img_dir="./data/images/images/test/",
    transform=preprocess,
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)


def split_training_set(seed_source=57473):
    global train_dataset, test_dataset, train_loader, test_loader

    def custom_hash(string):
        # generated by ChatGPT
        # we use this because python's hash function is not stable across runs
        prime = 31
        hash_value = 0
        for char in string:
            hash_value = (hash_value * prime + ord(char)) % (2**32)  # Use a 32-bit hash
        return hash_value

    seed = custom_hash(seed_source)
    print(f"Splitting training set using seed {seed} from {seed_source}")
    train_dataset, test_dataset = orig_train_dataset.class_split(
        0.7, random=np.random.default_rng(seed)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)


def prepare_pretrained_model(model):
    model.fc = nn.Linear(model.fc.in_features, 18)
    nn.init.xavier_uniform_(model.fc.weight)


def train(train_loader, test_loader, model, loss_fn, optimizer):
    size = len(train_loader.dataset)

    model.train()
    optimizer.zero_grad()

    for batch, (X, y) in enumerate(train_loader):
        print(".", end="")
        sys.stdout.flush()
        X, y = X.to(device), y.to(device)
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
        # del loss
        # gc.collect()
        optimizer.zero_grad()
    print()
    print("Train Error:")
    train_metrics = evaluate(model, loss_fn, train_loader)
    test_loss, accuracy, f_score = train_metrics
    print(
        f"Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f}, F1-score: {f_score:>8f} \n"
    )
    print("Test Error:")
    test_metrics = evaluate(model, loss_fn, test_loader)
    test_loss, accuracy, f_score = test_metrics
    print(
        f"Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f}, F1-score: {f_score:>8f} \n"
    )
    return test_metrics, train_metrics


def save_last_n(model, name, n):
    file = f"{name}_{n-1}.pth"
    if os.path.isfile(file):
        os.remove(file)
    for i in range(1, n):
        old_file = f"{name}_{i-1}.pth"
        file = f"{name}_{i}.pth"
        if os.path.isfile(file):
            os.rename(file, old_file)
    torch.save(model, f"{name}_0.pth")


def evaluate(model, loss_fn, loader):
    total_size = len(loader.dataset)
    with torch.no_grad():
        model.eval()
        test_loss, correct = 0, 0
        f_score = MulticlassF1Score(num_classes=18, average="weighted", device=device)

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


def convert_seconds(total_seconds):
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"ETA: {hours}h {minutes}m {seconds}s"


def train_fine_tuning(
    name, 
    model, 
    learning_rate=0.001, 
    weight_decay=0.001,
    other_seed_data="", 
    param_group=True, 
    from_epoch=0, 
    epochs=EPOCHS_PER_MODEL,
    stop_criterion = None
):
    if not learning_rate is float:
        learning_rate = float(learning_rate)
    if not weight_decay is float:
        weight_decay = float(weight_decay)
    if not epochs is int:
        epochs = int(epochs)
    if not from_epoch is int:
        from_epoch = int(from_epoch)
    split_training_set(name + other_seed_data)
    print(f"LR = {learning_rate}, WD = {weight_decay}")
    loss_fn = nn.CrossEntropyLoss(
        weight=orig_train_dataset.get_class_weights_tensor().to(device)
    )
    optimizer = None
    if param_group:
        params_1x = [
            param
            for name, param in model.named_parameters()
            if name not in ["fc.weight", "fc.bias"]
        ]
        optimizer = torch.optim.SGD(
            [
                {"params": params_1x},
                {"params": model.fc.parameters(), "lr": learning_rate * 10},
            ],
            lr=learning_rate,
            weight_decay=0.001,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    last_epoch = -1
    f_score = 0.0
    # epoch = 0
    best_f_score = None
    try:
        for epoch in range(from_epoch, epochs):
            tsi = time.time()
            print(f"Epoch: {epoch}")

            # ts stores the time in seconds

            test_metrics, train_metrics = train(
                train_loader, test_loader, model, loss_fn, optimizer
            )
            loss, accuracy, f_score = test_metrics
            if stop_criterion is not None and stop_criterion(test_metrics, train_metrics):
                print("Stopping training")
                return 0.0
            if epoch % 25 == 0:
                print("Saving model")
                save_last_n(model, f"training_{name}", 1)
                with open(f"{name}.txt", "a") as f:
                    f.write(f"Epoch: {epoch}, F1-score: {f_score}\n")
            if best_f_score is None or f_score > best_f_score:
                best_f_score = f_score
                print("New best model found")
                save_last_n(model, f"training_best_{name}", 1)
                with open(f"{name}.txt", "a") as f:
                    f.write(f"Epoch: {epoch}, Best F1-score: {f_score}\n")
            tsf = time.time()
            eta = (epochs - 1 - epoch) * (tsf - tsi)
            print(convert_seconds(eta))
            last_epoch = epoch

    except KeyboardInterrupt:
        print("Training stopped, saving current model")
        save_last_n(model, f"training_{name}_epoch_{last_epoch}", 1)
        cmd = input("If you want to exit, type q. Otherwise, hit enter.")
        if cmd == "q":
            exit(0)
    except Exception:
        print("Error during training:")
        print(traceback.format_exc())
        print("Saving model")
        save_last_n(model, f"training_{name}_epoch_{last_epoch}", 1)
    return f_score


def objective(
    trial, name, builder, param_group=True, from_epoch=0, epochs=EPOCHS_PER_TRIAL
):
    def stop_criterion(test_metrics, train_metrics):
        MIN_FSCORE = 0.05
        MAX_GENERALIZATION_GAP = 0.1
        loss, accuracy, f_score = test_metrics
        train_loss, train_accuracy, train_f_score = train_metrics
        generalization_gap = f_score - train_f_score
        stop = False
        if f_score < MIN_FSCORE:
            print(f"F1-score too low: {f_score} < {MIN_FSCORE}")
            stop = True
        if generalization_gap > MAX_GENERALIZATION_GAP:
            print(f"Generalization gap too high: {generalization_gap} > {MAX_GENERALIZATION_GAP}")
            stop = True
        return stop
    model = builder()
    
    # Define hyperparameters using trial object
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-1)
    return train_fine_tuning(name=name, 
                      model=model, 
                      other_seed_data=f"_trial{trial.number}",
                      learning_rate=learning_rate, 
                      weight_decay=weight_decay, 
                      param_group=param_group, 
                      from_epoch=from_epoch, 
                      epochs=epochs,
                      stop_criterion=stop_criterion)


models = [
    # ("alexnet", lambda : torchvision.models.alexnet(pretrained=True)),
    # ("regnet_y_400mf", lambda: torchvision.models.regnet_y_400mf(pretrained=True)),
    # ("regnet_x_400mf", lambda: torchvision.models.regnet_x_400mf(pretrained=True)),
    # ("regnet_x_800mf", lambda: torchvision.models.regnet_x_800mf(pretrained=True)),
    # ("regnet_y_800mf", lambda: torchvision.models.regnet_y_800mf(pretrained=True)),
    # ("regnet_x_1_6gf", lambda: torchvision.models.regnet_x_1_6gf(pretrained=True)),
    # ("googlenet", lambda: torchvision.models.googlenet(pretrained=True)),
    ("resnet18", lambda: torchvision.models.resnet18(pretrained=True)),
]

if __name__ == "__main__":
    # start = int(sys.argv[1])
    # print("Training the ", "odd" if start == 1 else "even", " models")
    # models = models[start::2]
    print([x[0] for x in models])

    for name, builder in models:
        print("=" * 100)
        print("=" * 100)
        print("Training model", name)
        print("=" * 100)
        print("=" * 100)
        try:
                
            model = builder()
            prepare_pretrained_model(model)
            model.to(device)
            # print(summary(model, (3, 300, 400)))
            
            kargs = {arg[0]:arg[1] for arg in (arg.split("=") for arg in sys.argv[1:])}

            train_fine_tuning(name, model, param_group=True, **kargs)
        except KeyboardInterrupt:
            cmd = input("If you want to exit, type q. Otherwise, hit enter.")
            if cmd == "q":
                exit(0)
        except Exception:
            print("Error during building model:")
            print(traceback.format_exc())
            print("Skipping")
