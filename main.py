
import CustomImageDataset as CID
import torch
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
from torcheval.metrics import MulticlassF1Score
from torch.utils.data import SubsetRandomSampler
from torch import nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(device)
print(torch.cuda.is_available())

preprocess = transforms.Normalize((0.49267729*255.,0.43429736*255.,0.37517854*255.),(0.27001957*255., 0.26459164*255., 0.26774524*255.))


train_dataset = CID.CustomImageDataset(annotations_file='./data/images/images/train.csv', img_dir='./data/images/images/train/',
                                       transform=preprocess
                                       )
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Load the test set
val_dataset = CID.CustomImageDataset(annotations_file='./data/images/images/test.csv', img_dir='./data/images/images/test/',
                                     transform=preprocess
                                     )
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Flatten(),
    nn.LazyLinear(1024),
    nn.ReLU(),
    nn.LazyLinear(256),
    nn.ReLU(),
    nn.LazyLinear(128),
    nn.ReLU(),
    nn.LazyLinear(64),
    nn.ReLU(),
    nn.LazyLinear(32),
    nn.ReLU(),
    nn.LazyLinear(18), # [100, 200, 1567, ...]
    #nn.Softmax(dim=1) # [0.1, 0.2, 1, 0, 0, ..., 0]   3
)
for batch, (X, y) in enumerate(train_loader):
    print(torch.max(X))
    print(torch.min(X))
    print(torch.mean(X))    
    break
def train(train_loader,test_loader, model, loss_fn, optimizer):
    size = int(len(train_loader.dataset)*0.7)
    
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        #y = nn.functional.one_hot(y, num_classes=18)
        #y = torch.tensor(y.clone().detach(),dtype=torch.float32)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        #loss, current = loss.item(), ((batch )*64+ len(X) )if not len(X)== 64 else (batch+1)*len(X)
        #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    with torch.no_grad():
        model.eval()
        test_loss, correct = 0, 0
        
        f_score = MulticlassF1Score(device=device)
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            f_score.update(pred, y)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= len(test_loader)
        correct /= int(len(test_loader.dataset)*0.3)
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}, F1-score: {f_score.compute():>8f} \n")
optimizer = optim.Adam(model.parameters(),)
loss_fn = nn.CrossEntropyLoss()
epochs = 10
model.to(device)
total_size = len(train_loader.dataset)
#indices = list(range(total_size))
split = int(0.7 * total_size)  


indices = np.arange(total_size)
print(indices)
train_indices = indices[:split]
test_indices = indices[split:]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=64, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=64, sampler=test_sampler)

for epoch in range(epochs):
    print("Epoch: ", epoch)

    train(train_loader,test_loader,model,loss_fn,optimizer)
    

# Save model


# model.eval()
# size = len(train_loader.dataset)
# for batch, (X, y) in enumerate(train_dataset):
#         X, y = X.to(device), y.to(device)

#         # Compute prediction error
#         pred = model(X)
#         loss = loss_fn(pred, y)

#         # Backpropagation
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         if batch % 100 == 0:
#             loss, current = loss.item(), (batch + 1) * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
#test_dataset = None
#y_pred_tensor = None

#submission = pd.DataFrame({'Id': test_dataset.img_labels.iloc[:, 0], 'main_type': y_pred_tensor})
#submission.to_csv('./submission.csv', index=False)