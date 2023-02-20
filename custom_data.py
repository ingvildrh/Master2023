#Creating a custom dataset for my files

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
from networks import *
from torchsummary import summary

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

'''
Labels:
0: Black Sea Sprat
1: Hourse Mackerel
2: Red Sea Bream
3: Shrimp
'''
transform = Compose([Resize((28, 28)), ToTensor(), transforms.Normalize([0.,], [1.])])

dataset = ImageFolder('C:/Users/ingvilrh/OneDrive - NTNU/Masteroppgave23/full_fishdata', transform=transform)

#Get the label of the nth image    
label = dataset.classes[dataset.targets[200]]
print("The length of the dataset: ")
print(len(dataset))

# Split dataset into train (70%), validation (15%), and test (15%) sets
train_size = int(0.7 * len(dataset))
val_size = test_size = (len(dataset) - train_size) // 2
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=True)

print(len(train_loader))
print(val_loader)
print(test_loader)

#Showing image and labeø
train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}") #prints (number of samples in batch, number of channels in image, img_size, img_size)
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
print(img.size())
img = img.squeeze()
print(img.size())
label = train_labels[0]
# Assuming your image is stored in the 'img' variable with shape (3, 28, 28)
img = img.permute(1, 2, 0).numpy()  # Convert to NumPy array and transpose dimensions
img = (img * 255).astype(np.uint8)  # Convert from [0, 1] float range to [0, 255] int range
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

#Want to train our model on a hardware acceleratos like the GPU. This is for checkin if torch.cuda i available, else we use CPU. 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")  

model = NeuralNetwork()
print(model)

#Once we set our hyperparams, we can train and optimize our model with an optimization loop. Each iteration of the
#optimization loop is called an epoch
#The Train Loop - iterate over the training dataset and try to converge to optimal parameters.
#The Validation/Test Loop - iterate over the test dataset to check if model performance is improving.

#hyperparameters:
learning_rate = 1e-3
epochs = 5
batch_size = 64

#Loss function measures the degree of dissimilarity of obtained result to the target value, and it is the loss functioon that we 
#want to minimize during training. Common loss functions is nn.MSELoss, nn.NLLLoss nn.CrossEntropyLoss

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1 ) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 10
for epo in range(epochs):
    print(f"Epoch {epo+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model, loss_fn)
print("Done!")