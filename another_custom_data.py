import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from PIL import Image
from torchsummary import summary
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from networks import *
import torchvision.models as models
from torch.utils.data import DataLoader, random_split


transform = Compose([Resize((256, 256)), ToTensor(), transforms.Normalize([0.,], [1.])])

dataset = ImageFolder('C:/Users/ingvilrh/OneDrive - NTNU/Masteroppgave23/full_fishdata', transform=transform)

# Split dataset into train (70%), validation (15%), and test (15%) sets
train_size = int(0.7 * len(dataset))
val_size = test_size = (len(dataset) - train_size) // 2
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


def iterate_visualize(labels_map):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

#train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
#val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, drop_last=True)



#Preapering your data for training with DataLoaders
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

#Iterate trough the DataLoader
# Display image and label.
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}") #prints (number of samples in batch, number of channels in image, img_size, img_size)
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# print(img.size())

# plt.imshow(img)

plt.show()
print(f"Label: {label}")


#Want to train our model on a hardware acceleratos like the GPU. This is for checkin if torch.cuda i available, else we use CPU. 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


model = LeNet()
print(model)
'''
input = torch.rand(1,1,32,32)
output = model(input)
print(output)

'''


#Once we set our hyperparams, we can train and optimize our model with an optimization loop. Each iteration of the
#optimization loop is called an epoch
#The Train Loop - iterate over the training dataset and try to converge to optimal parameters.
#The Validation/Test Loop - iterate over the test dataset to check if model performance is improving.

#hyperparameters:
learning_rate = 1e-3
batch_size = 64
epochs = 5

#Loss function measures the degree of dissimilarity of obtained result to the target value, and it is the loss functioon that we 
#want to minimize during training. Common loss functions is nn.MSELoss, nn.NLLLoss nn.CrossEntropyLoss

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
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


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
