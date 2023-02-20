import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from PIL import Image
from torchsummary import summary

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from networks import *
import torchvision.models as models

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


#Iterating and visualizing the dataset
labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

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

#iterate_visualize(labels_map)


#Preapering your data for training with DataLoaders
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

#Iterate trough the DataLoader
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}") #prints (number of samples in batch, number of channels in image, img_size, img_size)
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
print(img.size())
#plt.imshow(img, cmap="gray")
#plt.show()
print(f"Label: {label}")


#Want to train our model on a hardware acceleratos like the GPU. This is for checkin if torch.cuda i available, else we use CPU. 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


model = CustomNet()



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

image, label = training_data[0]
print(f"Image shape: {image.shape} -> [batch, height, width]") 
print(f"Label: {label}") # label is an int rather than a tensor (it has no shape attribute)

# Create a two layer neural network
model = Ingvildnet()
summary(model)
print(model)
# Pass the image through the model (this will error)
print(model(image))

# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train_loop(train_dataloader, model, loss_fn, optimizer)
#     test_loop(test_dataloader, model, loss_fn)
# print("Done!")



'''
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

input_image = torch.rand(3,28,28)
print(input_image.size())

#We initialize the nn.Flatten layer to onvert each 2D 28x28 image intro a contuguous array of 784 pixel values
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
#The linear layer is a module that applies a linear transformation on the input using its stored weights and biases.
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
#Non-linear activations are what create the complex mappings between the model’s inputs and outputs. They are applied after linear 
# transformations to introduce nonlinearity, helping neural networks learn a wide variety of phenomena.
#In this model, we use nn.ReLU between our linear layers, but there’s other activations to introduce non-linearity in your model.
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

#nn.Sequential is an ordered container of modules. The data is passed through all the modules in the same 
# order as defined. You can use sequential containers to put together a quick network like seq_modules.
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
#The last linear layer of the neural network returns logits - raw values in [-infty, infty] - which are passed to the nn.Softmax module. 
# The logits are scaled to values [0, 1] representing the 
# model’s predicted probabilities for each class. dim parameter indicates the dimension along which the values must sum to 1.
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)


#Many layers inside a neural network are parameterized, i.e. have associated weights and biases that are optimized during training. 
# Subclassing nn.Module automatically tracks all fields defined inside your model object, 
# and makes all parameters accessible using your model’s parameters() or named_parameters() methods.

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
'''