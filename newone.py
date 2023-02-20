import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from network import*
from torch import nn

#Image data dir:
dir = 'C:/Users/ingvilrh/OneDrive - NTNU/Masteroppgave23/full_fishdata'

#Transform for the images_
transform = Compose([Resize((256, 256)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#Define the total dataset
dataset = ImageFolder(dir, transform = transform)
print(len(dataset))

#Define sizes for training, testing and validation
train_size = int(0.7 * len(dataset))
val_size = test_size = (len(dataset) - train_size) // 2
print(train_size, val_size,test_size)

#Create the datasets for training, validation and testing
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

#Display the first image in the dataset
def display_img(img,label):
    print(f"Label : {dataset.classes[label]}")
    plt.imshow(img.permute(1,2,0))
    plt.show()
#display_img(*dataset[0])

#Set hyperparameters
batch_size = 14
epochs = 30
loss_fn = nn.CrossEntropyLoss()
lr = 0.001

#Load the train and validation data into batches
train_loader = DataLoader(train_dataset, batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size) #idk why *2
test_loader = DataLoader(test_dataset, batch_size)

#Construct and initialize the model
INPUT_COLS = 256
OUTPUT_COLS = 256

layers = [INPUT_COLS, 50, 50, OUTPUT_COLS]
net = Net(layers)

print(f'Layers: {layers}')
print(f'Number of model parameters: {net.get_num_parameters()}')

n_epochs = 100
lr = 0.001
l2_reg = 0.001  # 10

net = train(net, train_loader, val_loader, n_epochs, lr, l2_reg)