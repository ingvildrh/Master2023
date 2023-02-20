import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn.functional as F 

class ASLModel(nn.Module):
    def __init__(self, input, n_classes):
        super(ASLModel, self).__init__()
        self.n_classes = n_classes
        # Feature Extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 3, 3, stride=1, padding=1),  # b, 32, 416, 416
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 208, 208
            nn.Conv2d(3, 8, 3, stride=1, padding=1),  # b, 16, 208, 208
            nn.ReLU(True),
            nn.Conv2d(8, 8, 3, stride=1, padding=1),  # b, 16, 208, 208
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 104, 104
        )
        # Inner Representation
        self.inner = nn.Sequential(
            nn.Conv2d(8, 16, 3, stride=1, padding=1),  # b, 16, 104, 104
            nn.ReLU(True),
            nn.Conv2d(16, 1, 3, stride=1, padding=1),  # b, 1, 104, 104
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 1, 52, 52
        )
        # Bbox Extraction
        self.bbox = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2704, 1352),
            nn.ReLU()
        )
        self.bbox_xmin =  nn.Sequential(
            nn.Linear(1352, 1),
            nn.Sigmoid()
        )
        self.bbox_ymin =  nn.Sequential(
            nn.Linear(1352, 1),
            nn.Sigmoid()
        )
        self.bbox_xmax =  nn.Sequential(
            nn.Linear(1352, 1),
            nn.Sigmoid()
        )
        self.bbox_ymax =  nn.Sequential(
            nn.Linear(1352, 1),
            nn.Sigmoid()
        )
        # Class Extraction
        self.classificator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2704, 1352),
            nn.ReLU(),
            nn.Linear(1352, 676),
            nn.ReLU(),
            nn.Linear(676, self.n_classes),
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        ft_ex = self.feature_extractor(x)
        in_rep = self.inner(ft_ex)
        bbox_part = self.bbox(in_rep)
        bbox_xmin = self.bbox_xmin(bbox_part)
        bbox_ymin = self.bbox_ymin(bbox_part)
        bbox_xmax = self.bbox_xmax(bbox_part)
        bbox_ymax = self.bbox_ymax(bbox_part)
        clas = self.classificator(in_rep)
        return (bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax), clas
    
# build custom softmax module
class Softmax(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
 
    def forward(self, x):
        x = x.view(-1, 28*28)
        pred = self.linear(x)
        return pred
    

#Define the class for the NN
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

class ModifiedVGG16(nn.Module):

    def __init__(self):
        super(ModifiedVGG16, self).__init__()
        # Load pre-trained VGG16 model
        self.vgg16 = models.vgg16(pretrained=True)

        # Replace the first conv layer to accept 1 input channel instead of 3
        self.vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        # Remove max pooling layers or replace with conv layers with stride
        self.vgg16.features[4] = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.vgg16.features[5] = nn.ReLU(inplace=True)
        self.vgg16.features[6] = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.vgg16.features[7] = nn.ReLU(inplace=True)
        self.vgg16.features[8] = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.vgg16.features[9] = nn.ReLU(inplace=True)

        self.vgg16.features[10] = nn.MaxPool2d(kernel_size=2, stride=2) # remove
        self.vgg16.features[11] = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.vgg16.features[12] = nn.ReLU(inplace=True)

        # Remove or adjust more pooling layers as needed
        
        # Add a new layer at the end to reduce the number of output classes
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2)
        )

    def forward(self, x):
        x = x.view(28,28,-1)
        x = self.vgg16.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        
        # define the layers of your neural network
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # pass input through the layers of the network
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 62 * 62, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 62 * 62)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

#Trying to make a net to classify the fish
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 32 * 64 * 64)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Ingvildnet(nn.Module):
    def __init__(self):
        super(Ingvildnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        print("1---------------")
        print(x.size())
        x = self.pool(torch.relu(self.conv1(x)))
        print("2---------------")
        print(x.size())
        #x = x.view(-1, 128 * 7 * 7)
        print("3---------------")
        print(x.size())
        x = torch.relu(self.fc1(x))
        print("4---------------")
        print(x.size())
        x = self.fc2(x)
        print("5---------------")
        print(x.size())
        return x
    
class Nett(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
       
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5,1,1)
        self.conv2 = nn.Conv2d(6, 16, 5,1,1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(32, 61504)  # 5*5 from image dimension 4*4
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        print("1---------------")
        print(x.size())
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        print("2---------------")
        print(x.size())
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        print("3---------------")
        print(x.size())
        x = x.view(-1, self.num_flat_features(x))
        print("4---------------")
        print(x.size())
        x = F.relu(self.fc1(x))
        print("5---------------")
        print(x.size())
        x = F.relu(self.fc2(x))
        print("6---------------")
        print(x.size())
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

  
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

  
def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
    
    history = []
    optimizer = opt_func(model.parameters(),lr)
    for epoch in range(epochs):
        
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    
    return history

class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
            


class NaturalSceneClassification(ImageClassificationBase):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(82944,1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,6)
        )
    
    def forward(self, xb):
        return self.network(xb)
    
class Net0(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


