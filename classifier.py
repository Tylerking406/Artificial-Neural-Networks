
import torch
import torch.nn as nn
from torchvision import datasets
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
import random
import torchvision.transforms as transforms
import os
from torch.nn import functional as F

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

transform = transforms.Compose([transforms.ToTensor()])

# Load MNIST from file
DATA_DIR = "C:/Users/Student/OneDrive - University of Cape Town/Documents/2024/csc3022F/TUT/ML/TUT1"
download_dataset = False

train_mnist = datasets.MNIST(DATA_DIR, train=True, download=download_dataset, transform=transform)
test_mnist = datasets.MNIST(DATA_DIR, train=False, download=download_dataset, transform=transform)

# Shapes of train and test set
print('train_mnist.shape: ',len(train_mnist))
print('test_mnist.shape: ',test_mnist.data.shape)

# Make Dataset Iterable
batch_size = 64
iterations = 3000
num_epochs = iterations / (len(train_mnist) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_mnist, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_mnist, batch_size=batch_size, shuffle=False)

# feedforward neural network Model
class FeedForwardNeutralNet(nn.Module):
    def __init__(self, input_Size, hidden_size,output_size):
        super(FeedForwardNeutralNet,self).__init__(input_Size,hidden_size,output_size)
        
        # Linear function
        self.fully_connected_1 = nn.Linear(input_Size, hidden_size)  # Layer 1
        
        #Non Linear
        self.sigmoid = nn.Sigmoid # Layer 2 = Activation function
        
        # Linear function
        self.fully_connected_2 = nn.Linear(hidden_size,output_size) # Layer 3
    
    def forward(self,x):
        # Linear
        out = self.fc1(x)
        
        # Non Lonear
        out = self.sigmoid(out)
        
        # linear
        out = self.fc2(out)
        
        return out
    
# Instantiate the model
input_size = 28*28 # 784
hidden_size = 150
output_size = 10 # 0 - 9 

model = FeedForwardNeutralNet(input_size,hidden_size,output_size)   