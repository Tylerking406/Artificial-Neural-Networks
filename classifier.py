
import torch
import torch.nn as nn
from torchvision import datasets
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
import random
import torchvision.transforms as transforms

# Hyper Parameters
inputSize = 784 # 24x24
hiddenSize = 50 # try dif sizes
num_classes = 100
num_epochs =  36 # to be changed
batch_size = 100
lr = 0.001

transform = transforms.Compose([transforms.ToTensor()])

# Load MNIST from file
DATA_DIR = "C:/Users/Student/OneDrive - University of Cape Town/Documents/2024/csc3022F/TUT/ML/TUT1"
download_dataset = False

train_mnist = datasets.MNIST(DATA_DIR, train=True, download=download_dataset, transform=transform)
test_mnist = datasets.MNIST(DATA_DIR, train=False, download=download_dataset, transform=transform)

# Shapes of train and test set
print('train_mnist.shape: ',train_mnist.data.shape)
print('test_mnist.shape: ',test_mnist.data.shape)

