
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

# Data Loaders
batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset=train_mnist, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_mnist, batch_size=batch_size, shuffle=False)

# Iterate over the data loader
for images, labels in train_loader:
    print(images.shape, labels.shape)
    break  # Only print the shape of the first batch for demonstration

# Plot the first 6 samples
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i][0], cmap='gray')
#plt.show()

