
import torch
import torch.nn as nn
from torchvision import datasets
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
import random
import torchvision.transforms as transforms
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Hyper Parameters
inputSize = 784 # 24x24
hiddenSize = 50 # try dif sizes
num_classes = 100
num_epochs =  36 # to be changed
batch_size = 64
lr = 0.001

transform = transforms.Compose([transforms.ToTensor()])

# Load MNIST from file
DATA_DIR = "C:/Users/Student/OneDrive - University of Cape Town/Documents/2024/csc3022F/TUT/ML/TUT1"
download_dataset = False

mnist_real_train = datasets.MNIST(DATA_DIR, train=True, download=download_dataset, transform=transform)
test_mnist = datasets.MNIST(DATA_DIR, train=False, download=download_dataset, transform=transform)

train_mnist, mnist_validation = torch.utils.data.random_split(mnist_real_train, (48000, 12000)) # train = 48000 validation set = 12 000

# Shapes of train and test set
print('train_mnist.shape: ',len(train_mnist))
print('test_mnist.shape: ',test_mnist.data.shape)

# Data Loaders
train_loader = torch.utils.data.DataLoader(dataset=train_mnist, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_mnist, batch_size=batch_size, shuffle=False)
valid_loader = torch.utils.data.DataLoader(dataset=mnist_validation, batch_size=batch_size, shuffle=False)

# Iterate over the data loader
for images, labels in train_loader:
    print(images.shape, labels.shape)
    break  # Only print the shape of the first batch for demonstration

# Plot the first 6 samples
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i][0], cmap='gray')
plt.show()

