
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

train_mnist = datasets.MNIST(DATA_DIR, train=True, download=download_dataset, transform=transform)
test_mnist = datasets.MNIST(DATA_DIR, train=False, download=download_dataset, transform=transform)

# Shapes of train and test set
print('train_mnist.shape: ',train_mnist.data.shape)
print('test_mnist.shape: ',test_mnist.data.shape)

# Data Loaders
train_loader = torch.utils.data.DataLoader(dataset=train_mnist, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_mnist, batch_size=batch_size, shuffle=False)


# Iterate over the data loader
for images, labels in train_loader:
    # Print the shape of the current batch
    print('Batch shape:', images.shape)

    # Plot the first 5 images from the batch
    num_images_to_display = 5
    for i in range(num_images_to_display):
        plt.subplot(1, num_images_to_display, i+1)
        plt.imshow(images[i][0], cmap='gray')
        plt.title('Label: {}'.format(labels[i].item()))
        plt.axis('off')  # Hide axes
    plt.show()

    break  # Only print the first batch