
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

