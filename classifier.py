
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import os
import torch.utils.data as data


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # will be useful if you use matplot

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and standard deviation of MNIST dataset
])

# Load MNIST from file
DATA_DIR = "C:/Users/Student/OneDrive - University of Cape Town/Documents/2024/csc3022F/TUT/ML/TUT1"
download_dataset = False

train_mnist = datasets.MNIST(DATA_DIR, train=True, download=download_dataset, transform=transform)
test_mnist = datasets.MNIST(DATA_DIR, train=False, download=download_dataset, transform=transform)
train_mnist, valid_mnist = data.random_split(train_mnist, (48000, 12000))

# # Shapes of train and test set
# print('train_mnist.shape: ',len(train_mnist))
# print('test_mnist.shape: ',test_mnist.data.shape)
# print("Valid:", len(valid_mnist))

# Make Dataset Iterable
batch_size = 64
iterations = 3750
num_epochs = iterations / (len(train_mnist) / batch_size) # 5
num_epochs = int(num_epochs) - 2

train_loader = torch.utils.data.DataLoader(dataset=train_mnist, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_mnist, batch_size=batch_size, shuffle=False)
validation_loader = torch.utils.data.DataLoader(dataset=valid_mnist, batch_size=batch_size, shuffle=False)

# feedforward neural network Model
class FeedForwardNeutralNet(nn.Module):
    def __init__(self, input_Size, hidden_size,output_size):
        super(FeedForwardNeutralNet,self).__init__()
        
        # Linear function 1
        self.fully_connected_1 = nn.Linear(input_Size, hidden_size)  # Layer 1
        #Non Linear
        self.relu1 = nn.ReLU() #  Activation function
        
        # Linear function 2
        self.fully_connected_2 = nn.Linear(hidden_size,hidden_size) # Layer 3
        # Non Linear
        self.relu2 = nn.ReLU()
        
        #Linear Function 3
        self.fully_connected_3 = nn.Linear(hidden_size,hidden_size)
        self.relu3 = nn.ReLU()
        
        # Linear 4(output)
        self.fully_connected_4 = nn.Linear(hidden_size,output_size)
        
        
    
    def forward(self,x):
        # Linear
        out = self.fully_connected_1(x)
        out = self.relu1(out)
        
        # linear
        out = self.fully_connected_2(out)
        out = self.relu2(out)
        
        # Linear
        out = self.fully_connected_3(out)
        out = self.relu3(out)
        
        #Linear
        out = self.fully_connected_4(out)
        
        return out
    
# Instantiate the model
input_size = 28*28 # 784
hidden_size = 200
output_size = 10 # 0 - 9 

model = FeedForwardNeutralNet(input_size,hidden_size,output_size)

loss_class = nn.CrossEntropyLoss()   #Objective Function

#  Instantiate Optimizer Class
lr = 0.001
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)


# Train the model
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

         # Load images with gradient accumulation capabilities
        images = images.view(-1, 28*28).requires_grad_()

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        loss = loss_class(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        if (i+1) % 50 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            # Compute validation accuracy
            correct = 0
            total = 0
            for images, labels in validation_loader:
                # Load images with gradient accumulation capabilities
                images = images.view(-1, 28*28).requires_grad_()
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            accuracy = 100 * correct.item() / total

            print('Validation Accuracy: {} %'.format(accuracy))

# Save the trained model
save_model = False
if save_model is False:
    # Saves only parameters
    torch.save(model.state_dict(), 'my_Model.pkl')
    save_model = True
    print("Model Saved!!")
    
   