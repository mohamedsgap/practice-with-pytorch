#for building a neural network in pytorch we should import some packages.
# Import necessary packages

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import numpy as np
import torch

import helper
import matplotlib.pyplot as plt

"""
Now we're going to build a larger network that can solve a difficult problem,
 identifying text in an image. Here we'll use 
 the MNIST dataset which consists of greyscale handwritten digits. 
 Each image is 28x28 pixels, you can see a sample below

Our goal is to build a neural network that can 
take one of these images and predict the digit in the image.
First up, we need to get our dataset. 
This is provided through the torchvision package. 
The code below will download the MNIST dataset, 
then create training and test datasets for us. 
Don't worry too much about the details here, 
you'll learn more about this later.
"""

# import some packages and downlaod mnist dataset 

from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

"""
We have the training data loaded into trainloader 
and we make that an iterator with iter(trainloader). 
Later, we'll use this to loop through the dataset for training, like

for image, label in trainloader:
    ## do things with images and labels
You'll notice I created the trainloader 
with a batch size of 64, and shuffle=True. 
The batch size is the number of images we get in one iteration 
from the data loader and pass through our network, often called a batch. 
And shuffle=True tells it to shuffle the dataset every time we 
start going through the data loader again. But here I'm just grabbing 
the first batch so we can check out the data. We can see below that images is
 just a tensor with size (64, 1, 28, 28). So, 64 images per batch,
 1 color channel, and 28x28 images.
"""
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

# This is what one of the images looks like.

plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');

"""
First, let's try to build a simple network for 
this dataset using weight matrices and matrix multiplications. 
Then, we'll see how to do it using PyTorch's nn module which
 provides a much more convenient and powerful method
 for defining network architectures.

The networks you've seen so far are called fully-connected
 or dense networks. Each unit in one layer is connected to each
 unit in the next layer. In fully-connected networks, 
 the input to each layer must be a one-dimensional vector 
 (which can be stacked into a 2D tensor as a batch of multiple examples). 
 However, our images are 28x28 2D tensors, 
 so we need to convert them into 1D vectors. 
 Thinking about sizes, we need to convert the batch of images 
 with shape (64, 1, 28, 28) to a have a shape of (64, 784), 784 is 28 times 28.
 This is typically called flattening, we flattened
 the 2D images into 1D vectors.

Previously you built a network with one output unit. 
Here we need 10 output units, one for each digit.
 We want our network to predict the digit shown in an image,
 so what we'll do is calculate probabilities that the image is of any one
 digit or class. This ends up being a discrete probability distribution
 over the classes (digits) that tells us the most likely class for the image. 
 That means we need 10 output units for the 10 classes (digits).
 We'll see how to convert the network output into a probability distribution next.
"""

"""
building a multi-layer network with 784 input units, 
256 hidden units, and 10 output units using random tensors for 
the weights and biases. For now, use a sigmoid activation 
for the hidden layer. Leave the output layer without an activation

"""
def activation(x):
    return 1/(1+torch.exp(-x))

# Flatten the input images
inputs = images.view(images.shape[0], -1)

# Create parameters
w1 = torch.randn(784, 256)
b1 = torch.randn(256)

w2 = torch.randn(256, 10)
b2 = torch.randn(10)

h = activation(torch.mm(inputs, w1) + b1)

out = torch.mm(h, w2) + b2


"""
Now we have 10 outputs for our network. 
We want to pass in an image to our network and get out 
a probability distribution over the classes that tells us 
the likely class(es) the image belongs to. Something that looks like this: 

Here we see that the probability for each class is roughly the same. 
This is representing an untrained network, 
it hasn't seen any data yet so it just returns a uniform
 distribution with equal probabilities for each class.

To calculate this probability distribution,
 we often use the softmax function. 
What this does is squish each input  𝑥𝑖  between 0 and 1 and normalizes
 the values to give you a proper probability distribution 
 where the probabilites sum up to one.

Exercise: Implement a function softmax that performs
 the softmax calculation and returns probability distributions
 for each example in the batch. Note that you'll need to pay attention
 to the shapes when doing this. If you have a tensor a with shape (64, 10)
 and a tensor b with shape (64,), doing a/b will give you an error because 
 PyTorch will try to do the division across the columns (called broadcasting) 
 but you'll get a size mismatch. The way to think about this is for each of 
 the 64 examples, you only want to divide by one value,
 the sum in the denominator. So you need b to have a shape of (64, 1).
 This way PyTorch will divide the 10 values in each row of a by the one value 
 in each row of b. Pay attention to how you take the sum as well. 
 You'll need to define the dim keyword in torch.sum.
 Setting dim=0 takes the sum across the rows while dim=1 takes 
 the sum across the columns.

"""

def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)

probabilities = softmax(out)

# Does it have the right shape? Should be (64, 10)
print(probabilities.shape)
# Does it sum to 1?
print(probabilities.sum(dim=1))


"""
PyTorch provides a module nn that makes building networks much simpler. 
Here I'll show you how to build the same one as above with 784 inputs, 
256 hidden units, 10 output units and a softmax output.
"""

from torch import nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x
    
"""
Let's go through this bit by bit.

class Network(nn.Module):
Here we're inheriting from nn.Module. 
Combined with super().__init__() this creates a class that tracks the
 architecture and provides a lot of useful methods and attributes. 
 It is mandatory to inherit from nn.Module when you're creating a 
 class for your network. The name of the class itself can be anything.

self.hidden = nn.Linear(784, 256)
This line creates a module for a linear transformation, 
𝑥𝐖+𝑏 , with 784 inputs and 256 outputs and assigns it to self.hidden. 
 The module automatically creates the weight and bias tensors
 which we'll use in the forward method. You can access the weight and bias
 tensors once the network (net) is created with net.hidden.weight and net.hidden.bias.

self.output = nn.Linear(256, 10)
Similarly, this creates another linear transformation with 256 inputs and 10 outputs.

self.sigmoid = nn.Sigmoid()
self.softmax = nn.Softmax(dim=1)
Here I defined operations for the sigmoid activation and softmax output.
 Setting dim=1 in nn.Softmax(dim=1) calculates softmax across the columns.

def forward(self, x):
PyTorch networks created with nn.Module must have a forward method defined. 
It takes in a tensor x and passes it through the operations you defined in the __init__ method.

x = self.hidden(x)
x = self.sigmoid(x)
x = self.output(x)
x = self.softmax(x)
Here the input tensor x is passed through each operation a reassigned to x.
 We can see that the input tensor goes through the hidden layer, 
 then a sigmoid function, then the output layer, and finally the softmax function.
 It doesn't matter what you name the variables here,
 as long as the inputs and outputs of the operations match
 the network architecture you want to build. 
 The order in which you define things in the __init__ method doesn't matter, 
 but you'll need to sequence the operations correctly in the forward method.

Now we can create a Network object.

"""    
# Create the network and look at it's text representation
model = Network()
model

"""
You can define the network somewhat more concisely and clearly
 using the torch.nn.functional module.
 This is the most common way you'll see networks defined as
 many operations are simple element-wise functions.
 We normally import this module as F, import torch.nn.functional as F
"""

import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.sigmoid(self.hidden(x))
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)
        
        return x

"""
Activation functions
So far we've only been looking at the softmax activation, 
but in general any function can be used as an activation function.
 The only requirement is that for a network to approximate a non-linear function,
 the activation functions must be non-linear. 
 Here are a few more examples of common activation functions:
     Tanh (hyperbolic tangent), and ReLU (rectified linear unit).
""" 
 
"""  
Create a network with 784 input units,
 a hidden layer with 128 units and a ReLU activation, 
 then a hidden layer with 64 units and a ReLU activation,
 and finally an output layer with a softmax activation as shown above.
 You can use a ReLU activation with the nn.ReLU module or F.relu function.    
"""

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Defining the layers, 128, 64, 10 units each
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        
        return x

model = Network()
model

"""
The weights and such are automatically initialized for you,
 but it's possible to customize how they are initialized. 
 The weights and biases are tensors attached to the layer you defined,
 you can get them with model.fc1.weight for instance.
"""

print(model.fc1.weight)
print(model.fc1.bias)

# Set biases to all zeros
model.fc1.bias.data.fill_(0)

# sample from random normal with standard dev = 0.01
model.fc1.weight.data.normal_(std=0.01)

# Grab some data 
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels) 
images.resize_(64, 1, 784)
# or images.resize_(images.shape[0], 1, 784) to automatically get batch size

# Forward pass through the network
img_idx = 0
ps = model.forward(images[img_idx,:])

img = images[img_idx]
helper.view_classify(img.view(1, 28, 28), ps)

# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
print(model)

# Forward pass through the network and display output
images, labels = next(iter(trainloader))
images.resize_(images.shape[0], 1, 784)
ps = model.forward(images[0,:])
helper.view_classify(images[0].view(1, 28, 28), ps)

print(model[0])
model[0].weight

"""
You can also pass in an OrderedDict to name the individual layers and operations,
 instead of using incremental integers. Note that dictionary keys must be unique,
 so each operation must have a different name.
"""

from collections import OrderedDict
model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('output', nn.Linear(hidden_sizes[1], output_size)),
                      ('softmax', nn.Softmax(dim=1))]))
model

# Now you can access layers either by integer or the name
print(model[0])
print(model.fc1)


    







