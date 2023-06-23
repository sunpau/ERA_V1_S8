import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import GetCorrectPredCount
from tqdm import tqdm



def train_model(model, device, train_loader, optimizer,criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()  # zero the gradients- not to use perious gradients

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()   #updates the parameter - gradient descent
    
    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
  train_acc = 100*correct/processed
  train_loss = train_loss/len(train_loader)
  return train_acc, train_loss
  

def test_model(model, device, test_loader, criterion):
    model.eval() #set model in test (inference) mode

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_acc, test_loss


#----------------S6 Assignment-----------------------------
# #This class contains the architecture of the neural network
# class Net(nn.Module):
#   def __init__(self):
#     super(Net, self).__init__()
          
#     self.conv1 = nn.Sequential(
#         nn.Conv2d(1, 8, 3, bias=False), #Image Input: 28x28x1 -> 26x26x8  #Receptive Field 1 -> 3
#         nn.ReLU(),
#         nn.BatchNorm2d(8),
#         nn.Conv2d(8, 16, 3, bias=False), #Input: 26x26x8 -> 24x24x16  #Receptive Field 3 -> 5
#         nn.ReLU(),
#         nn.BatchNorm2d(16),
#         nn.Conv2d(16, 16, 3, bias=False), #Input: 24x24x16 -> 22x22x16  #Receptive Field  5-> 7
#         nn.ReLU(),
#         nn.BatchNorm2d(16),
#         nn.Conv2d(16, 32, 3, bias=False), #Input: 22x22x16 -> 20x20x32 #Receptive Field  7-> 9
#         nn.ReLU(),
#         nn.BatchNorm2d(32),
#         #Transition Block = MaxPool + 1x1 Convolution
#         nn.MaxPool2d(2, 2),    #Input: 20x20x32 -> 10x10x32  #Receptive Field  9 -> 10
#         nn.Conv2d(32, 8, 1, bias=False),   #Input: 10x10x32 -> 10x10x8  #Receptive Field  10 -> 10
#         nn.ReLU(),
#         nn.Dropout(0.1)
#     )
          
#     self.conv2 = nn.Sequential(
#         nn.Conv2d(8, 8, 3, bias=False),  #Input: 10x10x8 -> 10x10x8  #Receptive Field  10 -> 14
#         nn.ReLU(),
#         nn.BatchNorm2d(8),
#         nn.Conv2d(8, 16, 3, bias=False),  #Input: 10x10x8 -> 10x10x16  #Receptive Field  14 -> 18
#         nn.ReLU(),
#         nn.BatchNorm2d(16),
#         nn.Conv2d(16, 16, 3, bias=False),  #Input: 10x10x16 -> 10x10x16  #Receptive Field  18 -> 22
#         nn.ReLU(),
#         nn.BatchNorm2d(16),
#         nn.Conv2d(16, 32, 3, bias=False),  #Input: 10x10x16 -> 10x1032  #Receptive Field  22 -> 26
#         nn.ReLU(),
#         nn.BatchNorm2d(32),
#         nn.Dropout(0.1)
#     )
#     self.conv3 = nn.Sequential(
#         nn.Conv2d(32, 10, 1, bias=False),  #GAP implementation - 10 Classes to be predicted so 10 feature map is generated
#         nn.AdaptiveAvgPool2d((1,1))  #Average is calculated for each Feature Map.
        
#     )
          
          
#   def forward(self, x):
#     x = self.conv1(x)
#     x = self.conv2(x)
#     x = self.conv3(x)

#     x = x.view(-1, 10) 
#     x = F.log_softmax(x, dim=1)
#     return x


#----------------------S7 Assignment models--------------
# #------------- Model 1-------------
# #Skeleton formed
# class Net(nn.Module):
  # def __init__(self):
    # super(Net, self).__init__()
    # self.conv1 = nn.Sequential(
        # nn.Conv2d(1, 16, 3, bias=False, padding=1), #Image Input: 28x28x1 -> 28x28x16  #Receptive Field 1 -> 3
        # nn.ReLU(),
        # nn.Conv2d(16, 32, 3, bias=False), #Input: 28x28x16 -> 26x26x32  #Receptive Field 3 -> 5
        # nn.ReLU(),
        # #Transition Block = MaxPool + 1x1 Convolution
        # #nn.Conv2d(16, 8, 1, bias=False),
        # nn.MaxPool2d(2, 2),    #Input: 13x13x32 -> 13x13x32  #Receptive Field  5 -> 6
        # nn.Conv2d(32, 16, 1, bias=False)   #Input: 13x13x32 -> 13x13x16  #Receptive Field  6 -> 6
    # )

    # self.conv2 = nn.Sequential(
        # nn.Conv2d(16, 16, 3, bias=False),  #Input: 13x13x16 -> 11x11x16  #Receptive Field  6 -> 10
        # nn.ReLU(),
        # nn.Conv2d(16, 32, 3, bias=False),  #Input: 11x11x16 -> 9x9x32  #Receptive Field  10 -> 14
        # nn.ReLU()
    # )
    # self.conv3 = nn.Sequential(
        # nn.Conv2d(32, 16, 3, bias=False),  #Input: 9x9x32 -> 7x7x16  #Receptive Field  14 -> 18
        # nn.ReLU(),
        # nn.Conv2d(16, 16, 3, bias=False),  #Input: 7x7x16 -> 5x5x16  #Receptive Field  18 -> 22
        # nn.ReLU()
    # )
    # self.conv4 = nn.Sequential(
        # nn.Conv2d(16, 10, kernel_size=(5, 5), bias=False) #Input: 5x5x16 -> 1x1x10  #Receptive Field  22 -> 30
        # )

  # def forward(self, x):
    # x = self.conv1(x)
    # x = self.conv2(x)
    # x = self.conv3(x)
    # x = self.conv4(x)
    # x = x.view(-1, 10)
    # x = F.log_softmax(x, dim=1)
    # return x
	
	
# #------------- Model 2-------------
# #Lighter Model
# class Net(nn.Module):
  # def __init__(self):
    # super(Net, self).__init__()
    # self.conv1 = nn.Sequential(
        # nn.Conv2d(1, 8, 3, bias=False, padding=1), #Image Input: 28x28x1 -> 28x28x16  #Receptive Field 1 -> 3
        # nn.ReLU(),
        # nn.Conv2d(8, 16, 3, bias=False), #Input: 28x28x16 -> 26x26x32  #Receptive Field 3 -> 5
        # nn.ReLU(),
        # #Transition Block = MaxPool + 1x1 Convolution
        # #nn.Conv2d(16, 8, 1, bias=False),
        # nn.MaxPool2d(2, 2),    #Input: 13x13x32 -> 13x13x32  #Receptive Field  5 -> 6
        # nn.Conv2d(16, 8, 1, bias=False)   #Input: 13x13x32 -> 13x13x16  #Receptive Field  6 -> 6
    # )

    # self.conv2 = nn.Sequential(
        # nn.Conv2d(8, 8, 3, bias=False),  #Input: 13x13x16 -> 11x11x16  #Receptive Field  6 -> 10
        # nn.ReLU(),
        # nn.Conv2d(8, 8, 3, bias=False),  #Input: 11x11x16 -> 9x9x32  #Receptive Field  10 -> 14
        # nn.ReLU()
    # )
    # self.conv3 = nn.Sequential(
        # nn.Conv2d(8, 16, 3, bias=False),  #Input: 9x9x32 -> 7x7x16  #Receptive Field  14 -> 18
        # nn.ReLU(),
        # nn.Conv2d(16, 16, 3, bias=False),  #Input: 7x7x16 -> 5x5x16  #Receptive Field  18 -> 22
        # nn.ReLU()
    # )
    # self.conv4 = nn.Sequential(
        # nn.Conv2d(16, 10, kernel_size=(5, 5), bias=False) #Input: 5x5x16 -> 1x1x10  #Receptive Field  22 -> 30
        # )

  # def forward(self, x):
    # x = self.conv1(x)
    # x = self.conv2(x)
    # x = self.conv3(x)
    # x = self.conv4(x)
    # x = x.view(-1, 10)
    # x = F.log_softmax(x, dim=1)
    # return x

# #------------- Model 3-------------	
# #Batch Norm Added
# class Net(nn.Module):
  # def __init__(self):
    # super(Net, self).__init__()
    # self.conv1 = nn.Sequential(
        # nn.Conv2d(1, 8, 3, bias=False, padding=1), #Image Input: 28x28x1 -> 28x28x16  #Receptive Field 1 -> 3
        # nn.ReLU(),
        # nn.BatchNorm2d(8),
        # nn.Conv2d(8, 16, 3, bias=False), #Input: 28x28x16 -> 26x26x32  #Receptive Field 3 -> 5
        # nn.ReLU(),
        # nn.BatchNorm2d(16),
        # #Transition Block = MaxPool + 1x1 Convolution
        # #nn.Conv2d(16, 8, 1, bias=False),
        # nn.MaxPool2d(2, 2),    #Input: 13x13x32 -> 13x13x32  #Receptive Field  5 -> 6
        # nn.Conv2d(16, 8, 1, bias=False)   #Input: 13x13x32 -> 13x13x16  #Receptive Field  6 -> 6
    # )

    # self.conv2 = nn.Sequential(
        # nn.Conv2d(8, 8, 3, bias=False),  #Input: 13x13x16 -> 11x11x16  #Receptive Field  6 -> 10
        # nn.ReLU(),
        # nn.BatchNorm2d(8),
        # nn.Conv2d(8, 8, 3, bias=False),  #Input: 11x11x16 -> 9x9x32  #Receptive Field  10 -> 14
        # nn.ReLU(),
        # nn.BatchNorm2d(8)
    # )
    # self.conv3 = nn.Sequential(
        # nn.Conv2d(8, 16, 3, bias=False),  #Input: 9x9x32 -> 7x7x16  #Receptive Field  14 -> 18
        # nn.ReLU(),
        # nn.BatchNorm2d(16),
        # nn.Conv2d(16, 16, 3, bias=False),  #Input: 7x7x16 -> 5x5x16  #Receptive Field  18 -> 22
        # nn.ReLU(),
        # nn.BatchNorm2d(16)
    # )
    # self.conv4 = nn.Sequential(
        # nn.Conv2d(16, 10, kernel_size=(5, 5), bias=False) #Input: 5x5x16 -> 1x1x10  #Receptive Field  22 -> 30
        # )

  # def forward(self, x):
    # x = self.conv1(x)
    # x = self.conv2(x)
    # x = self.conv3(x)
    # x = self.conv4(x)
    # x = x.view(-1, 10)
    # x = F.log_softmax(x, dim=1)
    # return x
	
# #------------- Model 4-------------		
# #Added Dropout
# class Net(nn.Module):
  # def __init__(self):
    # super(Net, self).__init__()
    # dropout = 0.1
    # self.conv1 = nn.Sequential(
        # nn.Conv2d(1, 8, 3, bias=False, padding=1), #Image Input: 28x28x1 -> 28x28x16  #Receptive Field 1 -> 3
        # nn.ReLU(),
        # nn.BatchNorm2d(8),
        # nn.Dropout(dropout),
        # nn.Conv2d(8, 16, 3, bias=False), #Input: 28x28x16 -> 26x26x32  #Receptive Field 3 -> 5
        # nn.ReLU(),
        # nn.BatchNorm2d(16),
        # nn.Dropout(dropout),
        # #Transition Block = MaxPool + 1x1 Convolution
        # #nn.Conv2d(16, 8, 1, bias=False),
        # nn.MaxPool2d(2, 2),    #Input: 13x13x32 -> 13x13x32  #Receptive Field  5 -> 6
        # nn.Conv2d(16, 8, 1, bias=False)   #Input: 13x13x32 -> 13x13x16  #Receptive Field  6 -> 6
    # )

    # self.conv2 = nn.Sequential(
        # nn.Conv2d(8, 8, 3, bias=False),  #Input: 13x13x16 -> 11x11x16  #Receptive Field  6 -> 10
        # nn.ReLU(),
        # nn.BatchNorm2d(8),
        # nn.Dropout(dropout),
        # nn.Conv2d(8, 8, 3, bias=False),  #Input: 11x11x16 -> 9x9x32  #Receptive Field  10 -> 14
        # nn.ReLU(),
        # nn.BatchNorm2d(8),
        # nn.Dropout(dropout)
    # )
    # self.conv3 = nn.Sequential(
        # nn.Conv2d(8, 16, 3, bias=False),  #Input: 9x9x32 -> 7x7x16  #Receptive Field  14 -> 18
        # nn.ReLU(),
        # nn.BatchNorm2d(16),
        # nn.Dropout(dropout),
        # nn.Conv2d(16, 16, 3, bias=False),  #Input: 7x7x16 -> 5x5x16  #Receptive Field  18 -> 22
        # nn.ReLU(),
        # nn.BatchNorm2d(16)
    # )
    # self.conv4 = nn.Sequential(
        # nn.Conv2d(16, 10, kernel_size=(5, 5), bias=False) #Input: 5x5x16 -> 1x1x10  #Receptive Field  22 -> 30
        # )

  # def forward(self, x):
    # x = self.conv1(x)
    # x = self.conv2(x)
    # x = self.conv3(x)
    # x = self.conv4(x)
    # x = x.view(-1, 10)
    # x = F.log_softmax(x, dim=1)
    # return x

# #------------- Model 5-------------
# #GAP replaced heavy kernal
# class Net(nn.Module):
  # def __init__(self):
    # super(Net, self).__init__()
    # dropout = 0.1
    # self.conv1 = nn.Sequential(
        # nn.Conv2d(1, 8, 3, bias=False, padding=1), #Image Input: 28x28x1 -> 28x28x16  #Receptive Field 1 -> 3
        # nn.ReLU(),
        # nn.BatchNorm2d(8),
        # nn.Dropout(dropout),
        # nn.Conv2d(8, 16, 3, bias=False), #Input: 28x28x16 -> 26x26x32  #Receptive Field 3 -> 5
        # nn.ReLU(),
        # nn.BatchNorm2d(16),
        # nn.Dropout(dropout),
        # #Transition Block = MaxPool + 1x1 Convolution
        # #nn.Conv2d(16, 8, 1, bias=False),
        # nn.MaxPool2d(2, 2),    #Input: 13x13x32 -> 13x13x32  #Receptive Field  5 -> 6
        # nn.Conv2d(16, 8, 1, bias=False)   #Input: 13x13x32 -> 13x13x16  #Receptive Field  6 -> 6
    # )

    # self.conv2 = nn.Sequential(
        # nn.Conv2d(8, 8, 3, bias=False),  #Input: 13x13x16 -> 11x11x16  #Receptive Field  6 -> 10
        # nn.ReLU(),
        # nn.BatchNorm2d(8),
        # nn.Dropout(dropout),
        # nn.Conv2d(8, 8, 3, bias=False),  #Input: 11x11x16 -> 9x9x32  #Receptive Field  10 -> 14
        # nn.ReLU(),
        # nn.BatchNorm2d(8),
        # nn.Dropout(dropout)
    # )
    # self.conv3 = nn.Sequential(
        # nn.Conv2d(8, 16, 3, bias=False),  #Input: 9x9x32 -> 7x7x16  #Receptive Field  14 -> 18
        # nn.ReLU(),
        # nn.BatchNorm2d(16),
        # nn.Dropout(dropout),
        # nn.Conv2d(16, 10, 3, bias=False),  #Input: 7x7x16 -> 5x5x16  #Receptive Field  18 -> 22
        # nn.ReLU(),
        # nn.BatchNorm2d(10)
    # )
    # self.conv4 = nn.Sequential(
        # nn.AdaptiveAvgPool2d((1, 1))
        # )

  # def forward(self, x):
    # x = self.conv1(x)
    # x = self.conv2(x)
    # x = self.conv3(x)
    # x = self.conv4(x)
    # x = x.view(-1, 10)
    # x = F.log_softmax(x, dim=1)
    # return x
	
#------------- Model 6-------------	
# #Extra Layer added before GAP
# class Net(nn.Module):
#   def __init__(self):
#     super(Net, self).__init__()
#     dropout = 0.1
#     self.conv1 = nn.Sequential(
#         nn.Conv2d(1, 8, 3, bias=False, padding=1), #Image Input: 28x28x1 -> 28x28x16  #Receptive Field 1 -> 3
#         nn.ReLU(),
#         nn.BatchNorm2d(8),
#         nn.Dropout(dropout),
#         nn.Conv2d(8, 16, 3, bias=False), #Input: 28x28x16 -> 26x26x32  #Receptive Field 3 -> 5
#         nn.ReLU(),
#         nn.BatchNorm2d(16),
#         nn.Dropout(dropout),
#         #Transition Block = MaxPool + 1x1 Convolution
#         #nn.Conv2d(16, 8, 1, bias=False),
#         nn.MaxPool2d(2, 2),    #Input: 13x13x32 -> 13x13x32  #Receptive Field  5 -> 6
#         nn.Conv2d(16, 8, 1, bias=False)   #Input: 13x13x32 -> 13x13x16  #Receptive Field  6 -> 6
#     )

#     self.conv2 = nn.Sequential(
#         nn.Conv2d(8, 8, 3, bias=False),  #Input: 13x13x16 -> 11x11x16  #Receptive Field  6 -> 10
#         nn.ReLU(),
#         nn.BatchNorm2d(8),
#         nn.Dropout(dropout),
#         nn.Conv2d(8, 8, 3, bias=False),  #Input: 11x11x16 -> 9x9x32  #Receptive Field  10 -> 14
#         nn.ReLU(),
#         nn.BatchNorm2d(8),
#         nn.Dropout(dropout)
#     )
#     self.conv3 = nn.Sequential(
#         nn.Conv2d(8, 16, 3, bias=False),  #Input: 9x9x32 -> 7x7x16  #Receptive Field  14 -> 18
#         nn.ReLU(),
#         nn.BatchNorm2d(16),
#         nn.Dropout(dropout),
#         nn.Conv2d(16, 10, 3, bias=False),  #Input: 7x7x16 -> 5x5x16  #Receptive Field  18 -> 22
#         nn.ReLU(),
#         nn.BatchNorm2d(10)
#     )
#     self.conv4 = nn.Sequential(
#         nn.Conv2d(10, 10,3, bias=False), #Input: 5x5x16 -> 1x1x10  #Receptive Field  22 -> 26
#         nn.AdaptiveAvgPool2d((1, 1))
#         )

#   def forward(self, x):
#     x = self.conv1(x)
#     x = self.conv2(x)
#     x = self.conv3(x)
#     x = self.conv4(x)
#     x = x.view(-1, 10)
#     x = F.log_softmax(x, dim=1)
#     return x

    
#-----------------------S8 Assignment----------------
global GROUP_SIZE
GROUP_SIZE = 8
class Layer(nn.Module):
    """
    This class defines a convolution layer followed by
    normalization and activation function. Relu is used as activation function.
    """
    
    def __init__(self, input_size, output_size, padding=0, norm='bn'):
        """Initialize Block

        Args:
            input_size (int): Input Channel Size
            output_size (int): Output Channel Size
            padding (int, optional): Padding to be used for convolution layer. Defaults to 1.
            norm (str, optional): Type of normalization to be used. Allowed values ['bn', 'gn', 'ln']. Defaults to 'bn'.
        """
        super(Layer, self).__init__()
        
        self.conv1 = nn.Conv2d(input_size, output_size, 3, padding=padding)
        self.relu = nn.ReLU()
        if norm == 'bn':
            self.n1 = nn.BatchNorm2d(output_size)
        elif norm == 'gn':
            self.n1 = nn.GroupNorm(GROUP_SIZE, output_size)
        elif norm == 'ln':
            self.n1 = nn.GroupNorm(1, output_size)


    def __call__(self, x):
        """
        Args:
            x (tensor): Input tensor to this block
        Returns:
            tensor: Return processed tensor
        """
        x = self.conv1(x)
        x = self.n1(x)
        x = self.relu(x)
        return x
    
class DownSample(nn.Module):
    """
    This class defines a 1x1 convolution followed by an optional MaxPool layer.
    If using this class as Transition Block, set relu to true and usepool to true.
    If using this class as identity to input for skip connection, use relu=False, usepool=False
    """
    def __init__(self, input_size, output_size, stride=1, usepool=True):
        """Initialize Block

        Args:
            input_size (int): Input Channel Size
            output_size (int): Output Channel Size
            stride (int, optional): Stride to be used for convolution layer. Defaults to 1.
            usepool (bool, optional): Enable/Disable Maxpolling. Defaults to True.
        """
        super(DownSample, self).__init__()
        self.usepool = usepool
        self.conv1 = nn.Conv2d(input_size, output_size, 1, stride=stride)

        self.relu = nn.ReLU()
        if usepool:
            self.pool = nn.MaxPool2d(2, 2)

    def __call__(self, x, relu=True):
        """
        Args:
            x (tensor): Input tensor to this block
            relu (bool, optional): Use Relu for trasition block, not for resize in skip connection. Defaults to True.

        Returns:
            tensor: Return processed tensor
        """
        x = self.conv1(x)
        if relu:
            x = self.relu(x)
        if self.usepool:
            x = self.pool(x)

        return x
    
class Net(nn.Module):
    """ Network Class

    Args:
        nn (nn.Module): Instance of pytorch Module
    """

    def __init__(self, norm='bn'):
        """Initialize Network

        Args:
            drop (float, optional): Dropout value. Defaults to 0.1.
            norm (str, optional): Normalization type. Defaults to 'bn'.
        """
        super(Net, self).__init__()
        self.dropout = nn.Dropout(0.1)
        # Conv
        self.layer1 = Layer(3, 8, padding=1, norm=norm)
        self.layer2 = Layer(8, 16, padding=1, norm=norm)
        self.tb1 = DownSample(16, 8, stride=1, usepool=True)
        self.layer3 = Layer(8, 16, padding=1, norm=norm)
        self.layer4 = Layer(16, 16, padding=1, norm=norm)
        self.layer5 = Layer(16, 32, padding=1, norm=norm)
        self.tb2 = DownSample(32, 8, stride=1, usepool=True)
        self.layer6 = Layer(8, 16, norm=norm)
        self.layer7 = Layer(16, 16, norm=norm)
        self.layer8 = Layer(16, 32, norm=norm)
        #downsample
        self.downsample1 = DownSample(8, 16, stride=2, usepool=False)
        self.downsample2 = DownSample(16, 16, stride=2, usepool=False)

        self.downsample3 = DownSample(16, 8, stride=2, usepool=False)
        self.downsample4 = DownSample(16, 16, stride=3, usepool=False)
        self.downsample5 = DownSample(32, 16, stride=5, usepool=False)
        #output layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Conv2d(32, 10, 1)

    def forward(self, x):

        # Conv Layer
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x2 = self.dropout(x2)
        x = self.tb1(x2)
        x3 = self.layer3(x) + self.downsample1(x1, relu = False)
        x4 = self.layer4(x3) + self.downsample2(x2, relu = False)
        x4 = self.dropout(x4)
        x5 = self.layer5(x4)
        x5 = self.dropout(x5)
        x = self.tb2(x5) + self.downsample3(x3, relu = False)
        x6 = self.layer6(x) + self.downsample4(x4, relu = False)
        x7 = self.layer7(x6) + self.downsample5(x5, relu = False)
        x7 = self.dropout(x7)
        x8 = self.layer8(x7)
        # Output Layer
        x = self.gap(x8)
        x = self.flat(x)
        x = x.view(-1, 10)

        # Output Layer
        return F.log_softmax(x, dim=1)