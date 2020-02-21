"""
Starter Code in Pytorch for training a multi layer neural network. 

** Takes around 30 minutes to train. 
"""

import numpy as np
import pdb
import os
from tqdm import tqdm

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.datasets import FashionMNIST
from torchvision import transforms

from utils import AverageMeter


"""
---------------------------------------------------------------------------------------------------
----------------------------------------- Model - Pytorch -----------------------------------------
---------------------------------------------------------------------------------------------------
"""



class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(1,20,5)
        #self.bn1=nn.BatchNorm2d(20)
        self.conv2=nn.Conv2d(20,40,3)
        self.bn2=nn.BatchNorm2d(40)
        self.conv3=nn.Conv2d(40,80,3)
        self.bn3=nn.BatchNorm2d(160)
        self.conv4=nn.Conv2d(80,160,3)
        self.conv5=nn.Conv2d(160,200,3)
        self.fc2=nn.Linear(200,200)
        self.fc3=nn.Linear(200,10)
        self.pool=nn.MaxPool2d(2, 2)
        self.pool2=nn.MaxPool2d(3, 2)
    def forward(self,x):
        x=x.view(-1,1,28,28)
        x=(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.bn2(self.conv2(x))))
        x=(F.relu(self.conv3(x)))
        x=self.pool2(F.relu(self.bn3(self.conv4(x))))
        x=(F.relu(self.conv5(x)))
        x=x.view(-1,200)
        #x=F.dropout(F.relu(self.fc1(x)),p=0.1)
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x


"""
---------------------------------------------------------------------------------------------------
----------------------------------------- Training - Pytorch -----------------------------------------
---------------------------------------------------------------------------------------------------
"""


def train_one_epoch(model, trainloader, optimizer, device,scheduler):
    """ Training the model using the given dataloader for 1 epoch.

    Input: Model, Dataset, optimizer, 
    """

    model.train()
    avg_loss = AverageMeter("average-loss")
    for batch_idx,(img, target) in enumerate(trainloader):
        #img_gauss=torch.tensor(random_noise(img, mode='gaussian', mean=0, var=0.05, clip=True)).float()
        img = Variable(img).to(device)
        target = Variable(target).to(device)

        # Zero out the gradients
        optimizer.zero_grad()

        # Forward Propagation
        out = model(img)
        loss = F.cross_entropy(out, target)

        # backward propagation
        loss.backward()
        avg_loss.update(loss.data, img.shape[0])

        # Update the model parameters
        optimizer.step()
    scheduler.step()
    print('loss: ',avg_loss)

    return avg_loss.avg


if __name__ == "__main__":

    number_epochs = 20
    device = torch.device('cpu')
    # Use torch.device("cuda:0") if you want to train on GPU
    # OR Use torch.device("cpu") if you want to train on CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CNN().to(device)

    trans_img =  transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.5],[0.5])])
    dataset = FashionMNIST("./data/", train=True, transform=trans_img, download=True)
    train_ds, valid_ds = torch.utils.data.random_split(dataset, (55000, 5000))
    train_loader = DataLoader(train_ds,batch_size=64,shuffle=True,num_workers=4)
    val_loader = DataLoader(valid_ds,batch_size=32,shuffle=False,num_workers=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)

    optimizer = optim.Adam(model.parameters(), lr=0.005)

    track_loss = []
    for i in tqdm(range(number_epochs)):
        loss = train_one_epoch(model, train_loader, optimizer, device,exp_lr_scheduler)
        track_loss.append(loss)

    plt.figure()
    plt.plot(track_loss)
    plt.title("training-loss-ConvNet")
    plt.savefig("./img/training_convnet.jpg")

    torch.save(model.state_dict(), "./models/convNet11.pt")
