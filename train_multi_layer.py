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
#from skimage.util import random_noise


"""
---------------------------------------------------------------------------------------------------
----------------------------------------- Model - Pytorch -----------------------------------------
---------------------------------------------------------------------------------------------------
"""


class Neuralnet(nn.Module):
    def __init__(self):
        super(Neuralnet,self).__init__()
        self.fc1=nn.Linear(784,500)
        self.fc2=nn.BatchNorm1d(500)
        self.fc3=nn.Linear(500,200)
        self.fc4=nn.BatchNorm1d(200)
        self.fc5=nn.Linear(200,60)
        self.fc6=nn.BatchNorm1d(60)
        self.fc7=nn.Linear(60,10)
    def forward(self,x):
        x=x.view(-1,784)
        x=(self.fc1(x))
        x=F.dropout(F.relu(self.fc2(x)),p=0.0)
        x=(self.fc3(x))
        x=F.dropout(F.relu(self.fc4(x)),p=0.0)
        x=self.fc5(x)
        x=F.dropout(F.relu(self.fc6(x)),p=0.0)
        x=self.fc7(x)
        return x


"""
---------------------------------------------------------------------------------------------------
----------------------------------------- Training - Pytorch -----------------------------------------
---------------------------------------------------------------------------------------------------
"""


def train_one_epoch(model, trainloader, optimizer, device, scheduler):
    """ Training the model using the given dataloader for 1 epoch.

    Input: Model, Dataset, optimizer, 
    """

    model.train()
    avg_loss = AverageMeter("average-loss")
    for batch_idx, (img, target) in enumerate(trainloader):
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

    device = torch.device('cpu')  # Replace with torch.device("cuda:0") if you want to train on GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Neuralnet().to(device)

    trans_img =  transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.5],[0.5])])
    dataset = FashionMNIST("./data/", train=True, transform=trans_img, download=True)
    train_ds, valid_ds = torch.utils.data.random_split(dataset, (55000, 5000))
    train_loader = DataLoader(train_ds,batch_size=32,shuffle=True,num_workers=2)
    val_loader = DataLoader(valid_ds,batch_size=32,shuffle=False,num_workers=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)

    track_loss = []
    for i in tqdm(range(number_epochs)):
        loss = train_one_epoch(model, train_loader, optimizer, device,exp_lr_scheduler)
        track_loss.append(loss)

    plt.figure()
    plt.plot(track_loss)
    plt.title("training-loss-MLP")
    plt.savefig("./img/training_mlp.jpg")

    torch.save(model.state_dict(), "./models/MLP11.pt")
