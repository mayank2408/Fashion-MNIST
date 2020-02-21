"""
Code to use the saved models for testing
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
from sklearn.metrics import confusion_matrix

from utils import AverageMeter

import pickle


def test(model, testloader):
    """ Training the model using the given dataloader for 1 epoch.

    Input: Model, Dataset, optimizer,
    """

    model.eval()
    avg_loss = AverageMeter("average-loss")

    y_gt = []
    y_pred_label = []

    for batch_idx, (img, y_true) in enumerate(testloader):
        img = Variable(img)
        y_true = Variable(y_true)
        out = model(img)
        y_pred = F.softmax(out, dim=1)
        y_pred_label_tmp = torch.argmax(y_pred, dim=1)

        loss = F.cross_entropy(out, y_true)
        avg_loss.update(loss.data, img.shape[0])

        # Add the labels
        y_gt += list(y_true.numpy())
        y_pred_label += list(y_pred_label_tmp.numpy())

    return avg_loss.avg, y_gt, y_pred_label


if __name__ == "__main__":

    trans_img =  transforms.Compose([transforms.ToTensor(),
    								 transforms.Normalize([0.5],[0.5])])
    dataset = FashionMNIST("./data/", train=False, transform=trans_img, download=True)
    testloader = DataLoader(dataset, batch_size=1024, shuffle=False)

    from train_multi_layer import Neuralnet
    model_MLP = Neuralnet()
    #model_MLP.load_state_dict(torch.load("./models/MLP.pt"))
    model_MLP=pickle.load(open("models/MLP","rb"))
    from training_conv_net import CNN
    model_conv_net = CNN()
    #model_conv_net.load_s=tate_dict(torch.load("./models/convNet.pt"))
    model_conv_net=pickle.load(open("models/convNet","rb"))
    loss, gt, pred = test(model_MLP, testloader)
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt

    labels=np.arange(0,10)
    cm1=confusion_matrix(gt,pred,labels)
    df_cm = pd.DataFrame(cm1, index = [i for i in dataset.classes],
	                     columns = [i for i in dataset.classes])
    plt.figure(figsize = (15,11))
    sn.heatmap(df_cm, annot=True)
    plt.title("Confusion Matrix MLP")
    plt.savefig("./img/confusion_matrix_mlp.jpg")

    
    #print(cm1)
    with open("multi-layer-net.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))

    loss, gt, pred = test(model_conv_net, testloader)
    labels=np.arange(0,10)
    cm2=confusion_matrix(gt,pred,labels)
    df_cm = pd.DataFrame(cm2, index = [i for i in dataset.classes],
	                     columns = [i for i in dataset.classes])
    plt.figure(figsize = (15,11))
    sn.heatmap(df_cm, annot=True)
    plt.title("Confusion Matrix convNet")
    plt.savefig("./img/confusion_matrix_convNet.jpg")
    with open("convolution-neural-net.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))
