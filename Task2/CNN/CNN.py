import sys
sys.path.append("./")
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from preprocess import DataProcess
from torch.utils.data import Dataset, DataLoader
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np

import csv
data = ('./Test_Data.csv') 
print(data)
# Recall the architecture of the CNN
#data = pd.read_csv("D:/Data Science/Big data computing/train_attrs.csv")
#print(data.T)
filepath="./Train_Data.csv"
unused_attrs=['unit price of residence space','unit price of building space','city',
              'total cost','district','zip code','region']
data_int=DataProcess(filepath,unused_attrs)
data_int.getdata()

class CsvDataset():
    def __init__(self):
        super(CsvDataset, self).__init__()
 
        self.feature_path = './train_attrs.csv'
        self.label_path = './train_labels.csv'
 
        feature_df_ = pd.read_csv(self.feature_path).T
        label_df_ = pd.read_csv(self.label_path)
 
        #assert feature_df_.columns.tolist()[1:] == label_df_[label_df_.columns[0]].tolist(), \
         #   'feature name does not match label name'
 
        self.feature = [feature_df_[i].tolist() for i in feature_df_.columns[:]]
 
        self.label  = label_df_['labels']
 
        #assert len(self.feature) == len(self.label)
 
        self.length = len(self.feature)
 
    def __getitem__(self, index):
        x = self.feature[index]
        x = torch.Tensor(x)
        x = x.reshape(1,1,17)
 
        y = self.label[index]
 
        return x, y
 
    def __len__(self):
        return self.length



train_dataset = CsvDataset()
 
trainloader = DataLoader(dataset=train_dataset, batch_size=40,  shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(1,1))
        self.pool = nn.MaxPool2d(1, 17)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(1,1))
        self.fc1 = nn.Linear(16 * 1 * 1, 10)
        self.fc2 = nn.Linear(10, 6)
        self.fc3 = nn.Linear(6, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 1 *1 )
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    '''
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)  # (B, C, H ,W)
        self.linear = nn.Linear(in_features=5 * 1 * 17, out_features=5, bias=False)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        #print("[before flatten] x.shape: {}".format(x.shape))  # torch.Size([1, 5, 12, 12])
        x = self.flatten(x)
        #print("[after flatten] x.shape: {}".format(x.shape))  # torch.Size([1, 720])
        x = self.linear(x)
        x = self.relu(x)
        return x
'''
net = Net()
import torch.optim as optim

# Use the cross entropy loss function
criterion = nn.CrossEntropyLoss()

# Use the cross SGD algorithm
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Set the epochs=10
epochs = 100


def train(net, criterion, optimizer, epochs):
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)
        print(f'Epoch {epoch + 1}: train loss = {train_loss:.3f}')
        
    plt.plot(train_losses, label='Training loss')
    #plt.plot(test_losses, label='Testing loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
train(net, criterion, optimizer, epochs)



