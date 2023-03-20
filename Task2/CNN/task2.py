import sys
sys.path.append("./")
import torch
import torchvision
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.version.cuda)
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
#import cv2

#data = ('./Test_Data.csv') 
#print(data)
# Recall the architecture of the CNN
#data = pd.read_csv("D:/Data Science/Big data computing/train_attrs.csv")
#print(data.T)
filepath = "./train_data_final.csv"
unused_attrs=['unit price of residence space','unit price of building space','city',
              'total cost','district','zip code','region','date']#,'waterfront','building space','city','exchange rate','decoration year','building year','air quality level']
data_int=DataProcess(filepath,unused_attrs)
data_int.getdata()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df1 = pd.read_csv("./train.csv")
df2 = pd.read_csv("./augmented_data.csv")
df=pd.concat([df1,df2],axis=0)
print(df)
df.to_csv('data_train.csv',index=False)


class CSVDataset():
    def __init__(self,filepath):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(CSVDataset, self).__init__()
        '''
        self.feature_path = './train_attrs.csv'
        self.label_path = './train_labels.csv'
        self.feature_df= torch.tensor(pd.read_csv(self.feature_path).values,dtype=torch.float32)
        self.label_df= torch.tensor(pd.read_csv(self.label_path).values,dtype=torch.long).view(-1)
        '''
        df = pd.read_csv(filepath)
        # 获取特征和标签
        self.features = torch.tensor(df.drop(columns=['labels']).values, dtype=torch.float32)
        self.labels = torch.tensor(df['labels'].values, dtype=torch.long).view(-1)
        #self.feature = [feature_df_[i].tolist() for i in feature_df_.columns[:]]
        #self.label  = label_df_
        #self.label = [label_df_[i].tolist() for i in label_df_.columns[:]]
        self.length = len(self.labels)
 
    def __getitem__(self, index):
        x = self.features[index].cuda()
        #x = torch.Tensor(x)
        x = x.reshape(1,4,4)
        y = self.labels[index].cuda()
        #y = torch.Tensor(y)
        return x, y
 
    def __len__(self):
        return self.length
    

#train_dataset= CSVDataset('./augmented_data.csv')
#test_dataset = CSVDataset('./train.csv')
dataset=CSVDataset('./train.csv')
import torch.utils.data as Data
train_dataset, test_dataset = Data.random_split(dataset, [3200, 800])
print(train_dataset)
trainloader = DataLoader(dataset=train_dataset, batch_size=100,  shuffle=False) 
testloader = DataLoader(dataset=test_dataset, batch_size=8,  shuffle=False)
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(outchannel * self.expansion)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)#open shortcut connection
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=5):
        super(ResNet, self).__init__()
        self.inchannel = 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 3,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 6, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 12, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 24, 2, stride=2)
        self.fc1 = nn.Linear(24, 12)
        self.fc2 = nn.Linear(12, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        #out = self.dropout(out)
        out = self.layer1(out)
        #out = self.dropout(out)
        out = self.layer2(out)
       # out = self.dropout(out)
        out = self.layer3(out)
        #out = self.dropout(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out,1)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def ResNet18():
    return ResNet(ResidualBlock)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(2,2),stride=1,padding=1)
        self.pool1 =nn.MaxPool2d(2,2)
        #self.pool2  = nn.Maxpool2d(1,2)
        #self.conv2 = nn.Conv2d(10, 20, kernel_size=(1,1))
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc1 = nn.Linear(24,12)
        #self.fc2 = nn.Linear(16,8)
        self.fc2 = nn.Linear(12,5)
        #self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        #x = self.pool2(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 1 *1 )
        x = self.flatten(x)
        #x = self.dropout(x)
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        #x = F.softmax(self.fc2(x))
        #x = self.dropout(x)
        x = self.fc2(x)
        return x

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(16, 5)
        #self.fc2 = nn.Linear(8, 5)
        #self.fc4 = nn.Linear(6, 5)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        x = self.flatten(x)
        #x = F.relu(self.fc1(x))
        x = self.fc1(x)
        return x

import torch.optim as optim
net = Net().to(device)
#net = torch.load('net.pkl')
# Use the cross entropy loss function
criterion = nn.CrossEntropyLoss()
#criterion=nn.BCELoss()
# Use the cross SGD algorithm
#optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9,weight_decay=1e-5)
optimizer = optim.Adam(net.parameters(), lr=0.01)
#optimizer = optim.RMSprop(net.parameters(), lr=0.02)
# Set the epochs=10
# Set the epochs=10
epochs = 10000

def train(net, criterion, optimizer, epochs):
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        total=0
        correct=0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            #optimizer.zero_grad()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            #print(predicted)
            #print(labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)
        print(f'train accuracy = {100 * correct / total:.2f}%')
        # Test the model on the test set
        # Test the model on the test set
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                #print(outputs)
                _, predicted = torch.max(outputs.data, 1)
                #print(predicted)
                #print(labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = running_loss / len(testloader)
        test_losses.append(test_loss)

        print(f'Epoch {epoch + 1}: train loss = {train_loss:.3f}, test loss = {test_loss:.3f}, test accuracy = {100 * correct / total:.2f}%')

    
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Testing loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

train(net, criterion, optimizer, epochs)
#torch.save(net,'./net.pkl')


