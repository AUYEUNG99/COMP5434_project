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
from sklearn import preprocessing
from Alexnet import AlexNet
from resnet import ResNet18


data = DataProcess("./train_data21.csv", unused_attrs=['district', 'city', 'zip code', 'region',
                                                                 'unit price of residence space',
                                                                 'unit price of building space','total cost' ])

data.getdata(normalize=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CSVDataset():
    def __init__(self,filepath):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(CSVDataset, self).__init__()
        df = pd.read_csv(filepath)
        # 获取特征和标签
        self.features = torch.tensor(df.drop(columns=['labels']).values, dtype=torch.float32)
        self.labels = torch.tensor(df['labels'].values, dtype=torch.long).view(-1)
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

dataset=CSVDataset('./train.csv')
import torch.utils.data as Data
train_dataset, test_dataset = Data.random_split(dataset, [3600, 400])
trainloader = DataLoader(dataset=train_dataset, batch_size=36,  shuffle=False) 
testloader = DataLoader(dataset=test_dataset, batch_size=8,  shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(2,2),stride=1,padding=1)
        self.pool1 =nn.AvgPool2d(2,2)
        self.bn2 = nn.BatchNorm2d(6)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc1 = nn.Linear(24,12)
        self.fc2 = nn.Linear(12,4)
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.bn2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 4)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

import torch.optim as optim
net = AlexNet().to(device)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(net.parameters(), lr=0.01)
epochs = 200

def train(net, criterion, optimizer, epochs):
   train_losses = []
   test_losses = []
   with open("./net.train/Alexnet/record2.txt", "w") as f:
    for epoch in range(epochs):
        running_loss = 0.0
        total1=0
        correct1=0
        net.train()
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total1 += labels.size(0)
            correct1 += (predicted == labels).sum().item()
        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)
        correct = 0
        total = 0
        running_loss = 0.0
        net.eval()
        with torch.no_grad():
            for data in testloader:
                #net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = running_loss / len(testloader)
        test_losses.append(test_loss)
        f.write('%d  %.3f  %.3f  %.3f  %.3f'
              % (epoch + 1, train_loss , test_loss, 100 * correct1 / total1, 100. * correct / total))
        f.write('\n')
        f.flush()
        print(f'Epoch {epoch + 1}: train loss = {train_loss:.3f}, test loss = {test_loss:.3f},train accuracy = {100 * correct1 / total1:.3f}%, test accuracy = {100 * correct / total:.3f}%')
        torch.save(net,'./net.train/Alexnet/net2.pkl')
    
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Testing loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

train(net, criterion, optimizer, epochs)
#torch.save(net,'./net.train/Alexnet/net.pkl')
data = np.loadtxt('./net.train/Alexnet/record2.txt')
fig = plt.figure(figsize=(16,9))
plt.plot(data[:,0],data[:,1],label='Train loss')
plt.plot(data[:,0],data[:,2],label='Test loss')
plt.title('CNN',fontsize=20)
plt.xlabel('Epochs',fontsize=20)
plt.ylabel('loss',fontsize=20)
plt.grid(True)
plt.legend(fontsize=20)
plt.tick_params(labelsize=20)
plt.savefig('./resnet18-loss.png')
#plt.show()
fig = plt.figure(figsize=(16,9))
plt.plot(data[:,0],data[:,3],label='Train accuracy')
plt.plot(data[:,0],data[:,4],label='Test accuracy')
plt.title('CNN',fontsize=20)
plt.xlabel('Epochs',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.grid(True)
plt.legend(fontsize=20)
plt.tick_params(labelsize=20)
plt.savefig('./resnet18-acc.png')
#plt.show()