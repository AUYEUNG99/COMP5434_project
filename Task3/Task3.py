#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import HousepriceDataset_V1, HousepriceDataset_V2
import numpy as np
from preprocess import DataProcess
from mlp_global_attn import DenseNet
from train import train
from sklearn import preprocessing
from torch.utils.data import DataLoader
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Attention(nn.Module):
    def __init__(self, model_dim, hidden_dim, dropout=0.1):
        super(Attention, self).__init__()

        # for simplicity, set Value dimension = hidden_dim
        self.linear = nn.Linear(model_dim, hidden_dim * 3)
        # for numerical stable
        self.temperature = model_dim ** 0.5
        self.dropout = nn.Dropout(dropout)
        # the idea of Attention: input shape = output shape
        self.fc = nn.Linear(hidden_dim, model_dim)

    def forward(self, x):
        """
        :param x: [batch, 100, model_dim]
        :return: new features after attention
        """
        q, k, v = torch.chunk(self.linear(x), chunks=3, dim=-1)  # all in shape [Batch, 100, hidden_dim]
        residual = x
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        """
        mask operation waited to be implemented
        """
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        output = self.fc(output) + residual
        # print(output.shape)
        output = torch.permute(output, [0, 2, 1])  # in form [Batch, features(channels), sequence_len] to use BatchNorm
        return output


class DenseNet(nn.Module):
    def __init__(self, d_model, embed_dim, hidden_dim, city_num, zip_num, num_classes=4):
        """
        :param d_model:        original model dimension before embedding for cities
        :param embed_dim:      embedding dim for cities (district)
        :param hidden_dim:     hidden dim we want for Attention operation
        :param city_num:       unique cities in dataset
        """
        super(DenseNet, self).__init__()

        self.hidden = hidden_dim
        self.d_model = d_model
        self.city_embed = nn.Embedding(city_num, embed_dim)
        self.zip_embed = nn.Embedding(zip_num, embed_dim)

        self.fc1 = nn.Linear(embed_dim * 2 + d_model, 2 * hidden_dim)
        self.attn1 = Attention(2 * hidden_dim, 2 * hidden_dim)
        self.bn1 = nn.BatchNorm1d(2 * hidden_dim)

        self.fc2 = nn.Linear(2 * hidden_dim, 3 * hidden_dim)
        self.attn2 = Attention(3 * hidden_dim, 2 * hidden_dim)
        self.bn2 = nn.BatchNorm1d(3 * hidden_dim)

        self.activ = nn.ReLU()

        self.fc4 = nn.Linear(3 * hidden_dim, 2 * hidden_dim)
        self.bn4 = nn.BatchNorm1d(2 * hidden_dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, cities, zipcodes):
        """
        :param x:            house data input except cities (zipcode & district are abandoned)      [Batch, num, dim]
        :param cities:       numeric value of original city attribute                               [Batch, num, embed]
        :return:
        """

        """
        1. embed the cities attribute and concat the matrix 
        2. Linear + Attention (knn based) + Activation 
        3. mlp classification head
        """

        city_embedding = self.city_embed(cities)
        zip_embedding = self.zip_embed(zipcodes)
        features = torch.concat((x, city_embedding, zip_embedding), dim=-1)  # [Batch, num, dim + embed]
        f1 = self.fc1(features)
        attn1 = self.attn1(f1)
        # attn1 = torch.permute(f1, [0, 2, 1])
        # print(attn1.shape)
        f1 = self.bn1(attn1)
        f1 = torch.permute(f1, [0, 2, 1])
        f1 = self.activ(f1)

        f2 = self.fc2(f1)
        attn2 = self.attn2(f2)
        # attn2 = torch.permute(f2, [0, 2, 1])
        f2 = self.bn2(attn2)
        f2 = torch.permute(f2, [0, 2, 1])
        f2 = self.activ(f2)

        # flatten = attn2.view(x.shape[0], -1)

        f4 = self.fc4(f2)
        f4 = torch.permute(f4, [0, 2, 1])
        f4 = self.bn4(f4)
        f4 = torch.permute(f4, [0, 2, 1])
        f4 = self.activ(f4)

        output = self.mlp_head(f4)
        return F.softmax(output, dim=-1)


def local_trainer(dataset, net, global_round, device, local_epoch):
    train_losses = []
    train_x, train_city, train_zipcodes, train_label = dataset
    train_dataset = HousepriceDataset_V2(train_x, train_city, train_label, train_zipcodes, 40)
    trainloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    total_step = len(trainloader)
    for epoch in range(local_epoch):
        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(trainloader):
            inputs, cities, zipcodes, labels = data
            inputs = inputs.to(device)
            zipcodes = zipcodes.to(device)
            cities = cities.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs, cities, zipcodes)
            outputs = torch.permute(outputs, [0, 2, 1])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, epoch, i + 1, total_step,
                                                                         loss.item()))
    return net.state_dict()

def eval(model, dataset, train: bool):
    """
    since test data doesn't have labels at all, directly use train data to measure accuracy
    """
    train_losses = []
    train_x, train_city, train_zipcodes, train_label = dataset
    train_dataset = HousepriceDataset_V2(train_x, train_city, train_label, train_zipcodes, 40)
    testloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    model.eval()
    totalNum = 0
    correctNum = 0
    criterion = nn.CrossEntropyLoss()
    losses = 0.0
    with torch.no_grad():
        for i, (x, y, z, w) in enumerate(testloader):
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            w = w.to(device)
            predict = model(x, y, z)  # [Batch, sequence_len, classes]
            predicts = torch.permute(predict, [0, 2, 1])
            loss = criterion(predicts, w)
            losses += loss.item()
            '''
            for j in range(x.shape[0]):
              totalNum += x.shape[1]
              batch_pred, batch_gt = predict[j], z[j]
              predicted = torch.argmax(batch_pred, dim=-1)
              sum = torch.sum(predicted == batch_gt)
              correctNum += sum
            '''
            totalNum += x.shape[0] * x.shape[1]
            predict = torch.argmax(predict, dim=-1)
            correctNum += torch.sum(predict == w)
    acc = correctNum / totalNum
    losses /= len(testloader)
    # print(totalNum)
    if train:
        print("Train Accuracy: {}".format(acc))
    else:
        print("Valid Accuracy: {}".format(acc))
    return losses, acc




def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key].float(), len(w))
    return w_avg


test = DataProcess("../../Task1/train_data_final", unused_attrs=['district', 'city', 'zip code', 'region',
                                                                 'unit price of residence space',
                                                                 'unit price of building space', 'total cost', ])

data, cities, zipcodes, labels = test.getdata(normalize=True)
city_num = len(np.unique(cities))
zip_num = len(np.unique(zipcodes))
train_num = int(len(data) * 0.9)

train_x, valid_x = data[:train_num], data[train_num:]
train_city, valid_city = cities[:train_num], cities[train_num:]
train_zipcodes, valid_zipcodes = zipcodes[:train_num], zipcodes[train_num:]
train_label, valid_label = labels[:train_num], labels[train_num:]

avg_num = train_num // 4
train_data_list = []
for i in range(0, train_num, avg_num):
    sub_x = train_x[i:i + avg_num]
    sub_cities = train_city[i:i + avg_num]
    sub_zipcodes = train_zipcodes[i:i + avg_num]
    sub_labels = train_label[i:i + avg_num]
    train_data_list.append((sub_x, sub_cities, sub_zipcodes, sub_labels))

test_data_list = []
test_data_list.append((valid_x, valid_city, valid_zipcodes, valid_label))
# for i in range(0, train_num, avg_num):
#     sub_x = valid_x[i:i + avg_num]
#     sub_cities = valid_city[i:i + avg_num]
#     sub_zipcodes = valid_zipcodes[i:i + avg_num]
#     sub_labels = valid_label[i:i + avg_num]
#     test_data_list.append((sub_x, sub_cities, sub_zipcodes, sub_labels))



global_model = DenseNet(data.shape[1], 8, 32, city_num, zip_num).to(device)

global_rounds = 3
local_epochs = 1000
user_num = 4
losses = []
accs = []
best_acc = 0
for round_idx in range(global_rounds):
    local_weights = []
    local_losses = []
    global_acc = []

    for user_index in range(user_num):
        model_weights = local_trainer(train_data_list[user_index], copy.deepcopy(global_model), round_idx, device,
                                      local_epochs)
        local_weights.append(copy.deepcopy(model_weights))

    global_weight = average_weights(local_weights)
    global_model.load_state_dict(global_weight)
    loss, acc = eval(global_model, test_data_list[0], True)
    loss = float(loss)
    losses.append(loss)
    acc = float(acc)
    accs.append(acc)
    print('global losses:{losses}, global acc:{acc}'.format(losses=losses, acc=accs))
    if acc > best_acc:
        best_acc = acc
        torch.save(global_model.state_dict(), 'global_model.pkl')


x = np.linspace(0, 100, len(losses))
plt.plot(x, accs,color = 'blue', label='acc')
plt.show()
import pandas as pd
df = pd.DataFrame({'loss': losses, 'acc': accs})
df.to_csv('loss_acc.csv', index=False)
#