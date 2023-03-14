import torch
import numpy as np
from model import DenseNet
from dataset import HousepriceDataset
from torch.utils.data import DataLoader
from train import train


from preprocess import DataProcess

test = DataProcess("Train_Data.csv", unused_attrs=['district', 'city', 'zip code', 'region',
                                                           'exchange rate', 'unit price of residence space',
                                                           'unit price of building space', 'total cost',
                                                            ])
l1, l2, l3 = test.getdata(normalize=True)
city_num = len(np.unique(l2))

print(l1.shape)
"""
因为现在的数据集文件 total cost没补充, 值是nan，所以label均为4！
"""
# print(l3)

# print(l2)


"""
net = DenseNet(100, 64, 64, 10)

cities = torch.randint(9, size=(10, 200))
inputx = torch.randn((10, 200, 100))
output = net(inputx, cities)
print(output.shape)
"""

dataset = HousepriceDataset(l1, l2, l3, 50)
dataloader = DataLoader(dataset, batch_size=20, shuffle=True)
model = DenseNet(l1.shape[1], 32, 128, city_num)

train(dataloader, model)
