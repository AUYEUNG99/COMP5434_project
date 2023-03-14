import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from model import DenseNet

from preprocess import DataProcess

test = DataProcess("../data/Train_Data.csv", unused_attrs=['district', 'city', 'zip code', 'region',
                                                           'exchange rate', 'unit price of residence space',
                                                           'unit price of building space', 'total cost'])
l1, l2, l3 = test.getdata()
print(l1)
"""
因为现在的数据集文件 total cost没补充, 值是nan，所以label均为4！
"""
print(l3)

print(l2)


"""
net = DenseNet(100, 64, 64, 10)

cities = torch.randint(9, size=(10, 200))
inputx = torch.randn((10, 200, 100))
output = net(inputx, cities)
print(output.shape)
"""

