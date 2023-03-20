from torch.utils.data import DataLoader
from dataset import HousepriceDataset_V1, HousepriceDataset_V2
import numpy as np
from preprocess import DataProcess
from mlp_global_attn import DenseNet
from train import train

test = DataProcess("./train_data_final", unused_attrs=['district', 'city', 'zip code', 'region',
                                                       'unit price of residence space',
                                                       'exchange rate',
                                                       'unit price of building space', 'total cost',

                                                       ])

l1, l2, l3, l4 = test.getdata(normalize=True)
train_num = int(len(l1) * 0.9)
city_num = len(np.unique(l2))
zip_num = len(np.unique(l3))
train_x, valid_x = l1[:train_num], l1[train_num:]
train_city, valid_city = l2[:train_num], l2[train_num:]
train_zips, valid_zips = l3[:train_num], l3[train_num:]
train_label, valid_label = l4[:train_num], l4[train_num:]

"""
因为现在的数据集文件 total cost没补充, 值是nan，所以label均为4！
"""

"""
net = DenseNet(100, 64, 64, 10)

cities = torch.randint(9, size=(10, 200))
inputx = torch.randn((10, 200, 100))
output = net(inputx, cities)
print(output.shape)
"""

train_dataset = HousepriceDataset_V2(train_x, train_city, train_label, train_zips, 40)
test_dataset = HousepriceDataset_V2(valid_x, valid_city, valid_label, valid_zips, 20)
print(len(train_dataset))
print(len(test_dataset))
train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True,
                              )

test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=True,
                             )
model = DenseNet(l1.shape[1], 8, 32, city_num, zip_num)

train(train_dataloader, model, test_dataloader)

