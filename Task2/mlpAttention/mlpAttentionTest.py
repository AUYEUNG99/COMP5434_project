from torch.utils.data import DataLoader
from dataset import HousepriceDataset_V1, HousepriceDataset_V2
import numpy as np
from preprocess import DataProcess
from mlp_global_attn import DenseNet
from train import train
from sklearn import preprocessing
from torch.utils.data import DataLoader
import pandas as pd

test = DataProcess("../../Task1/train_data_final", unused_attrs=['district', 'city', 'zip code', 'region',
                                                                 'unit price of residence space',
                                                                 'unit price of building space', 'total cost', ])

l1, l2, l3 = test.getdata(normalize=True)
train_num = int(len(l1) * 0.9)
city_num = len(np.unique(l2))
"""
use augmented dataset to replace original dataset
"""
l1 = pd.read_csv("../ML/train_data18.csv").to_numpy()
l1 = l1[:, :-4]
l1 = preprocessing.normalize(l1, axis=0)
train_x, valid_x = l1[:train_num], l1[train_num:]
train_city, valid_city = l2[:train_num], l2[train_num:]
train_label, valid_label = l3[:train_num], l3[train_num:]

train_dataset = HousepriceDataset_V2(train_x, train_city, train_label, 40)
test_dataset = HousepriceDataset_V2(valid_x, valid_city, valid_label, 20)
train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True,
                              )

test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=True,
                             )
model = DenseNet(l1.shape[1], 8, 32, city_num)

trl, tra, tel, tea = list(), list(), list(), list()
train(train_dataloader, model, test_dataloader, trl, tra, tel, tea)
