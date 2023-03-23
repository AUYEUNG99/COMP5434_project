from dataset import HousepriceDataset_V1, HousepriceDataset_V2
import numpy as np
from preprocess import DataProcess
from mlp_global_attn import DenseNet
from mlp_global_attn import Attention
from train import train
from sklearn import preprocessing
from torch.utils.data import DataLoader
import pandas as pd
import torch

test = DataProcess("../../data/Test_Data.csv", unused_attrs=['district', 'city', 'zip code', 'region',
                                                             'total cost', 'exchange rate', ], train=False)

model = torch.load("./global_attention_without_augmentation/7675validation.pth", map_location='cpu')

l1, l2, l3 = test.getdata(normalize=True)
data = torch.tensor(l1, dtype=torch.float32)[None, :, :]
cities = torch.tensor(l2, dtype=torch.int32)[None, :]
zipcodes = torch.tensor(l3, dtype=torch.int32)[None, :]

outputs = model(data, cities, zipcodes)
# print(outputs.shape)  # shape [1, 400, 4]

outputs = torch.squeeze(outputs)
predict_labels = torch.argmax(outputs, dim=1)

print(predict_labels)


df = pd.read_csv("../../data/Test_Data.csv")
df['price range'] = predict_labels.numpy()

df.to_csv("./predict_test_set.csv")
