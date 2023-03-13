import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from model import DenseNet

net = DenseNet(100, 64, 64, 10)

cities = torch.randint(9, size=(10, 200))
inputx = torch.randn((10, 200, 100))
output = net(inputx, cities)
print(output.shape)
