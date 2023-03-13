import pandas as pd
import numpy as np
import torch
import torch.nn as nn


embed = nn.Embedding(10, 100)
x = torch.arange(10)

output = embed(x)
print(output.shape)
