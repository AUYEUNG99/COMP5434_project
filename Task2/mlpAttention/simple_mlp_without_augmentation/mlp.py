import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum


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
        self.bn1 = nn.BatchNorm1d(2 * hidden_dim)

        self.fc2 = nn.Linear(2 * hidden_dim, 3 * hidden_dim)
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
        zipcode_embedding = self.zip_embed(zipcodes)
        features = torch.concat((x, city_embedding, zipcode_embedding), dim=-1)  # [Batch, num, dim + embed]

        f1 = self.fc1(features)
        f1 = torch.permute(f1, [0, 2, 1])
        f1 = self.bn1(f1)
        f1 = torch.permute(f1, [0, 2, 1])
        f1 = self.activ(f1)

        f2 = self.fc2(f1)
        f2 = torch.permute(f2, [0, 2, 1])
        f2 = self.bn2(f2)
        f2 = torch.permute(f2, [0, 2, 1])
        f2 = self.activ(f2)

        f4 = self.fc4(f2)
        f4 = torch.permute(f4, [0, 2, 1])
        f4 = self.bn4(f4)
        f4 = torch.permute(f4, [0, 2, 1])
        f4 = self.activ(f4)

        output = self.mlp_head(f4)
        return F.softmax(output, dim=-1)


