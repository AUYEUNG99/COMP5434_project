import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, model_dim, hidden_dim, dropout=0.5):
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
        q, k, v = torch.chunk(self.linear(x), chunks=3, dim=-1)     # all in shape [Batch, 100, hidden_dim]
        residual = x
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        """
        mask operation waited to be implemented
        """
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        output = self.fc(output) + residual
        return output


class DenseNet(nn.Module):
    def __init__(self, d_model, embed_dim, hidden_dim, city_num, num_classes = 4):
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

        self.fc1 = nn.Linear(embed_dim + d_model, 3 * hidden_dim)
        self.attn1 = Attention(3 * hidden_dim, 2 * hidden_dim)
        self.fc2 = nn.Linear(3 * hidden_dim, hidden_dim)
        self.attn2 = Attention(hidden_dim, hidden_dim)

        self.activ = nn.ReLU()

        self.fc3 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc4 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.mlp_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, cities):
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
        features = torch.concat((x, city_embedding), dim=-1)        # [Batch, num, dim + embed]
        f1 = self.fc1(features)
        f1 = self.activ(f1)
        attn1 = self.attn1(f1)
        f2 = self.fc2(attn1)
        f2 = self.activ(f2)
        attn2 = self.attn2(f2)

        # flatten = attn2.view(x.shape[0], -1)

        f3 = self.fc3(attn2)
        f3 = self.activ(f3)
        f4 = self.fc4(f3)
        f4 = self.activ(f4)

        output = self.mlp_head(f4)
        return F.softmax(output, dim=-1)

