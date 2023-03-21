import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum


def select_neighbor_features(values, indices, dim=2):
    """
    :param values:
    Attention map , in shape [Batch, points_num, points_num, dim]
    relative_pos_embedding, [Batch, points_num, points_num, pos_embed]
    Value, ...
    :param indices: in shape [Batch, points_num, k_neighbor]
    :param dim:
    :return:
    """
    # 每个可能的value's feature map的torch.Size()
    value_dims = values.shape[(dim + 1):]

    # 单纯得到List类型的两个shape, shape类型如上所述
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))

    """
    结构解析: 
        len(value_dims) = 1
        (None,)是一个tuple, 乘以一个整数就是循环了几次
        -----比如 (None, ) * 2 = (None, None)-----

        用* unpack 所以indices = indices[:, :, :, None]
        '...'代表indices的所有维度
    """
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)


class SA_Layer(nn.Module):
    def __init__(self, model_dim, embed_dim, hidden_dim, dropout=0.1):
        super(SA_Layer, self).__init__()

        # for simplicity, set Value dimension = hidden_dim
        self.linear = nn.Linear(model_dim, hidden_dim * 3)

        self.pos_mlp = nn.Linear(embed_dim, hidden_dim)
        # for numerical stable
        self.temperature = hidden_dim ** 0.5
        self.dropout = nn.Dropout(dropout)
        # the idea of Attention: input shape = output shape
        self.fc = nn.Linear(hidden_dim, model_dim)

    def forward(self, x, pos, neighbors=5):
        """
        :param x: [batch, sample_num, model_dim]
        :param pos:  actually it's city embedding, we assume that it has position information
        :return: new features after attention
        """
        q, k, v = torch.chunk(self.linear(x), chunks=3, dim=-1)  # all in shape [Batch, points_num, hidden_dim]
        residual = x

        # use the idea proposed in point_transformer, subtract to reveal similarity
        qk_rel = q[:, :, None, :] - k[:, None, :, :]
        v = v[:, None, :, :].repeat(1, x.shape[1], 1, 1)  # each feature map is a complete value matrix
        # use city embedding as position information (Assume that embedding can reflect the relative distance)
        rel_dis = pos[:, :, None, :] - pos[:, None, :, :]
        rel_dis = self.pos_mlp(rel_dis)
        pos_real = torch.linalg.norm(rel_dis, dim=-1)  # [Batch, sample_num, points_num]
        # return 'neighbors' smallest points of each sample
        values, indices = torch.topk(pos_real, neighbors, largest=False)

        """
        Now our goal is to convert qk information and position information into shape 
        [Batch, sample, neighbors, model_dim]
        """
        qk_rel = select_neighbor_features(qk_rel, indices)
        rel_dis = select_neighbor_features(rel_dis, indices)
        v = select_neighbor_features(v, indices)

        # add relative positional embeddings to value
        v = v + rel_dis
        # attention
        attn = qk_rel.softmax(dim=-2)
        # aggregate
        agg = einsum('b i j d, b i j d -> b i d', attn, v)
        return agg


class DenseNet(nn.Module):
    def __init__(self, d_model, embed_dim, hidden_dim, city_num, zip_num, num_classes=4, num_neighbors=5):
        """
        :param d_model:        original model dimension before embedding for cities
        :param embed_dim:      embedding dim for cities (district)
        :param hidden_dim:     hidden dim we want for Attention operation
        :param city_num:       unique cities in dataset
        """
        super(DenseNet, self).__init__()

        self.hidden = hidden_dim
        self.d_model = d_model
        self.neighbors = num_neighbors
        self.city_embed = nn.Embedding(city_num, embed_dim)
        self.zip_embed = nn.Embedding(zip_num, embed_dim)

        self.attn = SA_Layer(d_model + embed_dim * 2, embed_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, hidden_dim * 2)
        self.BR = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim * 3)
        self.BR2 = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * 3),
            nn.ReLU()
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, cities, zipcodes):
        """
        :param x:            house data input except cities (zipcode & district are abandoned)      [Batch, num, dim]
        :param cities:       numeric value of original city attribute                               [Batch, num, embed]
        :return:
        """
        city_embedding = self.city_embed(cities)  # [Batch, sample_num, embed_dim]
        zipcode_embedding = self.zip_embed(zipcodes)

        features = torch.concat((x, city_embedding, zipcode_embedding), dim=-1)  # [Batch, num, dim + embed * 2]
        attn = self.attn(features, city_embedding)  # [batch, num, hidden_dim]

        f1 = self.fc(attn)
        f1 = torch.permute(f1, [0, 2, 1])
        f1 = self.BR(f1)
        f1 = torch.permute(f1, [0, 2, 1])

        f2 = self.fc2(f1)
        f2 = torch.permute(f2, [0, 2, 1])
        f2 = self.BR2(f2)
        f2 = torch.permute(f2, [0, 2, 1])

        output = self.mlp_head(f2)

        return F.softmax(output, dim=-1)
