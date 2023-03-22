import torch
from torch.utils.data import Dataset


class HousepriceDataset_V1(Dataset):
    def __init__(self, data, city, label, zipcodes, train_num):
        super(HousepriceDataset_V1, self).__init__()
        self.data = data
        self.city = city
        self.zipcodes = zipcodes
        self.label = label
        self.num = train_num

    def __len__(self):
        return len(self.label) - self.num

    def __getitem__(self, index):
        """
        data = torch.Tensor(self.data[index])
        city = torch.tensor(self.city[index], dtype=torch.int)
        label = torch.tensor(self.label[index], dtype=torch.long)
        """

        data = torch.Tensor(self.data[index:index + self.num])
        city = torch.tensor(self.city[index:index + self.num], dtype=torch.int)
        zipcodes = torch.tensor(self.zipcodes[index:index + self.num], dtype=torch.int)
        label = torch.tensor(self.label[index:index + self.num], dtype=torch.long)

        return data, city, zipcodes, label


class HousepriceDataset_V2(Dataset):
    def __init__(self, data, city, label, zipcodes, train_num):
        super(HousepriceDataset_V2, self).__init__()
        self.data = data
        self.city = city
        self.label = label
        self.zipcodes = zipcodes
        self.num = train_num

    def __len__(self):
        return len(self.label) // self.num

    def __getitem__(self, index):
        """
        data = torch.Tensor(self.data[index])
        city = torch.tensor(self.city[index], dtype=torch.int)
        label = torch.tensor(self.label[index], dtype=torch.long)
        """

        data = torch.Tensor(self.data[index * self.num:index * self.num + self.num])
        city = torch.tensor(self.city[index * self.num:index * self.num + self.num], dtype=torch.int)
        label = torch.tensor(self.label[index * self.num:index * self.num + self.num], dtype=torch.long)
        zipcodes = torch.tensor(self.zipcodes[index * self.num:index * self.num + self.num], dtype=torch.int)
        return data, city, zipcodes, label
