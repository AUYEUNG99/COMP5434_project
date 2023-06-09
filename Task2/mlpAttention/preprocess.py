import numpy as np
import pandas as pd
from sklearn import preprocessing


class DataProcess:
    def __init__(self, filepath, unused_attrs, train=True):
        self.data = None
        self.df = None
        self.cities = None
        self.zipcodes = None
        self.labels = None
        self.total_cost = None
        self.fp = filepath
        self.unused = unused_attrs
        self.train = train

    def read_data(self):
        self.df = pd.read_csv(self.fp)

    def filter(self):
        """
        We don't need attributes like 'District' & 'ZipCode' since they're mostly likely to be redundant,
        in addition to this, they're hard to deal with
        :return:
        """
        self.cities = self.df["city"]
        self.zipcodes = self.df["zip code"]
        if self.train:
            self.total_cost = self.df['total cost']
        self.df = self.df.drop(columns=self.unused)

    def encode(self, normalize: bool):
        """
        1. for date attribute : extract month as feature
        2. for city attribute : encoded through nn.Embedding, but first we need to convert them to numeric values
        3. for label attribute : follow the rules
        4. Feature Normalization
        :return:
        """

        # for each row in 'date' attribute, split the string and get the second value(as integer) which is month
        if 'date' in self.df.columns:
            self.df['date'] = self.df['date'].apply(lambda x: int(x.split('/')[1]))

        le = preprocessing.LabelEncoder()
        self.cities = le.fit_transform(self.cities)
        self.zipcodes = le.fit_transform(self.zipcodes)

        def classify(totalCost):
            if totalCost < 300000:
                return 1
            elif totalCost < 500000:
                return 2
            elif totalCost < 700000:
                return 3
            else:
                return 4

        if self.train:
            self.labels = self.total_cost.apply(lambda x: classify(x))
            self.labels = self.labels.to_numpy(dtype=np.int32)
            self.labels -= 1

        self.data = self.df.to_numpy()
        """
        it may not be as good as we think. because all values are too small
        we may consider that only do normalization for necessary columns???
        """
        if normalize:
            self.data = preprocessing.normalize(self.data, axis=0)

    def getdata(self, normalize: bool):
        """
        :return:  Normalized training data, city attribute which needs to be embedded, labels
        """
        self.read_data()
        self.filter()
        self.encode(normalize)
        if self.train:
            return self.data, self.cities, self.zipcodes, self.labels
        else:
            return self.data, self.cities, self.zipcodes
