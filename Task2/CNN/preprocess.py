import numpy as np
import pandas as pd
from sklearn import preprocessing


class DataProcess:
    def __init__(self, filepath, unused_attrs):
        self.data = None
        self.df = None
        self.cities = None
        self.labels = None
        self.total_cost = None
        self.fp = filepath
        self.unused = unused_attrs

    def read_data(self):
        self.df = pd.read_csv(self.fp)

    def filter(self):
        """
        We don't need attributes like 'District' & 'ZipCode' since they're mostly likely to be redundant,
        in addition to this, they're hard to deal with
        :return:
        """
        self.cities = self.df["city"]
        self.total_cost = self.df['total cost']
        self.district = self.df['district']
        self.zip_code = self.df['zip code']
        self.df = self.df.drop(columns=self.unused)
    def encode(self):
        """
        1. for date attribute : extract month as feature
        2. for city attribute : encoded through nn.Embedding, but first we need to convert them to numeric values
        3. for label attribute : follow the rules
        4. Feature Normalization
        :return:
        """
        header_str=self.df.columns.tolist()+['city','district','zip code']
        # for each row in 'date' attribute, split the string and get the second value(as integer) which is month
        self.df['date'] = self.df['date'].apply(lambda x: int(x.split('/')[1]))

        le = preprocessing.LabelEncoder()
        self.cities = le.fit_transform(self.cities).reshape(4000,1)
        self.district= le.fit_transform(self.district).reshape(4000,1)
        self.zip_code = le.fit_transform(self.zip_code).reshape(4000,1)
    
        
        def classify(totalCost):
            if totalCost < 300000:
                return 1
            elif totalCost < 500000:
                return 2
            elif totalCost < 700000:
                return 3
            else:
                return 4

        self.labels = self.total_cost.apply(lambda x: classify(x))
        self.labels = self.labels.to_numpy(dtype=np.int)
        self.attrs = np.hstack((self.df,self.cities, self.district, self.zip_code))
        
        """
        it may not be as good as we think. because all values are too small
        we may consider that only do normalization for necessary columns???
        """
        self.attrs = preprocessing.normalize(self.attrs, axis=0)
        self.attrs = pd.DataFrame(self.attrs,columns=header_str)
        self.attrs.to_csv('./train_attrs.csv',header=header_str,index=False)
        self.labels = pd.DataFrame(self.labels,columns=['labels'])
        self.labels.to_csv('./train_labels.csv',header=['labels'])
    def getdata(self):
        """
        :return:  Normalized training data, city attribute which needs to be embedded, labels
        """
        self.read_data()
        self.filter()
        self.encode()
        return self.attrs, self.labels
