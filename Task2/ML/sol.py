import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

trainData = pd.read_csv('data/Train_data.csv')
testData = pd.read_csv('data/Test_data.csv')


def countTotalCost(trainData):
    for rSpace, bSpace, rPrice, bPrice, rate, index in zip(trainData['residence space'],  trainData['building space'], trainData['unit price of residence space'], trainData['unit price of building space'], trainData['exchange rate'], trainData.index):
        # print(rSpace, bSpace, rPrice, bPrice, rate)
        trainData.loc[index, 'total cost'] = round((rSpace * rPrice + bSpace * bPrice)*rate)

    return trainData


def classify(totalCost):
    if totalCost < 300000:
        return 1
    elif totalCost < 500000:
        return 2
    elif totalCost < 700000:
        return 3
    else:
        return 4

if __name__ == "__main__":
    trainData['price range'] = 0
    trainData = countTotalCost(trainData)
    for totalCost, index in zip(trainData['total cost'], trainData.index):
        trainData.loc[index, 'price range'] = classify(totalCost)
    dropColumns = ['district', 'city', 'zip code', 'region', 'date','unit price of residence space','unit price of building space','total cost']
    trainData = trainData.drop(
            dropColumns, axis=1)
    trainData.to_csv(os.path.dirname(
        __file__)+"/train_data.csv")
    trainData = (trainData - trainData.min()) / (trainData.max() - trainData.min())

    dataX = trainData.drop(['price range'], axis=1)
    dataY = trainData['price range']
    trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=0.3)
    model = GradientBoostingRegressor()
    model.fit(trainX, trainY)
    pred = model.predict(testX)
    r2_score = model.score(testX, testY)
    print(r2_score)
    # print(trainData.info())
