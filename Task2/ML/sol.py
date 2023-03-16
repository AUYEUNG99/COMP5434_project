import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split
from catboost import CatBoostClassifier
import datetime as dt
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

trainData = pd.read_csv('data/Train_data.csv')
testData = pd.read_csv('data/Test_data.csv')


def max_min_scaler(x): return (x-np.min(x))/(np.max(x)-np.min(x))


def countTotalCost(trainData):
    for rSpace, bSpace, rPrice, bPrice, rate, index in zip(trainData['residence space'],  trainData['building space'], trainData['unit price of residence space'], trainData['unit price of building space'], trainData['exchange rate'], trainData.index):
        # print(rSpace, bSpace, rPrice, bPrice, rate)
        trainData.loc[index, 'total cost'] = round(
            (rSpace * rPrice + bSpace * bPrice)*rate)

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

    oneHotColumns = ['city', 'zip code']
    dum = pd.get_dummies(trainData,
                         columns=oneHotColumns, drop_first=True)
    # print(dum.info())
    trainData['date'] = pd.to_datetime(trainData['date'])
    trainData['year'] = trainData['date'].dt.year
    trainData['month'] = trainData['date'].dt.month
    trainData['day'] = trainData['date'].dt.day
    dropColumns = ['district', 'region', 'date',
                   'unit price of residence space', 'unit price of building space', 'total cost']
    dum = dum.drop(
        dropColumns, axis=1)
    trainData = trainData.drop(
        oneHotColumns, axis=1)
    trainData = trainData.drop(
        dropColumns, axis=1)

    # trainData = (trainData - trainData.min()) / \
    #     (trainData.max() - trainData.min())

    trainData = trainData.merge(dum, how="left")
    trainData[['number of rooms', 'security level of the community', 'residence space', 'building space', 'noise level',
               'view', 'air quality level', 'aboveground space ', 'basement space', 'exchange rate']].apply(max_min_scaler)
    trainData.to_csv(os.path.dirname(
        __file__)+"/train_data.csv")
    trainData['price range'] = trainData['price range'].astype('category')

    dataX = trainData.drop(['price range'], axis=1)
    dataY = trainData['price range']

    # trainX, testX, trainY, testY = train_test_split(
    #     dataX, dataY, test_size=0.3)
    # kf = KFold(n_splits=5, random_state=42, shuffle=True)
    # for i, (trainIndex, testIndex) in enumerate(kf.split(dataX, dataY)):
    #     trainX, testX = dataX.iloc[trainIndex,
    #                                :], dataX.iloc[testIndex, :]
    #     trainY, testY = dataY[trainIndex], dataY[testIndex]

    #     model = CatBoostClassifier(iterations=5000,
    #                                depth=5,
    #                                learning_rate=0.05,
    #                                loss_function='MultiClass',
    #                                logging_level='Verbose')
    #     model.fit(trainX, trainY)
    #     pred = model.predict(testX)
    #     acc = accuracy_score(testY, pred)
    #     print(acc)
    #     pred = model.predict(trainX)
    #     acc = accuracy_score(trainY, pred)
    #     print(acc)

    strtfdKFold = StratifiedKFold(n_splits=10)
    kfold = strtfdKFold.split(dataX, dataY)
    scores = []
    for k, (train, test) in enumerate(kfold):
        model = CatBoostClassifier(iterations=5000,
                                   depth=6,
                                   learning_rate=0.01,
                                   loss_function='MultiClass',
                                   logging_level='Silent',
                                   random_seed=2023)

        model.fit(dataX.iloc[train, :], dataY.iloc[train])
        trainScore = model.score(dataX.iloc[train, :], dataY.iloc[train])
        testScore = model.score(dataX.iloc[test, :], dataY.iloc[test])
        scores.append(testScore)
        print('Fold: %2d, Training accuracy: %.3f, Testing accuracy: %.3f' %
              (k+1, trainScore, testScore))
    print('Cross-Validation accuracy: %.3f +/- %.3f' %
          (np.mean(scores), np.std(scores)))
