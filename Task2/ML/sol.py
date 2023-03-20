import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import Normalizer, MaxAbsScaler
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


def process(df, item, top):
    list_of_items = list(df[item].apply(
        lambda x: [i['name'] for i in x] if x != {} else []).values)
    df.loc[:, 'num_' + item] = df[item].apply(
        lambda x: len(x) if x != {} else 0)
    df.loc[:, 'all_' + item] = df[item].apply(
        lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
    top_items = [m[0] for m in Counter(
        [i for j in list_of_items for i in j]).most_common(top)]
    for g in top_items:
        df.loc[:, item + '_' +
               g] = df.loc[:, 'all_' + item].apply(lambda x: 1 if g in x else 0)
    df = df.drop(['all_' + item], axis=1)
    return df


if __name__ == "__main__":
    trainData['price range'] = 0
    trainData = countTotalCost(trainData)
    for totalCost, index in zip(trainData['total cost'], trainData.index):
        trainData.loc[index, 'price range'] = classify(totalCost)
    # print(dum.info())
    trainData['date'] = pd.to_datetime(trainData['date'])
    trainData['year'] = trainData['date'].dt.year
    trainData['month'] = trainData['date'].dt.month
    trainData['day'] = trainData['date'].dt.day
    # trainData['decorated'] = 0
    # trainData.loc[trainData['decoration year'] != 0, 'decorated'] = 1
    # trainData['basement'] = 0
    # trainData.loc[trainData['basement space'] != 0, 'basement'] = 1
    trainData.loc[trainData['decoration year'] == 0,
                  'decoration year'] = trainData['building year']

    trainData['median_rank_zip code'] = trainData.groupby(
        'zip code')['price range'].transform('median')
    trainData['median_rank_city'] = trainData.groupby(
        'city')['price range'].transform('median')
    trainData['mean_rank_zip code'] = trainData.groupby(
        'zip code')['price range'].transform('mean')
    trainData['mean_rank_city'] = trainData.groupby(
        'city')['price range'].transform('mean')

    oneHotColumns = ['zip code','city']
    dum = pd.get_dummies(trainData,
                         columns=oneHotColumns, drop_first=True)
    dropColumns = [  'district',  'region', 'year', 'month', 'day',
                   'building year', 'unit price of residence space', 'unit price of building space', 'total cost']
    dum = dum.drop(
        dropColumns, axis=1)
    trainData = trainData.drop(
        oneHotColumns, axis=1)
    trainData = trainData.drop(
        dropColumns, axis=1)

    trainData = trainData.merge(dum, how="left")

    # trainData[['number of rooms', 'security level of the community', 'residence space', 'building space', 'noise level',
    #            'view', 'air quality level', 'aboveground space ', 'basement space', 'exchange rate']].apply(max_min_scaler)
    trainData.to_csv(os.path.dirname(
        __file__)+"/train_data.csv")
    # trainData['price range'] = trainData['price range'].astype('category')

    dataX = trainData.drop(['price range'], axis=1)
    # preprocess = MinMaxScaler()
    # dataX = preprocess.fit_transform(dataX)
    # preprocess = MaxAbsScaler()
    # dataX = preprocess.fit_transform(dataX)
    dataY = trainData['price range']

    trainX, testX, trainY, testY = train_test_split(
        dataX, dataY, test_size=0.3, shuffle=True)
    model = CatBoostClassifier(iterations=5000,
                               depth=5,
                               learning_rate=0.02,
                               loss_function='MultiClass',
                               logging_level='Verbose',
                               random_seed=2023)
    model.fit(trainX, trainY)
    pred = model.predict(testX)
    acc = accuracy_score(testY, pred)
    print(acc)
    pred = model.predict(trainX)
    acc = accuracy_score(trainY, pred)
    print(acc)
    print(trainData.info())
    # strtfdKFold = StratifiedKFold(n_splits=10)
    # kfold = strtfdKFold.split(dataX, dataY)
    # scores = []
    # for k, (train, test) in enumerate(kfold):
    #     model = CatBoostClassifier(iterations=10000,
    #                                depth=5,
    #                                learning_rate=0.01,
    #                                loss_function='MultiClass',
    #                                logging_level='Silent',
    #                                random_seed=2023)

    #     model.fit(dataX.iloc[train, :], dataY.iloc[train])
    #     trainScore = model.score(dataX.iloc[train, :], dataY.iloc[train])
    #     testScore = model.score(dataX.iloc[test, :], dataY.iloc[test])
    #     scores.append(testScore)
    #     print('Fold: %2d, Training accuracy: %.3f, Testing accuracy: %.3f' %
    #           (k+1, trainScore, testScore))
    # print('Cross-Validation accuracy: %.3f +/- %.3f' %
    #       (np.mean(scores), np.std(scores)))
