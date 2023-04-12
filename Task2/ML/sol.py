import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split
from catboost import CatBoostClassifier
import datetime as dt
import optuna
from optuna.samplers import TPESampler

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


def objective(trial):
    X_train, X_test, y_train, y_test = train_test_split(
        dataX, dataY, test_size=0.1)
    param = {
        "iterations": trial.suggest_int("iterations", 100, 5000),
        "loss_function": trial.suggest_categorical("loss_function", ["MultiClass"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e0),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 1),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 1e-2, 1e0),
        "depth": trial.suggest_int("depth", 5, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 20),
    }

    reg = CatBoostClassifier(
        **param, cat_features=categorical_features_indices)
    reg.fit(X_train, y_train, eval_set=[
            (X_test, y_test)], verbose=0, early_stopping_rounds=100)
    y_pred = reg.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    return score


if __name__ == "__main__":
    trainData['price range'] = 0
    trainData = countTotalCost(trainData)
    for totalCost, index in zip(trainData['total cost'], trainData.index):
        trainData.loc[index, 'price range'] = classify(totalCost)
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
    trainData.loc[trainData['decoration year'] < trainData['building year'],
                  'decoration year'] = trainData['building year']
    # trainData['house_age'] = trainData['year'] - trainData['building year']
    # trainData['decoration_age'] = trainData['year'] - \
    #     trainData['decoration year']
    # trainData['median_rank_zip code'] = trainData.groupby(
    #     'zip code')['price range'].transform('median')
    # trainData['median_rank_city'] = trainData.groupby(
    #     'city')['price range'].transform('median')
    # trainData['mean_rank_zip code'] = trainData.groupby(
    #     'zip code')['price range'].transform('mean')
    # trainData['mean_rank_city'] = trainData.groupby(
    #     'city')['price range'].transform('mean')
    # oneHotColumns = []
    # dum = pd.get_dummies(trainData,
    #                      columns=oneHotColumns, drop_first=True)
    dropColumns = ['district',  'region', 'year', 'month', 'day', 'date',
                    'unit price of residence space', 'unit price of building space', 'total cost']

    # dum = dum.drop(
    #     dropColumns, axis=1)
    # trainData = trainData.drop(
    #     oneHotColumns, axis=1)
    trainData = trainData.drop(
        dropColumns, axis=1)
    # trainData = trainData.merge(dum, how="left")
    print(trainData.info())
    trainData[['number of rooms', 'security level of the community', 'residence space', 'building space', 'noise level', 'waterfront',
               'view', 'air quality level', 'aboveground space ', 'basement space', 'exchange rate',]].apply(max_min_scaler)
    trainData.to_csv(os.path.dirname(
        __file__)+"/train_data.csv", index=False)

    # trainData['price range'] = trainData['price range'].astype('category')

    dataX = trainData.drop(['price range'], axis=1)

    dataY = trainData['price range']
    categorical_features_indices = np.where(dataX.dtypes != np.float64)[0]
    study = optuna.create_study(sampler=TPESampler(), direction="maximize")
    study.optimize(objective, n_trials=30, timeout=60)


    strtfdKFold = StratifiedKFold(n_splits=10)
    kfold = strtfdKFold.split(dataX, dataY)
    scores = []
    for k, (train, test) in enumerate(kfold):
        model = CatBoostClassifier(
            **study.best_params, cat_features=categorical_features_indices, silent=True)
        model.fit(dataX.iloc[train, :], dataY.iloc[train])
        trainScore = model.score(dataX.iloc[train, :], dataY.iloc[train])
        testScore = model.score(dataX.iloc[test, :], dataY.iloc[test])
        scores.append(testScore)
        print('Fold: %2d, Training accuracy: %.3f, Testing accuracy: %.3f' %
              (k+1, trainScore, testScore))
    print('Cross-Validation accuracy: %.3f +/- %.3f' %
          (np.mean(scores), np.std(scores)))
