import os
import numpy as np
import pandas as pd
import pydot
import graphviz
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import sklearn.metrics as met
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import normalize
from random import seed
from random import random
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# data_dir = 'drive/MyDrive/SampleStockData'
data_dir = 'Data/Stocks'
all_files = os.listdir(data_dir)
np.random.seed(43)


def get_stock_subset_ratio(ratio):
    num_requested = int(len(all_files) * ratio)
    return np.random.choice(all_files, num_requested)


def get_stock_subset_num(num):
    return np.random.choice(all_files, num)


def build_data_matrix(file, window_size):
    text_file = open(data_dir + '/' + file, 'r')
    lines = text_file.readlines()

    # remove newlines https://stackoverflow.com/questions/43447221/removing-all-spaces-in-text-file-with-python-3-x
    lines = [line.replace('\n', '') for line in lines]

    # must leave [window_size] entries to build row
    max_window_start = (len(lines) - 1) - window_size

    features = []
    labels = []

    for x in range(1, max_window_start):

        row = []
        for j in range(x, x + window_size):
            split_line = lines[j].split(',')
            # Date,Open,High,Low,Close,Volume,OpenInt
            # 0    1    2    3   4     5      6
            row.append(float(split_line[1]))
            row.append(float(split_line[2]))
            row.append(float(split_line[3]))
            row.append(float(split_line[4]))
            row.append(int(split_line[5]))
        features.append(row)

        # target is the closing price of the next day
        labels.append(float(lines[x + window_size].split(',')[4]))

    return features, labels


def build_np_matrix(file, window_size):
    text_file = open(data_dir + '/' + file, 'r')
    lines = text_file.readlines()

    # remove newlines https://stackoverflow.com/questions/43447221/removing-all-spaces-in-text-file-with-python-3-x
    lines = [line.replace('\n', '') for line in lines]

    # must leave [window_size] entries to build row
    max_window_start = (len(lines) - 1) - window_size

    features = []
    labels = []

    for x in range(1, max_window_start):

        row = []
        for j in range(x, x + window_size):
            split_line = lines[j].split(',')
            # Date,Open,High,Low,Close,Volume,OpenInt
            # 0    1    2    3   4     5      6
            row.append(float(split_line[1]))
            row.append(float(split_line[2]))
            row.append(float(split_line[3]))
            row.append(float(split_line[4]))
            row.append(int(split_line[5]))
        features.append(row)

        # target is the closing price of the next day
        # labels.append(float(lines[x + window_size].split(',')[4]))

        # instead, target is the the percentage increase for the closing price the next day
        next_day = lines[x + window_size].split(',')
        next_day_close = float(next_day[4])
        prev_day = lines[x + window_size - 1].split(',')
        prev_day_close = float(prev_day[4])
        difference = next_day_close - prev_day_close
        percent_increase = difference / prev_day_close
        labels.append(percent_increase)

    return np.array(features), np.array(labels)


def print_subset(subset):
    for i, file in enumerate(subset):
        if i > 3:
            break
        text_file = open(data_dir + '/' + file, 'r')
        print(file)
        lines = text_file.readlines()

        # remove newlines https://stackoverflow.com/questions/43447221/removing-all-spaces-in-text-file-with-python-3-x
        lines = [line.replace('\n', '') for line in lines]

        # print lines...
        for j, line in enumerate(lines):
            if j == 0:
                continue
            if j < 10:
                print(line)
            else:
                break


def print_file(file, num_lines):
    text_file = open(data_dir + '/' + file, 'r')
    print(file)

    lines = text_file.readlines()

    # remove newlines https://stackoverflow.com/questions/43447221/removing-all-spaces-in-text-file-with-python-3-x
    lines = [line.replace('\n', '') for line in lines]

    if (num_lines < 0):
        # print all
        for x in range(1, len(lines)):
            print(lines[x])
    else:
        # print first [num_lines] lines
        for x in range(1, num_lines + 1):
            print(lines[x])
        print('...')


def print_data(X, y):
    for x in range(len(X)):
        print(','.join(map(str, X[x])) + ' -> ' + str(y[x]))


# these are broken
def build_model(num_hidden_layers=2, num_neurons=100, output_neurons=2, learning_rate=3e-3,
                input_shape=(25)):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    # model.add(keras.layers.Flatten(input_shape=(input_shape)))
    for layer in range(num_hidden_layers):
        model.add(keras.layers.Dense(num_neurons, activation='relu'))
    model.add(keras.layers.Dense(output_neurons, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=learning_rate),
                  metrics=['accuracy'])
    return model


def LSTM_predict(X_train, X_test, y_train, y_test):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(
        units=128,
        input_shape=X_train.shape,
        return_sequences=True
    ))
    model.add(keras.layers.Dense(units=1))
    model.compile(
        loss='mean_squared_error',
        optimizer=keras.optimizers.Adam(0.001)
    )
    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=16,
        # validation_split=0.1,
        verbose=1,
        shuffle=False
    )
    return model.predict(X_test)


def linear_regression_predict(X_train, X_test, y_train, y_test):
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    lin_reg = LinearRegression(normalize=True).fit(X_train, y_train)
    return lin_reg.predict(X_test)


def mlp_regression_predict(X_train, X_test, y_train, y_test):
    mlp_reg = MLPRegressor(max_iter=5000000, hidden_layer_sizes=()).fit(X_train, y_train)
    return mlp_reg, mlp_reg.predict(X_test)


def rise_or_fall_accuracy(y_true, y_pred):
    count = 0
    for i in range(len(y_true)):
        if y_true[i] >= 0 and y_pred[i] >= 0:
            count += 1
        elif y_true[i] < 0 and y_pred[i] < 0:
            count += 1
    return (count / len(y_true)) * 100


def doStuff():
    # %%
    num_files = 1
    window_size = 15
    total_mse = 0
    total_acc = 0
    files = get_stock_subset_num(num_files)

    for file in files:
        print(file)

        # print_file(file, 10)
        # X, y = build_data_matrix(file, window_size)
        X, y = build_np_matrix(file, window_size)

        # normalize
        # X = normalize(X, axis=1)

        # dfx = pd.DataFrame(X)
        # dfy = pd.DataFrame(y)

        # print_data(X, y)

        # print(X.shape)

        X_train, X_test, y_train, y_test = [], [], [], []
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        except:
            num_files -= 1
            continue

        y_pred = linear_regression_predict(X_train, X_test, y_train, y_test)
        # mlp, y_pred = mlp_regression_predict(X_train, X_train, y_train, y_test)
        # y_pred = LSTM_predict(X_train, X_test, y_train, y_test)

        train_sizes = [1, 100, 500, 2000, 5000, 7654]
        train_sizes, train_scores, validation_scores = learning_curve(
            estimator=LinearRegression(),
            X=X,
            y=y, cv=5,
            scoring='neg_mean_squared_error')

        train_scores_mean = -train_scores.mean(axis=1)
        validation_scores_mean = -validation_scores.mean(axis=1)

        plt.figure()
        plt.style.use('seaborn')
        plt.plot(train_sizes, train_scores_mean, label='Training error')
        plt.plot(train_sizes, validation_scores_mean, label='Validation error')
        plt.ylabel('MSE', fontsize=14)
        plt.xlabel('Training set size', fontsize=14)
        plt.title('Learning curves for a linear regression model', fontsize=18, y=1.03)
        plt.legend()
        plt.ylim(0, .01)
        plt.show()
        # plt.savefig("test.png")

        # %%
        for score in train_scores:
            print(score)

        # print(mlp.n_iter_)

        # m = build_model(2, 50, 1, .01, 50)
        # keras.utils.plot_model(m)

        # history = m.fit(X_train, y_train)

        # TODO print mean squared error per iteration
        # report both
        # average multiple (~10) runs on one stock or take best run
        # test more on bibl.us.txt

        mse = met.mean_squared_error(y_test, y_pred)

        acc = rise_or_fall_accuracy(y_test, y_pred)

        total_mse += mse
        total_acc += acc

        print("Mean Squared Error           : " + str(mse))
        print("Predictive Gain/Loss accuracy: " + str(acc) + "\n")

    avg_mse = total_mse / num_files
    avg_acc = total_acc / num_files

    print('Average MSE:      ' + str(avg_mse))
    print('Average Accuracy: ' + str(avg_acc))


doStuff()

