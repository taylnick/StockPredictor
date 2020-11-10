import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import sklearn.metrics as met
# import tensorflow as tf
# from tensorflow import keras

data_dir = 'Data/Stocks'
all_files = os.listdir(data_dir)

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
        print(','.join(map(str, X[x])) + ' -> ' + y[x])

def linear_regression_predict(X_train, X_test, y_train, y_test):
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    lin_reg = LinearRegression().fit(X_train, y_train)
    return lin_reg.predict(X_test)
def mlp_regression_predict(X_train, X_test, y_train, y_test):
    mlp_reg = MLPRegressor().fit(X_train, y_train)
    return mlp_reg.predict(X_test)

files = get_stock_subset_num(5)
for file in files:
    print(file)
    window_size = 5

    # print_file(file, 10)
    X, y = build_data_matrix(file, window_size)

    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    y_pred = linear_regression_predict(X_train, X_test, y_train, y_test)
    # y_pred = mlp_regression_predict(X_train, X_test, y_train, y_test)

    mse = met.mean_squared_error(y_test, y_pred)

    print("MSE: " + str(mse) + "\n")
