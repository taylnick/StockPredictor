import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import tensorflow as tf
from tensorflow import keras

data_dir = 'Data/Stocks'
all_files = os.listdir(data_dir)

ratio_requested = .1
num_requested = int(len(all_files) * ratio_requested)

subset = np.random.choice(all_files, num_requested)

# print('Selected subset of ' + str(len(subset)) + ' files out of ' + str(len(all_files)))
# print(subset)

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

def getX(file):

    text_file = open(data_dir + '/' + file, 'r')
    print(file)
    lines = text_file.readlines()

    X = [None] * (len(lines) - 1)

    # remove newlines https://stackoverflow.com/questions/43447221/removing-all-spaces-in-text-file-with-python-3-x
    lines = [line.replace('\n', '') for line in lines]

    for j, line in enumerate(lines):
        if j == 0:
            continue

        split_line = line.split(',')

        dat = [None] * 5
        # dat[0] = split_line[0]
        dat[0] = float(split_line[1])
        dat[1] = float(split_line[2])
        dat[2] = float(split_line[3])
        dat[3] = float(split_line[4])
        dat[4] = int(split_line[5])
        # dat[6] = int(split_line[6])

        X[j-1] = dat
        print(", ".join(map(str, dat)))

    return X


test_file = subset[0]
X = getX(test_file)

window_size = 25
knn = KNeighborsRegressor(n_neighbors=7)

# iterate this
window_start = 0
window_end = window_start + window_size

samples = X[window_start:window_end]

y = X[window_end + 1:window_end +1 + window_size]

knn.fit(samples, y)

X_test = X[window_start + 1:window_end + 1]
prediction = knn.predict(X_test)
y_real = X[window_end + 1]

print("Input: ")
for x in X_test:
    print(",".join(map(str,x)))

print("Output: \n" + ",".join(map(str,prediction[-1])))
print("Actual: \n" + ",".join(map(str, y_real)))