###
# Following tutorial playlist from @sentdex on YouTube
# Playlist: Machine Learning with Python
# K-Nearest Neighbor Section (videos 14-)
###

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

# Read dataframe into program from file
df = pd.read_csv('./k-nearest-files/breast-cancer-wisconsin.data')

# Replace missing data, denoted by "?", as some tangible #
# (recognized as outlier instead)
df.replace('?', -99999, inplace=True)

# Remove the id column from dataframe, will only mess up
# our calculations currently
df.drop(['id'], 1, inplace=True)

# Define Xs and ys (features and labels/class)
X = np.array(df.drop(['class'],1))  # X is everything but class
y = np.array(df['class'])

# Test and train X and y
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)