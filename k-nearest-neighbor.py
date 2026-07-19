###
# Following tutorial playlist from @sentdex on YouTube
# Playlist: Machine Learning with Python
# K-Nearest Neighbor Section (videos 14-)
###

import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import cross_validate, train_test_split
import pandas as pd

# Read dataframe into program from file
df = pd.read_csv('./k-nearest-files/breast-cancer-wisconsin.data')

# Replace missing data, denoted by "?", as some tangible #
# (recognized as outlier instead)
df.replace('?', -99999, inplace=True)

# Remove the id column from dataframe, will only mess up
# our calculations currently
df.drop(['id'], axis=1, inplace=True)

# Define Xs and ys (features and labels/class)
X = np.array(df.drop(['class'], axis=1))  # X is everything but class
y = np.array(df['class'])

# Test and train X and y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Set up our classifier
clf = neighbors.KNeighborsClassifier()
# Then fit with trained data
clf.fit(X_train, y_train)
# And score on test data
accuracy = clf.score(X_test, y_test)

#print(accuracy)

# Make an example to predict with
example_measures = np.array([4,2,1,1,1,2,3,2,1])
# Reshape data (important for data with single sample)
example_measures = example_measures.reshape(1,-1)
# Use predict method to predict with example
prediction = clf.predict(example_measures)

print(prediction)