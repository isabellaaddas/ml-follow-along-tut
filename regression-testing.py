###
# Following tutorial playlist from @sentdex on YouTube
# Playlist: Machine Learning with Python
# Regression Section (videos 2-)
###

# Using stock dataset from Nasdaq Data Link, practicing with Google
import math
import pandas as pd
import nasdaqdatalink as ndl
import numpy as np
# preprocessing for scaling and speed, cross_validate for
# creating testing and training samples, svm for regression
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LinearRegression

ndl.ApiConfig.api_key = "UKgdknEoTdVZdSv9PKLf"

df = ndl.get_table(datatable_code="WIKI/PRICES",
                              ticker="TSLA",
                              paginate=True,
                              api_key=ndl.ApiConfig.api_key)
df = df[['open', 'high', 'low', 'close', 'volume']]

# Newly defined columns to track daily changes
df['HL_PCT'] = (df['high'] - df['close']) / df['close'] * 100.0
df['PCT_change'] = (df['close'] - df['open']) / df['open'] * 100.0

# Columns to keep / use
df = df[['close', 'HL_PCT', 'PCT_change', 'volume']]

# The forecast column (prediction)
forecast_col = 'close'

# Fill in missing data with some value
# Cannot use NAN data with ML, must be replaced in some
# way to be treated as outlier
df.fillna(-9999, inplace=True)

# Generally, regression is for forecasting out
# math.ceil used like this will round up the length of
# df to predict the price out some number of days
forecast_out = int(math.ceil(0.01 * len(df)))

# Labels!
# Shift the columns negatively so it shows X number of
# days into the future
df['label'] = df[forecast_col].shift(-forecast_out)

# Features = X
# Labels = y
X = np.array(df.drop(['label'],axis=1))
X = X[:-forecast_out]
# What we are predicting against, values we aren't using
X_lately = X[-forecast_out:]

# Scale our features together alongside all other values
X = preprocessing.scale(X)

df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])

# Here, we will train and test our data using train_test_split
# to shuffle the data and use about 20% of it for training
# and testing purposes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define classifier and train/fit using features and labels
clf = LinearRegression(n_jobs=-1)
#clf = svm.SVR(kernel='poly') <-- example of switching algorithms
# Fit = train
# Score = test
clf.fit(X_train, y_train)
accuracy_score = clf.score(X_test, y_test)
#print(accuracy_score)

# Passing an array of values - predicting per value
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy_score, forecast_out)