###
# Following tutorial playlist from @sentdex on YouTube
# Playlist: Machine Learning with Python
# K-Nearest Neighbor Section (from scratch) (videos 16-)
###

import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random

# Function that defines the k-nearest neighbors algorithm
# that we will be creating and using
def k_nearest_neighbors(data, predict, k=3):
    # If we have a dataset with as much or more than
    # defined in k, send the user a warning that k is
    # supposed to be a value less than the total number
    # of classes
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    # Create a list of lists for all our distances
    distances = []

    # For loop iterates through each class in given data
    # and accesses each of their features (a point)
    # group = class
    # features = features of class
    for group in data:
        for features in data[group]:
            # A faster way of calculating utilizes numpy methods
            # to operate on a dynamic set of data as arrays
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            # Then add this distance alongside its group to
            # our distances
            distances.append([euclidean_distance, group])

    # Single-line for loop grabs the top k votes from
    # a sorted distances list (the smallest distances)
    # i[1] = group
    # [:k] splices the sorted list to cut off the first
    # k shortest distances (by default, 3)
    votes = [i[1] for i in sorted(distances)[:k]]

    # Using Counter module, the result is the most
    # common group
    # 1 = only grab THE most common, return 1 value
    # [0][0] will specifically return a tuple
    # containing the most common group and amount
    # (most_common() returns a list of tuples)
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

# Read in the data file and save it to our dataframe
df = pd.read_csv('./k-nearest-files/breast-cancer-wisconsin.data')

# Replace all '?' values with outlier value
df.replace('?', -99999, inplace=True)

# Drop the entire ID column from set
df.drop(['id'], axis=1, inplace=True)

# To correct for some values coming in as strings or
# other value, copy data where every value is converted
# to a float
full_data = df.astype(float).values.tolist()

# Use the random module to shuffle the data
random.shuffle(full_data)

# Define the size of our test and initialize both
# the train and test sets as empty lists
# (where the keys are the signifiers for benign or
# malignant tumors)
test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}

# Take number of values corresponding to the
# percentage (defined as test_size) of full_data's
# length (20% of full data in this case)
# int() call to correct final value for splicing
train_data = full_data[:-int(test_size*len(full_data))]

# *** THE FOLLOWING WAS MERELY A DEMONSTRATION ***

# Instantiate plot points for demonstrating euclidean
# distance
#plot1 = [1,3]
#plot2 = [2,5]

# Utilize the sqrt method from math module to perform
# Euclidean distance calculation on example points
#euclidean_distance = sqrt((plot1[0] - plot2[0])**2 +
#                          (plot1[1] - plot2[1])**2)

#print(euclidean_distance)