###
# Following tutorial playlist from @sentdex on YouTube
# Playlist: Machine Learning with Python
# K-Nearest Neighbor Section (from scratch) (videos 16-)
###

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter

# Use 538 style for graph
style.use('fivethirtyeight')

# Create a dictionary for our dataset
# k is a class (a list of lists)
# r is a class corresponding to k's features
dataset = {'k': [[1,2], [2,3], [3,1]], 'r': [[6,5], [5,7], [8,6]]}

# Add new features for the scenario
new_features = [5,7]

# Create for loop to iterate through the classes in our
# dataset, accessing each feature in each class
# i = class ('k', 'r')
for i in dataset:
    # ii = feature set ([1,2], etc.)
    for ii in dataset[i]:
        # For every point in our dataset, scatter it on
        # our pyplot graph
        plt.scatter(ii[0], ii[1], s=100, color=i)

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