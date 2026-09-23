###
# Following tutorial playlist from @sentdex on YouTube
# Playlist: Machine Learning with Python
# SVM Section (videos 25-)
###

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

# Create SVM as a class so it can be saved as an object
# and avoid retraining
class Support_Vector_Machine:
    # Set visualization setting to True by default in
    # order to see the data
    def __init__(self, visualization=True):
        self.visualization = visualization
        # Assign colors to classes
        self.colors = {1: 'r', -1: 'b'}
        # If visualizing, initialize the figure and
        # subplot for the graph
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # AKA training the data
    def fit(self, data):
        self.data = data

        # { ||w||: [w,b] }
        opt_dict = {}

        # To be applied to vector of w, transform it
        # with these values (getting the product)
        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]

        all_data = []
        # For loops access each feature in all classes
        # and add it to the all_data list
        # yi = class
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        # Initialize values for the max and min feature
        # values
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)

        # Dump all_data to free up memory after
        all_data = None

        # Find the big steps using max feature value,
        # continuously scaling down for each smaller step
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # Point of expense (useless to keep
                      # going smaller after this)
                      self.max_feature_value * 0.001]

        # Set the b-range; extremely expensive
        # (in terms of computation)
        # Don't need to take as small of steps as w
        b_range_multiple = 5
        b_multiple = 5

        # First element of vector w, which will be set
        # to max feature value multiplied by 10
        # (Saves processing cost doing this)
        latest_optimum = self.max_feature_value * 10

        # For loop to iterate through each step in the
        # step sizes list
        for step in step_sizes:
            # Vector w will be set to two same values,
            # latest optimum in this case
            w = np.array([latest_optimum, latest_optimum])

            # Can be done because of convex, False until
            # we run out of steps to take
            optimized = False
            # Iterate through b
            while not optimized:
                # arange() to define how much of a step
                # to take at a time
                # b multiples defined as 5 to reduce cost
                for b in np.arange(-1*(self.max_feature_value * b_range_multiple),
                                   self.max_feature_value * b_range_multiple,
                                   step*b_multiple):
                    # For each transformation, apply w
                    for transformation in transforms:
                        w_t = w * transforms
                        found_option = True
                        # Weakest link in algorithm
                        # i is the class
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                # If a sample does not fit definition,
                                # found_option no longer true (should
                                # be broken immediately))
                                if not yi*(np.dot(w_t, xi) + b) >= 1:
                                    found_option = False

                        # If found_option still true, all samples fit
                        # definition
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

            # "Step optimized"
            if w[0] < 0:
                optimized = True
                print('Optimized a step.')
            else:
                # Ex: [5, 5] becomes [4, 4]
                w = w - step

    # Method uses the formula for calculating the
    # prediction in SVM (sign of x.w+b)
    def predict(self, features):
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        return classification

# Initialize dictionary for data
# Keys = classes
data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),

             1:np.array([[5,1],
                         [6,-1],
                         [7,3],])}