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

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

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