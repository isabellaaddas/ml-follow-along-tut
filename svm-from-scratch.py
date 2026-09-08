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

# Initialize dictionary for data
# Keys = classes
data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),

             1:np.array([[5,1],
                         [6,-1],
                         [7,3],])}