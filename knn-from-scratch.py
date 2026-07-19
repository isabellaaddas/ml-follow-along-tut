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