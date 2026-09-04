###
# Following tutorial playlist from @sentdex on YouTube
# Playlist: Machine Learning with Python
# SVM Section (videos 25-)
###

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

# Initialize dictionary for data where the keys are
# classes
data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),
             1:np.array([[1,7],])}