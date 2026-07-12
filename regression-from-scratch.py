###
# Following tutorial playlist from @sentdex on YouTube
# Playlist: Machine Learning with Python
# Regression (from scratch) Section (videos 8-)
###

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

# Visualize data with plot; set the style
style.use('fivethirtyeight')

# Use small sets of data for this algorithm (to start)
# Note: explicitly setting data type for later usage
xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

# Translate the formula for finding slope (m)

# Define function that will find the best fit slope
# from our data (m) *and intercept (b)
def best_fit_slope_and_intercept(xs, ys):
    # Define the formula using mean function and
    # power operator
    m = ( ((mean(xs) * mean(ys)) - mean(xs * ys)) /
          ((mean(xs) ** 2) - mean(xs ** 2)) )

    # Define the formula using calculated m
    b = mean(ys) - m * mean(xs)

    return m, b

# Function finds squared error to use for determining how
# good our best-fit line is for our data
def squared_error(ys_orig, ys_line):
    # Return squared error of entire line
    return sum((ys_line - ys_orig) ** 2)

# Function finds coefficient of determination
def coefficient_of_determination(ys_orig, ys_line):
    # Calculates the mean of every individual y value
    # in our data, creating a new line of all means
    y_mean_line = [mean(ys_orig) for y in ys_orig]

    # Find squared error of the regression line and the
    # y mean line
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)

    # Use both variables to find r^2 (coefficient needed)
    return 1 - (squared_error_regr / squared_error_y_mean)

m, b = best_fit_slope_and_intercept(xs, ys)

#print(m, b)

# For loop will create regression line for all
# xs in example data using slope-intercept form
regression_line = [(m*x)+b for x in xs]

# Example prediction
predict_x = 8
predict_y = (m * predict_x) + b

r_squared = coefficient_of_determination(ys, regression_line)

#print(r_squared)

# Plot our points and new regression line
plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='red')
plt.plot(xs, regression_line)
plt.show()