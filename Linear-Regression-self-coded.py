# Computing best line from scratch
# Sourabh Kulkarni

from statistics import mean
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style
import random

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def create_dataset(amount, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(amount):
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation == "pos":
            val += step
        elif correlation and correlation == "neg":
            val -= step
    xs = [i for i in range(amount)]
    return np.array(xs, dtype=np.float64),np.array(ys, dtype = np.float64)

#function for calculating best fit slope and intercept
def best_fit_line(xs,ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs*ys)) / (((mean(xs)**2) - mean(xs**2))) )
    #m = (((mean(xs)*mean(ys)) - mean(xs*ys))/((mean(xs)*mean(xs))-(mean(xs*xs))))
    b = mean(ys) - m*mean(xs)
    return m, b 

def squared_error(ys_orig, ys_new):
    return sum((ys_new -ys_orig)**2)

def coefficient_of_determination(ys_orig,ys_new):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig,ys_new)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

xs,ys = create_dataset(40, 40, 2, correlation='neg')

m, b = best_fit_line(xs,ys)

ys_new = m*xs + b
r_squared = coefficient_of_determination(ys,ys_new)

print (r_squared)
# Plot the best fit line

plt.scatter(xs,ys)
plt.plot(xs, ys_new)
plt.show()

# R squared error computation
