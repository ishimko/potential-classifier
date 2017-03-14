import matplotlib.pyplot as plt
import numpy as np

STEP = 0.01

def plot_function_with_break_point(function, break_point, interval):
    interval_start, interval_end = interval
    arange = np.arange(interval_start, break_point - STEP, STEP)
    plt.plot(arange, function(arange), color='b')
    arange = np.arange(break_point + STEP, interval_end, STEP)
    plt.plot(arange, function(arange), color='b')
    plt.show()
