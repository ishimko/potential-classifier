import matplotlib.pyplot as plt
from functools import reduce
import numpy as np
from operator import add


STEP = 0.001


def get_limits(classified_data):
    all_vectors = reduce(add, classified_data.values())
    xs = list(map(lambda x: x[0], all_vectors))
    ys = list(map(lambda x: x[1], all_vectors))
    return {
        'x': (min(xs) - 1, max(xs) + 1),
        'y': (min(ys) - 1, max(ys) + 1),
    }


def show_results(function, break_point, classified_data):
    limits = get_limits(classified_data)
    plt.xlim(limits['x'])
    plt.ylim(limits['y'])
    plot_function(function, break_point, limits['x'])    
    draw_dots(classified_data)
    plt.show()


def draw_dots(classified_data):
    colors = {True: 'green', False: 'red'}
    for decision, vectors in classified_data.items():
        xs = list(map(lambda x: x[0], vectors))
        ys = list(map(lambda x: x[1], vectors))
        plt.scatter(xs, ys, c=colors[decision], marker='.')


def plot_function(function, break_point, interval):
    interval_start, interval_end = interval
    if break_point is not None:
        arange = np.arange(interval_start, break_point - STEP, STEP)
        plt.plot(arange, function(arange), color='b')
        arange = np.arange(break_point + STEP, interval_end, STEP)
        plt.plot(arange, function(arange), color='b')
    else:
        arange = np.arange(interval_start, interval_end, STEP)
        plt.plot(arange, function(arange), color='b')
