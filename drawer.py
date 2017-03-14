import matplotlib.pyplot as plt
from functools import reduce
import numpy as np
from operator import add


STEP = 0.001


def get_function_break_point(coefficients):
    if coefficients[3]:
        return -coefficients[2] / coefficients[3]
    else:
        return None

def get_function_for_plotting(coefficients):
    def f(x):
        return (-coefficients[1] * x - coefficients[0]) / (coefficients[3] * x + coefficients[2])
    return f


def get_limits(classified_data):
    all_vectors = reduce(add, classified_data.values())
    xs = list(map(lambda x: x[0], all_vectors))
    ys = list(map(lambda x: x[1], all_vectors))
    return {
        'x': (min(xs) - 1, max(xs) + 1),
        'y': (min(ys) - 1, max(ys) + 1),
    }


def show_results(coefficiens, classified_data):
    limits = get_limits(classified_data)
    plt.xlim(limits['x'])
    plt.ylim(limits['y'])
    plot_function(coefficiens, limits['x'])
    draw_dots(classified_data)
    plt.show()


def draw_dots(classified_data):
    colors = {True: 'green', False: 'red'}
    for decision, vectors in classified_data.items():
        xs = list(map(lambda x: x[0], vectors))
        ys = list(map(lambda x: x[1], vectors))
        plt.scatter(xs, ys, c=colors[decision], marker='.')

def is_intersecting_lines(coefficients):
    return coefficients[2] * coefficients[1] == coefficients[3] * coefficients[0]

def plot_function(coefficients, interval):
    function = get_function_for_plotting(coefficients)
    break_point = get_function_break_point(coefficients)
    interval_start, interval_end = interval
    if break_point is not None:
        arange = np.arange(interval_start, break_point - STEP, STEP)
        plt.plot(arange, function(arange), color='b')
        arange = np.arange(break_point + STEP, interval_end, STEP)
        plt.plot(arange, function(arange), color='b')
        if is_intersecting_lines(coefficients):
            plt.axvline(break_point, color='b')
    else:
        arange = np.arange(interval_start, interval_end, STEP)
        plt.plot(arange, function(arange), color='b')
