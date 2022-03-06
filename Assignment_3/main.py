
import numpy as np
import sys
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ann import *
import helper
from evolutionary_algorithm import *

# ***************** STEP 1 - DEFINE FUNCTION TO OPTIMISE HERE:
opt_func = "rosenbrock"  # Set to "rosenbrock" or to "rastigrin"

# ---------------- FUNCTIONS -------------------
#  Rosenbrock function to optimise
def rosenbrock(x, y, a=0, b=150):
    return ((a - x) ** 2) + b * ((y - (x ** 2)) ** 2)

def neg_rosenbrock(x, y, a=0, b=150):
    return (-1)*((a - x) ** 2) + b * ((y - (x ** 2)) ** 2)


def rosenbrock_grad(x, y, a=1, b=150):
    return np.array([(2 * (x - a) - 4*b*x * (y - (x**2))), (2*b * (y - (x**2)))])


def rastigrin(x, y):
    return 20 + x ** 2 - 10 * np.cos(2 * np.pi * x) + y ** 2 - 10 * np.cos(2 * np.pi * y)


def rastigrin_grad(x, y):
    return [2 * x + 20 * np.pi * np.sin(2 * np.pi * x), 2 * y + 20 * np.pi * np.sin(2 * np.pi * y)]

def rosenbrock_modified(x, y, a=0, b=150):
    return 1/(((a - x) ** 2) + b * ((y - (x ** 2)) ** 2)+1e-3)

#  ---------------- EVOLUTION -------------------

population = Population(num_individuals=30, ann_layers=(2,10,2), bias_nodes=(False,True,True,False), fitness_func=rosenbrock_modified)
population.evolution(num_generations=20, time_for_generation=30, get_ann_inputs=rosenbrock_grad, width=1, mutation_rate=0)
fig = population._history.plot_fitness()
plt.show()
anim = population._history.animate_positions()
