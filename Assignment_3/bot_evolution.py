from pymunk_animation import *
from evolutionary_algorithm import *
import pymunk_animation
from animation import *
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import pymunk_classes
import evolutionary_algorithm
import struct
import tqdm
import helper

# 1. Initialisation
# 1.1 Make a Population of weights (either new class, or use old one....)
# 1.2 For each of these weights, make an ANN (in the class itself?)
# 1.3 For each of these ANNs, make a pymunk_bot with the ANN as an element

# 2. Walkabout and Evaluation
# 2.1 Run a simulation for a limited time (i.e. i=50,000) each element of the population
#     Edit class so that if ANN=/=empty, then uses ANN for movement instead of keys
# 2.2 When time is up, evaluate the fitness of the robot.
# 2.3 Store the returned fitnesses in the population

# 3. Evolution
# 3.1 From this population create a new generation using crossover, mutation, etc
# 3.2 Repeat steps 1.2 onwards with this new generation of weights
# 3.3 Repeat this for as many generations as specified


pygame.init()

display_width = 1000
display_height = 600
grid_size = 20

pygame_display = pygame.display.set_mode((display_width, display_height))
pymunk_space = pymunk.Space()  # Holds our pymunk physics space
dust_grid = np.zeros((int(display_width / grid_size), int(display_height / grid_size)))

# positional value variables
bot_radius = 20

# set colours for display
white = (255, 255, 255)
black = (0, 0, 0)
pygame_display.fill(white)

# Define the locations of obstacle edges, used both pygame/pymunk, & for sensors (for intersection & collision checks)
edge_north = [(10, 5), (990, 5)]
edge_south = [(200, 595), (990, 595)]
edge_west = [(5, 0), (5, 400)]
edge_east = [(990, 0), (990, 600)]
edge_sw = [(5, 400), (200, 595)]
edge_mid_n = [(350, 10), (350, 100)]
edge_mid_s = [(350, 250), (350, 595)]
edges = [edge_north, edge_south, edge_east, edge_west, edge_sw, edge_mid_s, edge_mid_n]

pymunk_walls = []
for edge in edges:
    pymunk_walls.append(Pymunk_Obstacle(pygame_display=pygame_display, pymunk_space=pymunk_space, radius=10, color=black, p=edge))

#  --------- fitness_func

def bot_fitness(pymunk_bot):
    collision_count = pymunk_bot.collision_counter
    dust_count = pymunk_bot.get_dust_count()
    return dust_count / collision_count+1

# ------------- calling evolution --------------

population = Population(num_individuals=40, ann_layers=(8,8,4,2), bias_nodes=(False,True,True,False), fitness_func=bot_fitness)
population.bot_evolution(population, edges, pymunk_walls, pygame_display, pymunk_space, num_generations=80)
fig = population._history.plot_fitness()
plt.show()
