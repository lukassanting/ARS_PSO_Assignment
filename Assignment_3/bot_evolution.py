import pymunk_animation
import pymunk_classes
import evolutionary_algorithm
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
