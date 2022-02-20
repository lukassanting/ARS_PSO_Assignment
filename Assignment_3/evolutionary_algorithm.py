import numpy as np
from helper import *

# ------------------------------------------------------
# ----------------- GENETIC OPERATORS ------------------
# ------------------------------------------------------

GENE_LENGTH = 99

# Selection

# https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)

def roulette_wheel_selection(fitness: np.array) -> np.array:
    '''
        Return the indices of the randomly selected individuals proportional to their FITNESS value
        Careful! One individual may be drawn multiple times
    '''

    # normalizing
    norm_fitness = fitness / sum(fitness)

    # accumulated normalized fitness values
    cum_sum = np.cumsum(norm_fitness)

    # random numbers
    rand = np.random.random(size=len(fitness))

    # select individuals
    inds = np.array([])
    for r in rand:
        inds = np.append(inds, np.argmax(cum_sum >= r))

    return np.unique(inds)

def linear_rank_selection(fitness: np.array) -> np.array:
    '''
        Return the indices of the randomly selected individuals proportional to their RANK value
        Mostly used when the individuals in the population have very close fitness values
    '''
    
    # calculate probabilities based on rank
    sorted_fitness = np.sort(fitness)

    pass

# Crossover

# to be implemented in the NN file, return all the population genotypes
def get_population():
    pass

def N_point_crossover(parent1: str, parent2: str, nr_points: int) -> str:
    assert nr_points < 10, "crossover points is too high"
    
    crossover_points = np.random.randint(1, min([len(parent1), len(parent2)]), nr_points)

    print(f'Performing {nr_points} point crossover on {parent1} and {parent2}')
    print(f'Crossover points: {crossover_points}')

    sliced_parent1, sliced_parent2, prev_point = [], [], 0

    # slices all parents
    for i, cp in enumerate(crossover_points):
        sliced_parent1.append(parent1[prev_point:cp])
        sliced_parent2.append(parent2[prev_point:cp])

        prev_point = cp

    sliced_parent1.append(parent1[prev_point:])
    sliced_parent2.append(parent2[prev_point:])

    child1, child2, prev_point = "", "", 0

    # alternate slices
    for i in range(len(sliced_parent1)):
        if (i % 2) == 0:
            child1 += sliced_parent1[i]
            child2 += sliced_parent2[i]
        else:
            child1 += sliced_parent2[i]
            child2 += sliced_parent1[i]

    return child1, child2

# Mutation


# https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm)

def bit_string_mutation(individual: str, mutation_rate=0.15) -> str:

    '''
        Bit flips of the individual genome according on the mutation rate
    '''

    mutated_individual = []
    for i in range(len(individual)):
        if np.random.random() < mutation_rate:
            mutated_individual.append(str(int(individual[i]) ^ 1))
        else:
            mutated_individual.append(individual[i])
    
    mutated_individual = "".join(mutated_individual)

    print(individual)

    return mutated_individual

