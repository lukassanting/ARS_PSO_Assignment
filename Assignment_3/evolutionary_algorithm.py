import numpy as np
import helper
import struct

# https://stackoverflow.com/questions/8751653/how-to-convert-a-binary-string-into-a-float-value?noredirect=1&lq=1

# return array of binarized weights
# def weights_to_binary(weights: np.array) -> np.array:
#     return np.array([float_to_binary(w) for w in weights])

class Individual():
    
    def __init__(self, fitness=0, size=92, from_genes=False, bin_genes=[]) -> None:
        self._float_genotype = np.random.uniform(size=size)
        self._binary_genotype = helper.array_to_binary(self._float_genotype)
        self._fitness = fitness
        self._nr_genes = size

        # if the list of genes is foreknown construct the individual from them
        if from_genes:
            self._binary_genotype = bin_genes
            print(bin_genes)
            self._float_genotype = helper.array_to_float(self._binary_genotype)

    def __str__(self) -> str:
        return f'Binary implementation: {self.binary_genotype} \n Float implementation: {self._float_genotype} \n Fitness: {self._fitness}'

    @property
    def float_genotype(self):
        return self._float_genotype

    @property
    def binary_genotype(self):
        return self._binary_genotype

    @property
    def fitness(self):
        return self._fitness

    @property
    def nr_genes(self):
        return self._nr_genes

    def update_fitness(self, fitness):
        self._fitness = fitness

# General methods

def get_all_fitness(individuals: np.array) -> np.array:
    ''' returns an array with all fitness values of the individuals '''
    
    fitness = np.array([ind.fitness for ind in individuals])

    return fitness

def verbose_individuals_fitness(individuals: np.array, bin_genes=False, float_genes=False) -> None:
    ''' prints out individuals id and fitness, and genes optionally '''

    for ind in individuals:
        print(f'Id: {id(ind)} Fitness: {ind.fitness}')

        if bin_genes:
            print(f"Binary genes: {ind.binary_genotype}")

        if float_genes:
            print(f"Binary genes: {ind.float_genotype}")

def genotype_to_genes(genotype: str, gene_length: int) -> np.array:
    ''' converts genotype chain into list of genes '''

    return np.array([genotype[i:i+gene_length] for i in range(0, len(genotype), gene_length)])

def chained_genotype(genes: np.array):
    ''' converts genes list to chained genotype'''
    chained_genes = "".join(genes)
    
    return chained_genes


# ------------------------------------------------------
# ----------------- GENETIC OPERATORS ------------------
# ------------------------------------------------------

# Selection

# https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)

def roulette_wheel_selection(individuals: np.array) -> np.array:
    '''
        Return the randomly selected individuals proportional to their FITNESS value
        Careful! One individual may be drawn multiple times
    '''

    # normalizing
    fitness_sum = sum(get_all_fitness(individuals))
    assert fitness_sum > 0, "Fitness sum is zero"

    norm_fitness = get_all_fitness(individuals) / fitness_sum

    # accumulated normalized fitness values
    cum_sum = np.cumsum(norm_fitness)

    # random numbers
    rand = np.random.random(size=len(individuals))

    # select individuals
    inds = []
    for r in rand:
        inds.append(int(np.argmax(cum_sum >= r)))
    
    selected_individuals = np.take(individuals, inds)

    return selected_individuals

def linear_rank_selection(individuals: np.array) -> np.array:
    '''
        Return the randomly selected individuals proportional to their RANK value
        Mostly used when the individuals in the population have very close fitness values
    '''
    
    # rank individuals
    sorted_individuals = sorted(individuals, key=lambda x: x.fitness)

    # calculate probabilities based on rank
    n = len(sorted_individuals)
    probs = np.array([i for i in range(1, n+1)]) / (n * (n+1) / 2)

    # accumulated normalized fitness values
    cum_sum = np.cumsum(probs)

    # random numbers
    rand = np.random.random(size=n)

    # select individuals
    inds = []
    for r in rand:
        inds.append(int(np.argmax(cum_sum >= r)))

    selected_individuals = np.take(sorted_individuals, inds)

    return selected_individuals

# Crossover

def N_point_crossover(parent1: Individual, parent2: Individual, nr_points: int) -> tuple:
    assert nr_points < 10, "crossover points are too high"
    
    # concatenate all genes
    genotype_parent1 = "".join(parent1._binary_genotype)
    genotype_parent2 = "".join(parent2._binary_genotype)

    crossover_points = np.random.randint(1, min([len(genotype_parent1), len(genotype_parent2)]), nr_points)

    print(f'Performing {nr_points} point crossover on {id(parent1)} and {id(parent2)}')
    print(f'Crossover points: {crossover_points}')

    sliced_parent1, sliced_parent2, prev_point = [], [], 0

    # slices all parents
    for i, cp in enumerate(crossover_points):
        sliced_parent1.append(genotype_parent1[prev_point:cp])
        sliced_parent2.append(genotype_parent2[prev_point:cp])
        prev_point = cp

    sliced_parent1.append(genotype_parent1[cp:])
    sliced_parent2.append(genotype_parent2[cp:])

    child1, child2 = "", ""

    # alternate slices
    for i in range(len(sliced_parent1)):
        if (i % 2) == 0:
            child1 += sliced_parent1[i]
            child2 += sliced_parent2[i]
        else:
            child1 += sliced_parent2[i]
            child2 += sliced_parent1[i]

    child1 = Individual(from_genes=True, bin_genes=genotype_to_genes(child1, 32))
    child2 = Individual(from_genes=True, bin_genes=genotype_to_genes(child2, 32))

    return (child1, child2)

# Mutation

# https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm)

def bit_string_mutation(individual: Individual, mutation_rate=0.05) -> str:

    '''
        Bit flips of the individual genome according on the mutation rate (probability)
    '''

    dna = chained_genotype(individual._binary_genotype)

    mutated_individual = ""
    for i in range(len(dna)):
        if np.random.random() < mutation_rate:
            mutated_individual += str(int(dna[i]) ^ 1)
        else:
            mutated_individual += dna[i]

    mutated_individual = Individual(from_genes=True, bin_genes=genotype_to_genes(mutated_individual, 32))

    return mutated_individual


POPULATION_SIZE = 10
GENOTYPE_LENGTH = 10

population = [Individual(np.random.randint(100), GENOTYPE_LENGTH) for i in range(POPULATION_SIZE)]

# dominant_pop = roulette_wheel_selection(inds)
dominant_pop = linear_rank_selection(population)

# get unique individuals
dominant_pop = list(set(dominant_pop))

# crossover 2 dominant individuals and get 2 children
offsprings = N_point_crossover(dominant_pop[0], dominant_pop[1], 1)
verbose_individuals_fitness(offsprings, True)

# mutate 1 individual

print(bit_string_mutation(offsprings[0]))
