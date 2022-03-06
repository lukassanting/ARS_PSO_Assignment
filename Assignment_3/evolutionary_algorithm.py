import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import func
import helper
import struct
from typing import Tuple, List
import tqdm
from animation import *

# https://stackoverflow.com/questions/8751653/how-to-convert-a-binary-string-into-a-float-value?noredirect=1&lq=1

# return array of binarized weights
# def weights_to_binary(weights: np.array) -> np.array:
#     return np.array([float_to_binary(w) for w in weights])

class History():

    def __init__(self) -> None:
        self._num_generations = 0
        self._positions = None # each list element stores the positions of all individuals of a generation
        self._fitness = None # each listelement stores the fitness of all individuals of a generation
        self._avg_fitness = []
        self._sd_avg_fitness = [] # standard deviation of average
        self._fitness_best = []
        self._genotypes_best = None


    def add_generation_to_history(self, population, pos) -> None:
        if self._positions is None:
            self._positions = np.array([pos])
        else: self._positions = np.concatenate((self._positions, np.array([pos])), axis=0)

        fitness_values = population.get_all_fitness()
        if self._fitness is None:
            self._fitness = np.array([fitness_values])
        else: self._fitness = np.concatenate((self._fitness, np.array([fitness_values])), axis=0)

        self.store_best_in_generation(population.get_all_genotypes_float())

        self._avg_fitness.append(np.mean(fitness_values))
        self._sd_avg_fitness.append(np.std(fitness_values))


    def store_best_in_generation(self, genotypes) -> None:
        # requires the fitness of the new generation to be stored as the last element in self._fitness
        # if multiple individuals have the highest fitness in a population, only the first one is kept
        index_fittest = np.argmax(self._fitness[-1])
        self._fitness_best.append(self._fitness[-1][index_fittest])

        if self._fitness[-1][index_fittest] > np.max(np.array(self._fitness_best)):
            if self._genotype_historic_best is None:
                self._genotype_historic_best = np.array([genotypes[index_fittest]])                
            else: self._genotype_historic_best = np.concatenate((self._genotype_historic_best, np.array([genotypes[index_fittest]])), axis=0)

    def animate_positions(self):
        # print(self._positions)
        print(len(self._positions))
        for i in range(len(self._positions)):
            print("testprint")
            position_animation = Animation("neg_rosenbrock", self._positions[i], title=f"generation_{i}")
            anim = position_animation.animate()

    def plot_fitness(self):
        fig = plt.figure()
        plt.errorbar(range(len(self._avg_fitness)), self._avg_fitness, self._sd_avg_fitness)
        plt.errorbar(range(len(self._fitness_best)), self._fitness_best)
        # plt.legend(loc='lower right')
        return fig



class Population():

    def __init__(self, num_individuals, ann_layers:Tuple[int], bias_nodes:Tuple[bool], fitness_func:func) -> None:
        self._size = num_individuals
        self._layers = ann_layers
        self._bias = bias_nodes
        self._fit_func = fitness_func
        self._fit_func_dim = self._fit_func.__code__.co_argcount-len(self._fit_func.__defaults__)
        self._indiv_gene_length = helper.get_network_size(self._layers)
        self._individuals = np.array([Individual(num_genes=self._indiv_gene_length) for i in range(self._size)])
        self._history = History()


    def get_all_fitness(self) -> np.ndarray:
        ''' returns an array with all fitness values of the individuals '''
        fitness = np.array([ind.fitness for ind in self._individuals])
        return fitness

    
    def get_all_genotypes_float(self) -> np.ndarray:
        genotypes = np.array([ind.float_genotype for ind in self._individuals])
        return genotypes


    def update_fitness(self, positions: np.ndarray) -> None:
        """Updates the fitness of all individuals in the population.

        Args:
            positions (np.ndarray): Multi-dimensional array of coordinates of each individual. In case of 2D, row 1 should have x-coordinates
                                    and row 2 should have y-coordinates.
        """        

        assert positions.shape == (self._fit_func_dim, self._size), 'Dimensions of positions do not match fitness function dimensions and population size'

        for indiv_number, individual in enumerate(self._individuals):
            coordinates = [positions[dim][indiv_number] for dim in range(self._fit_func_dim)]
            individual.update_fitness(self._fit_func(*coordinates))
            print(f'Individual no. {indiv_number} fitness is: {individual.fitness}')


    def initial_positions(self, center:np.ndarray=None, width:float=1) -> np.ndarray:
        """Generates initial positions for individuals in a "num_dimensions"-dimensional space.
        Coordinates are drawn from a uniform random distribution centered at "center" width of intervall_length

        Args:
            center (np.ndarray): center of the uniform distribution
            width (float): width of the uniform distribution

        Returns:
            np.ndarray: starting positions for "self._size" individuals to start their movements
        """ 
        if center is None:
            center = np.zeros(self._fit_func_dim)    

        assert center.ndim == 1, 'Center only accepts one-dimensional arrays.'
        assert center.shape[0] == self._fit_func_dim, 'Number of dimensions does not match dimensionality of center for the uniform distribution'

        XY = np.random.rand(self._fit_func_dim, self._size) * width
        # print(f'initial coordinates matrix {XY}') (works)
        center_shift = np.array([np.ones(self._size)*c for c in center])
        XY = np.add(XY, center_shift)
        # print(f'coordinate matrix after applying shift {XY}') (works)
        return XY


    def lifecycle(self, time:int, get_ann_inputs:func, update_rate:float=1/50, center:np.ndarray=None, width:float=1, max_velocity:float=None) -> np.ndarray:
        """Initializes ANNs according to Genotypes of the individuals and let the individuals move. After a set number of
        iterations, the fitness of every individual is updated.

        Args:
            time (int): determines together with update_rate how many times the individuals are allowed to move
            get_ann_inputs (func): function that takes the position of the individual as an input and returns 
            the values that should be passed as inputs to the ANN (e.g. the gradients for the benchmark functions) or 
            the distance sensor measurements for the robot.

            update_rate (float, optional): needs to be 1/XX where XX is an integer. Defaults to 1/50.
            center (np.ndarray, optional): see description of initial_position. Defaults to None.
            width (float, optional): see description of initial_position. Defaults to 1.
            max_velocity (np.float, optional): maximum velocity in the x and y direction with which the individuals
            will move per timestep. This number will be taken as the maximum the minimum will be the number mulitplied by -1.

        Returns:
            np.ndarray: final positions of all individuals
        """
        if max_velocity is None:
            max_velocity = 1
        
        assert max_velocity > 0, 'Specify a positive maximum velocity. Otherwise the minimu velocity is ill-defined.'

        networks = [helper.array_to_network(individual.float_genotype, self._layers, self._bias) for individual in self._individuals]
        pos = self.initial_positions(center, width)
        pos_generation = np.array([pos])

        for step in range(int(time / update_rate)):
            for (pos_x, pos_y, network) in zip(pos[0], pos[1], networks):
                inputs = get_ann_inputs(pos_x, pos_y)
                velocity = network.prop_forward(inputs)
                velocity_capped = np.clip(velocity, a_min = (-1)*max_velocity, a_max=max_velocity)

                pos_x += update_rate*velocity_capped[0]
                pos_y += update_rate*velocity_capped[1]
            pos_generation = np.concatenate((pos_generation, np.array([pos])), axis=0)
        
        # self.update_fitness(pos)
        self.update_fitness(pos_generation[0])
        return pos_generation


    def generational_change(self, mutation_rate:float=0.001, verbose=False) -> None:
        '''
            Performs all the genetic processes.
        '''
        
        # Selection
        if verbose: print("Applying selection...")
        
        selected_individuals = list(set(linear_rank_selection(self._individuals)))
        # make sure at least 2 individuals are selected
        while len(selected_individuals) < 2:
            selected_individuals = list(set(linear_rank_selection(self._individuals)))

        # for debugging
        for index, indiv in enumerate(selected_individuals):
            print(f'Fitness of selected individual {index}: {indiv._fitness}')

        if verbose: print("Selected {} out of {} initial individuals".format(len(selected_individuals), len(self._individuals)))

        # Reproduction/crossover
        needed = len(self._individuals) - len(selected_individuals)
        
        offsprings = []
        for child in range(needed):
            # randomly choose two parents for the reproduction
            p1_i, p2_i = np.random.choice(range(0, len(selected_individuals)), 2, replace=False)
            children = N_point_crossover(selected_individuals[p1_i], selected_individuals[p2_i], 1)
            
            # discard one child >:)
            selected_child = children[np.random.randint(0, 1)]
            offsprings.append(selected_child)
        
        new_population = selected_individuals + offsprings
        
        for ind in new_population:
            ind = bit_string_mutation(ind, mutation_rate=mutation_rate)
        
        self._individuals = new_population


    def evolution(self, num_generations:int, time_for_generation:int, get_ann_inputs:func, update_rate:float=1/50,
                        center:np.ndarray=None, width:float=1, mutation_rate=0.001, verbose=False) -> None:
        """Performs the entire evolutionary algorithm.

        Args:
            num_generations (int): number of generations before the algorithm ends
            time_for_generation (int): see description of "time" parameter in lifecycle
            get_ann_inputs (func): see description of lifecycle
            update_rate (float): see description of lifecycle
            center (np.ndarray, optional): see description of initial_position. Defaults to None.
            width (float, optional): see description of initial_position. Defaults to 1.
        """
        for i in tqdm.trange(num_generations):
            pos_history = self.lifecycle(time_for_generation, get_ann_inputs, update_rate, center, width)
            self._history.add_generation_to_history(self, pos_history)
            self.generational_change(mutation_rate, verbose)

            

class Individual():
    
    def __init__(self, binary_genotype=None, num_genes=None, fitness=None) -> None:

        # randomly initialize genotype if it is no specified
        if binary_genotype is None:
            assert num_genes is not None, 'Neither binary_genotype nor number of genes is specified. Need\
            to specify at least one of the two.'
            self._num_genes = num_genes
            self._float_genotype = np.random.uniform(size=num_genes)
            self._binary_genotype = helper.array_to_binary(self._float_genotype)

        else:
            self._binary_genotype = binary_genotype
            self._float_genotype = helper.array_to_float(self._binary_genotype)
            self._num_genes = self._float_genotype.shape[0]

        self._fitness = fitness
 


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
    def num_genes(self):
        return self._num_genes

    def update_fitness(self, fitness):
        self._fitness = fitness

# General methods

def get_all_fitness(individuals: np.ndarray) -> np.ndarray:
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

def roulette_wheel_selection(individuals: np.ndarray) -> np.ndarray:
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

def linear_rank_selection(individuals: np.ndarray) -> np.ndarray:
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

def N_point_crossover(parent1: Individual, parent2: Individual, nr_points: int) -> Tuple:
    assert nr_points < 10, "crossover points are too high"
    
    # concatenate all genes
    genotype_parent1 = "".join(parent1._binary_genotype)
    genotype_parent2 = "".join(parent2._binary_genotype)

    crossover_points = np.random.randint(1, min([len(genotype_parent1), len(genotype_parent2)]), nr_points)

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

    child1 = Individual(binary_genotype=genotype_to_genes(child1, 32))
    child2 = Individual(binary_genotype=genotype_to_genes(child2, 32))

    return (child1, child2)

# Mutation

# https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm)

def bit_string_mutation(individual: Individual, mutation_rate=0.001) -> Individual:

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

    mutated_individual = Individual(binary_genotype=genotype_to_genes(mutated_individual, 32))

    return mutated_individual


POPULATION_SIZE = 10
GENOTYPE_LENGTH = 10

# individual = Individual(np.random.randint(100), GENOTYPE_LENGTH)
# individual2 = Individual(np.random.randint(100), GENOTYPE_LENGTH)
# print(individual.float_genotype)
# print(individual2.float_genotype)
#
#
# child, child2 = N_point_crossover(individual, individual2, 5)
# print(child.float_genotype)
# print(child2.float_genotype)
#
# population = [Individual(np.random.randint(100), GENOTYPE_LENGTH) for i in range(POPULATION_SIZE)]
#
# # dominant_pop = roulette_wheel_selection(inds)
# dominant_pop = linear_rank_selection(population)
#
# # get unique individuals
# dominant_pop = list(set(dominant_pop))
#
# # crossover 2 dominant individuals and get 2 children
# offsprings = N_point_crossover(dominant_pop[0], dominant_pop[1], 1)
# verbose_individuals_fitness(offsprings, True)
#
# # mutate 1 individual
#
# print(bit_string_mutation(offsprings[0]))