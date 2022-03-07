import numpy as np
from typing import List, Tuple
from numpy.random import default_rng

class population():

    def __init__(self, num_individuals:int, ann_structure:List, range_weights:Tuple=(-10,10)) -> None:

        self._fitness = np.zeros(num_individuals)
        self._genotypes = []
        for i in range(num_individuals):
            # initialize genotypes
            genotype = generate_random_genotype(ann_structure=ann_structure, borders_uniform_distribution=range_weights)
            self._genotypes.append(genotype)



def generate_random_genotype(ann_structure:List, borders_uniform_distribution:Tuple=(-10,10)) -> List[np.ndarray]:
    """Randomly generates a genotype, i.e. List of weights for the ANN.

    Args:
        ann_structure (List): Ann_structure[i] defines the number of nodes in layer i, not counting bias nodes.\
            All hidden layers have a bias node. Output layer has memory.
        borders_uniform_distribution (Tuple, optional): Defines the range of the Uniform distribution from which weights are generated\
                in case the genotype is not specified. Defaults to (-10,10).

    Returns:
        List[np.ndarray]: Genotype; Output[i][j][k] is the weight between node j of layer i and node k of layer i+1.
    """    

    assert borders_uniform_distribution[0] < borders_uniform_distribution[1], 'Borders_of_uniform[0] must be the lower border.f'
    border_low = borders_uniform_distribution[0]
    border_high = borders_uniform_distribution[1]
    genotype = []
    rng = default_rng()

    for layer_number, number_nodes in enumerate(ann_structure):

        if layer_number == len(ann_structure)-1:
            # output layer has memory
            genotype.append(rng.uniform(low=-10, high=10, size=(number_nodes, number_nodes)))
            break

        number_nodes_next_layer = ann_structure[layer_number+1]

        if layer_number == 0:
            # input layer has no bias node
            genotype.append(rng.uniform(low=-10, high=10, size=(number_nodes, number_nodes_next_layer)))
            continue

        genotype.append(rng.uniform(low=border_low, high=border_high, size=(number_nodes+1, number_nodes_next_layer)))

    return genotype