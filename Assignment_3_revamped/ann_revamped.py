from attr import attrib
import numpy as np
from typing import List, Tuple
from scipy.special import expit as sigmoid
from numpy.random import default_rng
from population import generate_random_genotype

# ---------- Artificial Neural Network Class ---------- #

class Ann():

    def __init__(self, genotype:List[np.ndarray]=None, structure:List=None, range_weights:Tuple=(-10,10)) -> None:
        """ANN with no recurrent connections (except for memory nodes for last layer). All hidden layers have bias nodes.

        Args:
            genotype (List[np.ndarray], optional): List of weights for the ANN. Genotype[i][j][k] is the weight\
                between node j of layer i and node k of layer i+1. Defaults to None, in which case weights will\
                be randomly generated according to structure from a uniform distribution with border range_weights.
            structure (List, optional): Defines the number of nodes per layer. Remains unused if genotype is specified.\
                Defaults to None.
            range_weights(Tuple, optional): Defines the range of the Uniform distribution from which weights are generated\
                in case the genotype is not specified. Defaults to (-10,10).
        """

        assert (genotype is not None or structure is not None), 'ANN not specified. Either provide genotype or structur.'

        self._genotype = genotype        

        if self._genotype is None:
            self._genotype = generate_random_genotype(ann_structure=structure, borders_uniform_distribution=range_weights)

        self._previous_output = np.zeros(2) # needed for memory of output layer


    def prop_forward(self, input:np.ndarray) -> List[np.ndarray]:
        """Propagates inputs through the network.

        Args:
            input (np.ndarray): _description_

        Returns:
            List[np.ndarray]: Activations of all nodes in all layers. Output[i][j] is activation of the j'th node in the i'th layer.\
                For a hidden layer k, Output[k][1] will always be =1 to represent the bias node.
        """

        assert np.squeeze(input).ndim == 1, 'Input must be 1D array.'
        assert input.shape[0] == self._genotype[0].shape[0], 'Input dimension does not match number of nodes in input layer.'      

        node_activations = [input]

        for layer_number, weights in enumerate(self._genotype):

            if layer_number == len(self._genotype)-2:
                inputs_from_previous_layer = np.matmul(node_activations[layer_number], weights)
                # adding memory effect
                memory_weights = self._genotype[-1]
                memory_effect = np.matmul(self._previous_output, memory_weights)
                inputs_next = inputs_from_previous_layer + memory_effect

                next_activations = np.apply_along_axis(sigmoid, 0, inputs_next)
                # saving activation of output layer for next prop_forward
                self._previous_output = next_activations

                node_activations.append(next_activations)
                break

            inputs_next = np.matmul(node_activations[layer_number], weights)
            next_activations_without_bias = np.apply_along_axis(sigmoid, 0, inputs_next)
            next_activations = np.append(1, next_activations_without_bias)
            node_activations.append(next_activations)
            
        return node_activations


# testing if code works
# network = Ann(structure=(2,4,2))
# coords = network.prop_forward(input=np.array([1, 1]))[-1]
# coords2 = network.prop_forward(input=coords)[-1]
# coords3 = network.prop_forward(input=coords2)[-1]
# coords4 = network.prop_forward(input=coords3)[-1]
