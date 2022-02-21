import numpy as np
import struct
from typing import List
from ann import *


# https://stackoverflow.com/questions/8751653/how-to-convert-a-binary-string-into-a-float-value?noredirect=1&lq=1

def binary_to_float(number: str) -> float:
    """ Convert binary string to float. """
    return struct.unpack('!f', struct.pack('!I', int(number, 2)))[0]


def float_to_binary(number: float) -> str:
    """ Convert float to binary string. """
    return bin(struct.unpack('!I', struct.pack('!f', number))[0])[2:].zfill(32)


def array_to_binary(weights: np.array) -> np.array:
    """ Convert array of floats to binary strings. """
    return np.array([float_to_binary(w) for w in weights])


def array_to_float(binary: np.array) -> np.array:
    """ Convert array of binary strings to floats. """
    return np.array([binary_to_float(b) for b in binary])


def weightlayer_to_binary(weights: np.array) -> np.array:
    """ Convert matrix of floats (single NN layer) to matrix of binary strings. """
    return np.array([array_to_binary(arr) for arr in weights])


def binary_to_weightlayer(binary: np.array) -> np.array:
    """ Convert matrix of binary strings to matrix of floats. """
    return np.array([array_to_float(arr) for arr in binary])


def weights_to_binary(weights: List) -> List:
    """ Convert list of matrices of floats to list of matrices of binary strings. """
    return [weightlayer_to_binary(l) for l in weights]


def binary_to_weights(binary: List) -> List:
    """ Convert list of matrices of binary strings to list of matrices of floats. """
    return [binary_to_weightlayer(l) for l in binary]


def ann_to_list(network: Ann) -> List:
    """ Convert instance of Ann to list of np.arrays for each layer connection. """
    return [layer.weights for layer in network.layers]


def array_to_network(weights: np.array, network_layer_sizes: Tuple[int], network_layer_biases:Tuple[bool]) -> Ann:
    """ weights:                single-dimensional (floating point number) array with all the weights for the network
        network_layer_sizes:    Tuple specifying the number of nodes in each layer
        network_layer_biases:   Tuple specifying which layers have bias nodes (only included for network creation at
                                of this function, this function assumes that only middle nodes have biases.

        Function takes an array of weight values, and a specified shape of a network, and makes a recurrent neural
        network of specified shape using the specified weights.
    """

    # Split the full 1D weight array into an individual 1D array for each individual layer
    weight_layer_sizes = get_network_layer_size(network_layer_sizes)  # number of weight connections between layers
    split_points = accumulate(weight_layer_sizes)   # e.g. [168, 68, 14] --> [168, 236]
    sub_arrays = np.split(weights, split_points)    # returns split arrays of correct sizes, allows us to reshape

    # Reshape the 1D arrays into 2D arrays and add them to a list
    weight_layers = []
    # Do first connections separately, as they have no bias connections
    first_weight_layer = np.reshape(sub_arrays[0], (network_layer_sizes[1], (network_layer_sizes[0] + network_layer_sizes[1])))
    weight_layers.append(first_weight_layer)

    # Next do each other layer, starting at the weights between layer 2 and layer 3
    for l in range(len(sub_arrays)-1):
        next_weight_layer = np.reshape(sub_arrays[l+1], (network_layer_sizes[l+2], (network_layer_sizes[l+1] + network_layer_sizes[l+2] + 1)))
        weight_layers.append(next_weight_layer)

    # Create an ANN using the new list of 2D weight arrays
    return Ann(layers=network_layer_sizes, bias=network_layer_biases, genotype=weight_layers)

def accumulate(input):
    output = [0] * (len(input)-1)
    output[0] = input[0]
    for i in range(len(input)-2):
        output[i+1] = input[i+1] + output[i]
    return output


def get_network_size(layers: Tuple[int]) -> int:
    """ Return the total number of weights in a network based on size of each layer """
    return sum(get_network_layer_size(layers))


def get_network_layer_size(layers: Tuple[int]) -> List[int]:
    """ Return a list of the number of weights in each layer of a network based on size of each layer """
    size = [layers[1] * (layers[0] + layers[1])]
    if len(layers) > 2:
        for i in range(len(layers) - 2):
            # for each layer after the first, calculate the same as above but add 1 in multiplication for the bias node
            size.append(layers[i + 2] * (layers[i + 1] + layers[i + 2] + 1))
    return size
