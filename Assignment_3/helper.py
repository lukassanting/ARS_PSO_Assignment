import numpy as np
import struct

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

def weights_to_binary(weights: list) -> list:
    return [weightlayer_to_binary(l) for l in weights]

def binary_to_weights(binary: list) -> list:
    return [binary_to_weightlayer(l) for l in binary]
