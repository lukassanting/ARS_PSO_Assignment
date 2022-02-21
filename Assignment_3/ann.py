import numpy as np
from typing import List, Tuple
from scipy.special import expit as sigmoid

# ---------- Artificial Neural Network Class ---------- #

class Ann():
    def __init__(self, layers:Tuple[int], bias:Tuple[bool], genotype:List[np.ndarray]=None) -> None:
        # "layers" should be a Tuple of integers where the first entry corrensponds to how many nodes (not including the bias) the
        # first layer should have, the second element says how many nodes the second layer (not including the bias) has and so on.
        # "bias" should be a Tuple of booleans where the first element says if the first layer has a bias node, the second element says
        # if the second layer has a bias node and so on. The first layer (input layer) and the last layer (output layer) should not have a bias nodes.
        # "genotype" should be a list of 2-dimensional numpy-arrays (matrix). The first element of the list, i.e. the first matrix should
        # specify the weights of the connections between the first and the second layer of the ANN. Therefore, the list should have a length of
        # n-1, where n is the number of layers (including input and output layers) that the ANN has
        self._layers = []

        if genotype is None:
            # randomly initialize weights
            # primarily for debugging purposes
            for index, (num_nodes, has_bias) in enumerate(zip(layers[:-1], bias[:-1])):
                # only loop until the last layer is reached: the last layers has no weights and would throw an exception when
                # layer[index+1] is being called
                self._layers.append(Layer(weights=None, num_nodes=num_nodes, num_nodes_next_layer=layers[index+1], has_bias_node=has_bias))
            self._layers.append(Layer(weights=None, num_nodes=layers[-1], num_nodes_next_layer=0, has_bias_node=bias[-1]))

        else:
            # use weights defined by genotype to initialize weights
            for index, (weights, num_nodes, has_bias) in enumerate(zip(genotype, layers[:-1], bias[:-1])):
                # only loop until the last layer is reached: the last layers has no weights and would throw an exception when
                # layer[index+1] is being called
                self._layers.append(Layer(weights=weights, num_nodes=num_nodes, num_nodes_next_layer=layers[index+1], has_bias_node=has_bias))
            self._layers.append(Layer(weights=None, num_nodes=layers[-1],  num_nodes_next_layer=None, has_bias_node=bias[-1]))

    def prop_forward(self, input_sensors:np.ndarray):
        # input_sensors are the activations of the input layer given as an np.ndarray
        node_activations = input_sensors
        inputs_next_layer = self._layers[0].calc_inputs_next_layer(node_activations) # inputs from input layer to second layer

        for index, layer in enumerate(self._layers[1:-1]):
            # input and output layers are handled outside the loop
            node_activations = layer.calc_activations(inputs_next_layer)
            inputs_next_layer = layer.calc_inputs_next_layer(node_activations)
            self._layers[index]._previous_activations_next_layer = node_activations[layer._bias:] # updating next layer activations for recurrent nodes

        activations_output_layer = self._layers[-1].calc_activations(inputs_next_layer)
        self._layers[-2]._previous_activations_next_layer = activations_output_layer # updating next layer activations for recurrent nodes
        return activations_output_layer



class Layer():
    def __init__(self, weights:np.array, num_nodes:int, num_nodes_next_layer:int, has_bias_node:bool) -> None:
        self._bias = has_bias_node
        self._num_nodes = num_nodes # num_nodes is not counting the bias node
        self._num_next = num_nodes_next_layer
        self._previous_activations_next_layer = np.zeros(self._num_next) # needed for recurrent/memory nodes

        if weights is not None:
            self._weights = weights

        else:
            if self._num_next is None:
                # only the last layer should have num_nodes_next_layer=None
                self._weights = None

            else:
                num_weights = (self._bias + self._num_nodes + self._num_next) * self._num_next # connections from bias node + normal nodes + recurrent/memory nodes to next layer
                self._weights = np.random.uniform(low=-10, high=10, size = num_weights).reshape(self._num_next, self._bias + self._num_nodes + self._num_next)

    def calc_activations(self, inputs_prev_layer:np.ndarray, activation_function=sigmoid):
        assert (inputs_prev_layer.shape[0] == self._num_nodes), f'Array inputs_from_prev_layer is incompatible with layer size. Array has shape {inputs_prev_layer.shape}, but layer requires shape {(self._num_nodes,)}'
        
        if self._bias:
            # set activation of bias node to 1
            return np.append(np.array([1]), np.apply_along_axis(activation_function, 0, inputs_prev_layer))
        else:
            return np.apply_along_axis(activation_function, 0, inputs_prev_layer)


    def calc_inputs_next_layer(self, activations):
        # print(self._weights)
        # print()
        # print()
        # print('Activations')
        # print(np.append(activations, self._previous_activations_next_layer))
        return np.matmul(self._weights, np.append(activations, self._previous_activations_next_layer))



# testing if code works
network = Ann(layers=(2,12,4,2), bias=(False,True,True,False))
coords = network.prop_forward(input_sensors=np.array([1, 4]))
print(type(coords))
coords2 = network.prop_forward(input_sensors=coords)
coords3 = network.prop_forward(input_sensors=coords2)
coords4 = network.prop_forward(input_sensors=coords3)



# ---------- Old Code ---------- #
# no need to look at this

# class layer():
#     def __init__(self, num_nodes:int, num_nodes_next_layer:int=None, bias:bool=False, recurrent:bool=False) -> None:
#         assert ((num_nodes is None) and (not recurrent)), 'Last layer has memory, recurrency is already included. Please specify recurrent=False for the last layer.'

#         self._bias_node = bias
#         self._num_nodes = num_nodes # num_nodes is not counting the bias node
#         self._num_next = num_nodes_next_layer
#         self._recurrent = recurrent
    
#         if self._num_next is None:
#             # only the last layer should have num_nodes_next_layer=None
#             # this is done to give the ANN "memory" of its previous final outputs
#             self._weights_to_next = None
#             self._weights_to_self = np.random.uniform(low=-10, high=10, size = self._num_nodes**2).reshape(self._num_nodes, self._num_nodes)
#         else:
#             self._num_weights = (self._num_nodes + self._bias_node) * self._num_next + (self._recurrent * self._num_nodes) # connections from nodes to next layer + recurrent connections

#         self._weights = np.random.uniform(low=-10, high=10, size = self._num_weights).reshape(self._num_next, self._num_nodes)

#         if self._recurrent or (self._num_next is None):
#             # previous activations need to be stored for recurrent layers and for ouput layer to implement memory
#             self._previous_activations = np.zeros(self._num_nodes)
#         else: self._previous_activations = None

#     @property
#     def bias_node(self):
#         return self._has_bias_node

#     @property
#     def weights(self):
#         return self._weights

#     @weights.setter
#     def weights(self, weights:np.ndarray):
#         assert (weights.shape == (self._num_weights,)), f'Weight array incompatible with layer size. Array has shape {weights.shape}, but layer requires shape {(self._num_weights,)}'
#         self._weights = weights

#     def calc_activations(self, inputs_from_prev_layer:np.ndarray):
#         assert (inputs_from_prev_layer.shape == (self._num_nodes)), 'Array inputs_from_prev_layer is incompatible with layer size. Array has shape {inputs_from_prev_layer.shape}, but layer requires shape {(self._num_nodes,)}'
        
#         if self._recurrent:
#             # if layer is a recurrent layer, own activations from the previous timestep also influence activations of next time step
#             num_connections_to_next = (self._num_nodes + self._bias_node) * self._num_next
#             inputs = inputs_from_prev_layer + (np.multiply(self.weights[num_connections_to_next:], self._previous_activations))
#             return np.apply_along_axis(sigmoid, 0, inputs)
        
#         elif self._num_next is None:
#             # only the last layer should have num_nodes_next_layer=None
#             inputs = inputs_from_prev_layer + (np.multiply(self.weights, self._previous_activations))
#             pass
        
#         else:
#             return np.apply_along_axis(sigmoid, 0, inputs_from_prev_layer)

#     def calc_inputs_next_layer(self):
        
#         pass
