import numpy as np
from typing import List, Tuple
from scipy.special import expit as sigmoid

# ---------- Artificial Neural Network Class ---------- #

class Ann():
    def __init__(self, layers:Tuple[int], bias:Tuple[bool], genotype:List[np.ndarray]=None) -> None:
        """Initializes an ANN with number of layers equal to len(layers) (this includes both the input
        and output layer). The i'th layer has number of nodes equal to layers[i].

        Args:
            layers (Tuple[int]): i'th entry specifies how many nodes the i'th layer should have (not counting a possible bias node).
            bias (Tuple[bool]): i'th entry specifies if the i'th layer should have a bias node. Input and output layer cannot have a bias node.
            genotype (List[np.ndarray], optional): List of 2-dimensional numpy-arrays (matrices). The i'th array (matrix) specifies
            the weights going from the i'th layer to the (i+1)'th layer. Length of genotype should therefore be len(layers)-1. If genotype
            is None, weights are initialized randomly according to distribution Uni(-10, 10). Defaults to None.
        """
        self._layers = []

        if genotype is None:
            # will lead to random initialization of weights drawn from Unif(-10,10)
            genotype = [None]*(len(layers)-1)

        # use weights defined by genotype to initialize weights
        assert len(genotype)
        for index, (weights, num_nodes, has_bias) in enumerate(zip(genotype, layers[:-1], bias[:-1])):
            # only loop until the last layer is reached: the last layers has no weights and would throw an exception when
            # layer[index+1] is being called
            self._layers.append(Layer(weights=weights, num_nodes=num_nodes, num_nodes_next_layer=layers[index+1], has_bias_node=has_bias))
        self._layers.append(Layer(weights=None, num_nodes=layers[-1], num_nodes_next_layer=None, has_bias_node=bias[-1]))


    def print_layers(self):
        for layer in self._layers:
            print(layer)
        print()


    def output_activation_function(self, x:np.ndarray, min=-30, max=30):
        """Activation function that is used for the nodes in the output layer

        Args:
            x (np.ndarray): _description_
            min (int, optional): _description_. Defaults to -30.
            max (int, optional): _description_. Defaults to 30.
        """
        return np.clip(x, a_min=min, a_max=max)


    def prop_forward(self, input_sensors:np.ndarray):
        # input_sensors are the activations of the input layer given as an np.ndarray
        node_activations = input_sensors
        # print(f'Inputs to the ANN: {node_activations}\n')
        # print(self._layers[0], '\n')
        inputs_next_layer = self._layers[0].calc_inputs_next_layer(node_activations) # inputs from input layer to second layer
        # print(f'Inputs given to the second layer: {inputs_next_layer}\n')

        for index, layer in enumerate(self._layers[1:-1]):
            # input and output layers are handled outside the loop
            node_activations = layer.calc_activations(inputs_next_layer)
            # print(f'Activations of this layer: {node_activations}\n')
            # print(layer, '\n')
            inputs_next_layer = layer.calc_inputs_next_layer(node_activations)
            # print(f'Inputs given to the next layer: {inputs_next_layer}\n')
            self._layers[index]._previous_activations_next_layer = node_activations[layer._bias:] # updating next layer activations for recurrent nodes

        activations_output_layer = self._layers[-1].calc_activations(inputs_next_layer, activation_function=self.output_activation_function)
        # print(f'Activations of the output layer: {activations_output_layer}\n')
        self._layers[-2]._previous_activations_next_layer = activations_output_layer # updating next layer activations for recurrent nodes
        return activations_output_layer



class Layer():

    def __init__(self, weights:np.array, num_nodes:int, num_nodes_next_layer:int, has_bias_node:bool) -> None:
        self._bias = has_bias_node
        self._num_nodes = num_nodes # num_nodes is not counting the bias node
        self._num_next = num_nodes_next_layer

        if self._num_next is None:
            # only output layer should have num_nodes_next_layer equal to None
            self._weights = None
            self._previous_activations_next_layer = None

        else:
            self._previous_activations_next_layer = np.zeros(self._num_next) # needed for recurrent/memory nodes

            if weights is None:
                num_weights = (self._bias + self._num_nodes + self._num_next) * self._num_next # connections from bias node + normal nodes + recurrent/memory nodes to next layer
                self._weights = np.random.uniform(low=-1, high=1, size = num_weights).reshape(self._num_next, self._bias + self._num_nodes + self._num_next)

            else: self._weights = weights


    def __str__(self) -> str:
        return f'Layer(bias:{self._bias}, num_nodes:{self._num_nodes}, num_next:{self._num_next}, weights:{self._weights})'


    def calc_activations(self, inputs_prev_layer:np.ndarray, activation_function=sigmoid):
        assert (inputs_prev_layer.shape[0] == self._num_nodes), f'Array inputs_from_prev_layer is incompatible with layer size. Array has shape {inputs_prev_layer.shape}, but layer requires shape {(self._num_nodes,)}'
        
        if self._bias:
            # set activation of bias node to 1
            return np.append(np.array([1]), np.apply_along_axis(activation_function, 0, inputs_prev_layer))
        else:
            return np.apply_along_axis(activation_function, 0, inputs_prev_layer)


    def calc_inputs_next_layer(self, activations):
        assert self._num_next is not None, 'Can\'t calculate inputs for the next layer, as there is no next layer.'
        # print(f'activations shape: {activations.shape}')
        # print(f'previous_activations_next_layer shape: {self._previous_activations_next_layer.shape}')
        return np.matmul(self._weights, np.append(activations, self._previous_activations_next_layer))


# testing if code works
# network = Ann(layers=(2,4,2), bias=(False,True,False))
# coords = network.prop_forward(input_sensors=np.array([1, 1]))
# coords2 = network.prop_forward(input_sensors=coords)
# coords3 = network.prop_forward(input_sensors=coords2)
# coords4 = network.prop_forward(input_sensors=coords3)
