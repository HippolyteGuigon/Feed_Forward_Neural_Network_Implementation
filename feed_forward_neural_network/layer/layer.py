import torch
from feed_forward_neural_network.neuron.neuron import neuron

class layer:
    """
    The goal of this class
    is the implementation of
    linear layer
    
    Arguments:
        -input_size: int: The size
        of the input for each
        neuron of the layer
        -hidden_size: int: The number
        of neurons in the layer
    Returns:
        -None
    """

    def __init__(self, input_size: int, hidden_size: int)->None:
        self.input_size=input_size
        self.hidden_size=hidden_size

        