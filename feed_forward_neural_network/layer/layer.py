import torch
import numpy as np
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

    def __init__(self, input_size: int, hidden_size: int, *args, **kwargs)->None:
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.layer_neurons=np.array([neuron(bias=x, weight=y) 
                                     for x, y in zip(torch.randn(
                                    size=(self.input_size,1)),
                                    torch.randn(size=(self.input_size,1)))])
        self.biases=torch.tensor([neuron.bias for neuron in self.layer_neurons])
        self.weights=torch.tensor([neuron.weight for neuron in self.layer_neurons])
        
   