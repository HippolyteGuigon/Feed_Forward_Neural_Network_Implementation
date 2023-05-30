import torch
import numpy as np
import warnings
from feed_forward_neural_network.neuron.neuron import neuron

warnings.filterwarnings("ignore")

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
        -first_layer: bool: Whether
        or not it is the firs layer, 
        in which case all weights are 
        set to 1 and biaises to 0
    Returns:
        -None
    """

    def __init__(self, input_size: int, hidden_size: int, 
                 first_layer:str=False, *args, **kwargs)->None:
        
        self.input_size=input_size
        self.hidden_size=hidden_size
        
        if first_layer:
            self.layer_neurons=np.array([neuron(bias=x, weight=y) 
                                         for x, y in zip(torch.zeros(
                                            size=(self.hidden_size,self.input_size)),
                                            torch.ones(size=(self.hidden_size,self.input_size)))])
            self.is_first_layer=True
        else:
            self.layer_neurons=np.array([neuron(bias=x, weight=torch.randn(size=(self.input_size,self.hidden_size))) 
                                     for x in torch.randn(
                                    size=(self.hidden_size,1))])
            self.is_first_layer=False

        self.biases=np.array([neuron.bias for neuron in self.layer_neurons])
        self.weights=np.array([neuron.weight for neuron in self.layer_neurons])
        
    def get_all_outputs(self)->None:
        """
        The goal of this function
        is to get all outputs values
        of all neurons in a specific
        layers
        
        Arguments:
            -None
        Returns:
            -None
        """

        if not np.all([hasattr(neuron,"output_value") for neuron in self.layer_neurons]):
            raise ValueError("Neurons do not all have outputs values\
                             compute them first before calling this function")
        
        self.all_outputs=np.array([torch.squeeze(neuron.output_value,-1) for neuron in self.layer_neurons])
        