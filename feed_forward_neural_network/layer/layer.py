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
        -activation: str: The activation
        function for all the neurons of
        this layer 
        -first_layer: bool: Whether
        or not it is the firs layer, 
        in which case all weights are 
        set to 1 and biaises to 0
        -last_layer: bool: Whether
        or not it is the last layer
        of the network
    Returns:
        -None
    """

    def __init__(self, input_size: int, hidden_size: int, 
                 first_layer:bool=False, last_layer:bool=False,activation: str="ReLU",*args, **kwargs)->None:
        
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.activation=activation
        self.last_layer=last_layer

        if first_layer:
            self.layer_neurons=np.array([neuron(bias=x, weight=y, activation="identity") 
                                         for x, y in zip(torch.zeros(
                                            size=(1,self.input_size)),
                                            torch.ones(size=(1,self.input_size)))])
            self.is_first_layer=True
        else:
            self.layer_neurons=np.array([neuron(bias=x, weight=torch.randn(size=(self.input_size,1))
                                                ,activation=activation) 
                                     for x in torch.randn(
                                    size=(self.hidden_size,1))])
            self.is_first_layer=False

        if self.hidden_size==1:
            self.biases=self.layer_neurons[0].bias
            self.weights=self.layer_neurons[0].weight
        else:
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
        
        self.all_outputs=torch.tensor([neuron.output_value.item() for neuron in self.layer_neurons])
        