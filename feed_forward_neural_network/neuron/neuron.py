import torch
from feed_forward_neural_network.activation.activation import sigmoid

class neuron:
    """
    The goal of this class is
    to implement a neuron
    
    Arguments:
        -bias: torch.tensor: The
        bias of the neuron
        -weight: torch.tensor: The
        weight of the neuron
        -activation: str: The 
        activation function of the
        network
    Returns:
        -None
    """
    
    def __init__(self, bias: torch.tensor, weight:torch.tensor, activation: str ="sigmoid"):
        self.bias=bias
        self.weight=weight
        
        if activation=="sigmoid":
            self.activation=staticmethod(sigmoid).__func__
        
    def compute_output_value(self,input:torch.tensor)->None:
        """
        The goal of this function is
        to compute the output value
        of a neuron once fed with the
        input values
        
        Arguments:
            -input: torch.tensor: The
            input values of the neuron
        Returns:
            -None
        """
        
        output_value=torch.transpose(self.weight,0,-1)@input
        output_value+=self.bias
        
        output_value=self.activation(output_value)
        self.output_value=output_value