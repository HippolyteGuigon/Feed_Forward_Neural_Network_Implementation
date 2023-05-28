import torch

class neuron:
    """
    The goal of this class is
    to implement a neuron
    
    Arguments:
        -bias: torch.tensor: The
        bias of the neuron
        -weight: torch.tensor: The
        weight of the neuron
    Returns:
        -None
    """
    
    def __init__(self, bias: torch.tensor, weight:torch.tensor):
        self.bias=bias
        self.weight=weight