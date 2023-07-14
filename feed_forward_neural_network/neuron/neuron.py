import torch
import torch.nn as nn
import numpy as np

from feed_forward_neural_network.activation.activation import (
    sigmoid,
    ReLU,
    elu,
    swish,
    softmax,
    tanh,
    identity,
)

class neuron(nn.Module):
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

    def __init__(
        self, bias: torch.tensor, weight: torch.tensor, activation: str = "sigmoid"
    ):
        super(neuron, self).__init__()
        self.bias = bias
        self.weight = weight
        self.dropout_weight = self.weight

        activation_dict = {
            "ReLU": ReLU,
            "sigmoid": sigmoid,
            "elu": elu,
            "swish": swish,
            "softmax": softmax,
            "tanh": tanh,
            "identity": identity,
        }

        assert (
            activation in activation_dict.keys()
        ), f"The activation function\
            must be in {[option for option in activation_dict.keys()]}"

        self.activation = activation_dict[activation]

    def compute_output_value(self, input: torch.tensor, dropout: bool=False,*args, **kwargs) -> None:
        """
        The goal of this function is
        to compute the output value
        of a neuron once fed with the
        input values

        Arguments:
            -input: torch.tensor: The
            input values of the neuron
            -dropout: bool: Whether or 
            not the dropout should be taken
            into account in computing the
            output value
        Returns:
            -None
        """
        
        if not dropout:
            output_value = self.weight.T @ input
            output_value += self.bias
            self.dropout_index=[]
        else:
            self.dropout_weight = self.weight
            self.dropout_index=kwargs["dropout_index"]
            indexes=[i for i in range(self.dropout_weight.size()[0]) if i not in self.dropout_index]
            self.dropout_weight=torch.tensor(torch.index_select(self.dropout_weight,dim=0, index=torch.tensor(indexes)), requires_grad=True)
            self.dropout_weight.retain_grad()
            output_value = self.dropout_weight.T @ input
            output_value += self.bias
           
        intermediate_output = output_value
        output_value = sigmoid(output_value)
        self.output_value = output_value.squeeze()

    def dropout_param(self, dropout_proba: float)->None:
        """
        The goal of this function is, 
        when dropout is decided, to 
        decide if a given neuron is
        part of the process
        
        Arguments:
            -dropout_proba: float: The
            probability for a given neuron
            to be part of the dropout proces
        Returns:
            -None    
        """

        if np.random.random()<dropout_proba:
            self.dropout=True
        else:
            self.dropout=False

       