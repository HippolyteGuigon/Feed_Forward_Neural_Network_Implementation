import torch
import torch.nn as nn

from feed_forward_neural_network.activation.activation import (
    ReLUFunction,
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

    def compute_output_value(self, input: torch.tensor) -> None:
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

        output_value = self.weight.T @ input
        output_value += self.bias
        output_value=ReLUFunction.apply(output_value)
        #output_value=torch.stack([self.activation(x) for x in output_value[0]])
        #output_value = output_value.detach().apply_(lambda x: self.activation(x))
        self.output_value = output_value.squeeze()
