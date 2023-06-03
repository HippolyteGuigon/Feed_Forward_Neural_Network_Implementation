import numpy as np
import torch


def elu(x: torch.tensor, alpha: float) -> float:
    """
    The goal of this function is
    to get the activation value
    of a given tensor with the
    elu function

    Arguments:
        -x: torch.tensor: The tensor
        to be activated
        -alpha: float: The strictly
        positive alpha value of the
        elu function
    Returns:
        -activation_value: torch.tensor:
        the value of the activation
    """

    assert (
        alpha > 0
    ), "The alpha value must\
        be strictly positive"

    if x <= 0:
        activation_value = alpha * (torch.exp(x) - 1)
    else:
        activation_value = x
    return activation_value


def ReLU(x: torch.tensor) -> float:
    """
    The goal of this function is
    to get the activation value
    of a given tensor with the
    ReLU function

    Arguments:
        -x: torch.tensor: The tensor
        to be activated
    Returns:
        -activation_value: torch.tensor:
        the value of the activation
    """

    x = torch.cat((torch.tensor([0]), torch.tensor([x])))
    activation_value = torch.max(x)

    return activation_value


def sigmoid(x: torch.tensor) -> torch.tensor:
    """
    The goal of this function is
    to get the activation value
    of a given tensor with the
    sigmoid function

    Arguments:
        -x: torch.tensor: The tensor
        to be activated
    Returns:
        -activation_value: torch.tensor:
        the value of the activation
    """

    activation_value = 1 / (1 + torch.exp(-x))

    return activation_value


def softmax(x: torch.tensor) -> torch.tensor:
    """
    The goal of this function is
    to get the activation value
    of a given tensor with the
    softmax function

    Arguments:
        -x: torch.tensor: The tensor
        to be activated
    Returns:
        -activation_value: torch.tensor:
        the value of the activation
    """

    denominator = np.sum(np.exp(x))
    activation_value = np.array([np.exp(t) / denominator for t in x])
    return activation_value


def tanh(x: torch.tensor) -> torch.tensor:
    """
    The goal of this function is
    to get the activation value
    of a given tensor with the
    tanh function

    Arguments:
        -x: torch.tensor: The tensor
        to be activated
    Returns:
        -activation_value: torch.tensor:
        the value of the activation
    """

    activation_value = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
    return activation_value


def swish(x: torch.tensor, beta: float) -> torch.tensor:
    """
    The goal of this function is
    to get the activation value
    of a given tensor with the
    swish function

    Arguments:
        -x: torch.tensor: The tensor
        to be activated
        -beta: float: The ponderated
        argument for the swish function
    Returns:
        -activation_value: torch.tensor:
        the value of the activation
    """

    activation_value = x * torch.exp(beta * x) / (1 + torch.exp(beta * x))

    return activation_value


def identity(x: torch.tensor) -> torch.tensor:
    return x
