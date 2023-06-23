import numpy as np
import torch

def safe_softmax(input):
    max_vals, _ = input.max(dim=1, keepdim=True)
    shifted_input = input - max_vals
    exp_shifted_input = torch.exp(shifted_input)
    softmax_output = exp_shifted_input / exp_shifted_input.sum(dim=1, keepdim=True)
    softmax_output = torch.where(torch.isnan(softmax_output), torch.zeros_like(softmax_output), softmax_output)
    softmax_output = torch.where(torch.isinf(softmax_output), torch.zeros_like(softmax_output), softmax_output)
    softmax_output = softmax_output / softmax_output.sum(dim=1, keepdim=True)
    return softmax_output
    
class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)  # Enregistrez les tenseurs nécessaires pour la rétropropagation
        output = torch.relu(input)  # Appliquez la fonction ReLU sur le tenseur d'entrée
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors  # Récupérez les tenseurs enregistrés pendant le forward
        grad_input = grad_output.clone()  # Clonez le gradient de sortie
        grad_input[input < 0] = 0  # Appliquez le masque de la fonction ReLU pour les gradients négatifs
        return grad_input

class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        exp_input = torch.exp(input)
        softmax_output = exp_input / exp_input.sum(dim=1, keepdim=True)
        ctx.save_for_backward(softmax_output)  # Enregistrez les tenseurs nécessaires pour la rétropropagation
        return softmax_output

    @staticmethod
    def backward(ctx, grad_output):
        softmax_output, = ctx.saved_tensors  # Récupérez les tenseurs enregistrés pendant le forward
        grad_input = softmax_output * (grad_output - (softmax_output * grad_output).sum(dim=1, keepdim=True))
        return grad_input   
    
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

    max_val = torch.max(x)
    log_softmax = x - max_val - torch.log(torch.sum(torch.exp(x - max_val)))
    return torch.exp(log_softmax)


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
