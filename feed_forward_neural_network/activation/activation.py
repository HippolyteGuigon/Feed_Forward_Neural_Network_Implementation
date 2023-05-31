import torch

def elu(x: torch.tensor)->float:
    pass

def ReLU(x: torch.tensor)->float:
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
    
    x=torch.cat((torch.tensor([0]),x))
    activation_value=torch.max(x)

    return activation_value

def sigmoid(x: torch.tensor)->torch.tensor:
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

    activation_value=1/(1+torch.exp(-x))
    
    return activation_value

def tanh(x: torch.tensor)->float:
    pass

