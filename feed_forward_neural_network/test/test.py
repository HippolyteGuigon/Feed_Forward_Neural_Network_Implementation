import torch
import numpy as np

def get_label(x: int)->torch.tensor:
    """
    The goal of this function
    is to generate fake labels
    for the try of the neural 
    network on the MNIST dataset
    
    Arguments:
        -x: int: The number that's
        represented on a given image
    Returns:
        -labels: torch.tensor: The tensor
        containing binary position of the 
        given number
    """
    labels=torch.zeros(size=(10,))
    labels[x]=1
    return np.array(labels)
