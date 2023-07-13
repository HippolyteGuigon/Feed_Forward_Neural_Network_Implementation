import torch

def lasso_regularization(x: torch.tensor, lambda_coefficient: float=1)->float:
    """
    The goal of this function is
    to calculate the L1 Lasso penalization
    deduced from the computed weights
    during the neural network training
    
    Arguments:
        -x: torch.tensor: The weight tensor
        of the neural network
        -lambda_coefficient: float: The L1
        lambda regularization coefficient
    Returns:
        -penalization: float: The computed
        penalization
    """
    
    penalization=lambda_coefficient*torch.norm(x, p=1)
    return  penalization

def ridge_regularization(x: torch.tensor, mu_coefficient: float=1)->float:
    """
    The goal of this function is
    to calculate the L2 Ridge penalization
    deduced from the computed weights
    during the neural network training
    
    Arguments:
        -x: torch.tensor: The weight tensor
        of the neural network
        -mu_coefficient: float: The L2
        mu regularization coefficient
    Returns:
        -penalization: float: The computed
        penalization
    """

    penalization=mu_coefficient*torch.norm(x, p=2)
    return  mu_coefficient*torch.norm(x, p=2)

def elastic_net_regularization(x: torch.tensor, alpha: float)->float:
    """
    The goal of this function is
    to calculate the Elastic-net
    deduced from the computed weights
    during the neural network training
    
    Arguments:
        -x: torch.tensor: The weight tensor
        of the neural network
        -alpha: float: The elastic-net
        alpha regularization coefficient
    Returns:
        -penalization: float: The computed
        penalization
    """

    penalization=alpha*lasso_regularization(x) + (1-alpha)*ridge_regularization(x)
    return penalization