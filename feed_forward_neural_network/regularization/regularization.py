import torch

def lasso_regularization(layer_list, lambda_coefficient: float=1)->float:
    """
    The goal of this function is
    to calculate the L1 Lasso penalization
    deduced from the computed weights
    during the neural network training
    
    Arguments:
        -layer_list: Class: The layer_list
        from which the weights will be retrieved
        -lambda_coefficient: float: The L1
        lambda regularization coefficient
    Returns:
        -penalization: float: The computed
        penalization
    """ 
    
    weights_list=torch.stack([weight for layer in layer_list for neuron in layer.layer_neurons for weight in neuron.weight if not layer.is_first_layer])
    penalization=lambda_coefficient*torch.norm(weights_list, p=1)
    return  penalization

def ridge_regularization(layer_list, mu_coefficient: float=1)->float:
    """
    The goal of this function is
    to calculate the L2 Ridge penalization
    deduced from the computed weights
    during the neural network training
    
    Arguments:
        -layer_list: Class: The layer_list
        from which the weights will be retrieved
        -mu_coefficient: float: The L2
        mu regularization coefficient
    Returns:
        -penalization: float: The computed
        penalization
    """

    weights_list=torch.stack([weight for layer in layer_list for neuron in layer.layer_neurons for weight in neuron.weight if not layer.is_first_layer])
    penalization=mu_coefficient*torch.norm(weights_list, p=2)
    return penalization

def elastic_net_regularization(layer_list, alpha: float)->float:
    """
    The goal of this function is
    to calculate the Elastic-net
    deduced from the computed weights
    during the neural network training
    
    Arguments:
        -layer_list: Class: The layer_list
        from which the weights will be retrieved
        -alpha: float: The elastic-net
        alpha regularization coefficient
    Returns:
        -penalization: float: The computed
        penalization
    """

    weights_list=torch.stack([weight for layer in layer_list for neuron in layer.layer_neurons for weight in neuron.weight if not layer.is_first_layer])
    penalization=alpha*lasso_regularization(weights_list) + (1-alpha)*ridge_regularization(weights_list)
    return penalization