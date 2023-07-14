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
    
    assert lambda_coefficient>=0, "The lambda L1 coefficient must be positive"
    
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

    assert mu_coefficient>=0, "The L2 regularization coefficient must be positive"
    weights_list=torch.stack([weight for layer in layer_list for neuron in layer.layer_neurons for weight in neuron.weight if not layer.is_first_layer])
    penalization=mu_coefficient*torch.norm(weights_list, p=2)
    return penalization

def elastic_net_regularization(layer_list, alpha: float=0.5, **kwargs)->float:
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
        -**kwarhs: dict: 
    Returns:
        -penalization: float: The computed
        penalization
    """


    assert alpha>=0 and  alpha<=1,"The alpha coefficient for Elastic-net must be between 0 and 1"

    if alpha==0:
        penalization=ridge_regularization(layer_list)
    elif alpha==1:
        penalization=lasso_regularization(layer_list)
    elif "lambda_coefficient" and "mu_coefficient" in kwargs.keys():
        lambda_coefficient=alpha*(kwargs["lambda_coefficient"]+kwargs["mu_coefficient"])
        mu_coefficient=(1-alpha)*lambda_coefficient/alpha
        penalization=alpha*lasso_regularization(layer_list,lambda_coefficient=lambda_coefficient) + (1-alpha)*ridge_regularization(layer_list,mu_coefficient=mu_coefficient)
    elif "lambda_coefficient" in kwargs.keys():
        penalization=alpha*lasso_regularization(layer_list,lambda_coefficient=lambda_coefficient) + (1-alpha)*ridge_regularization(layer_list)
    elif "mu_coefficient" in kwargs.keys():
        penalization=alpha*lasso_regularization(layer_list) + (1-alpha)*ridge_regularization(layer_list,mu_coefficient=mu_coefficient)
    else:
        penalization=alpha*lasso_regularization(layer_list) + (1-alpha)*ridge_regularization(layer_list)
    
    return penalization