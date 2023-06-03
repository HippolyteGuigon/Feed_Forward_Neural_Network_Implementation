import torch

def categorical_cross_entropy(y_pred: torch.tensor, y_true:torch.tensor):
    """
    The goal of this function
    is to compute the loss of
    a prediction regarding ground 
    truth witth respct to categorical
    cross-entropy function
    
    Arguments:
        -y_pred: torch.tensor: The prediction
        made by the model
        -y_true: torch.tensor: The true values
    Returns:
        -loss: torch.tensor(float): The computed
        loss     
    """

    loss=-(y_true.float().T@torch.log2(y_pred))
    return loss