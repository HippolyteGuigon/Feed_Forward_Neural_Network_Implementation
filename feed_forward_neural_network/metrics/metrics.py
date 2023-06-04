import torch 

def accuracy(y_true:torch.tensor, y_pred:torch.tensor)->torch.tensor:
    """
    The goal of this function
    is the computation of the 
    accuracy metric
    
    Arguments:
        -y_true: torch.tensor: 
        The real labels 
        -y_pred: torch.tensor:
        The predicted labels
    """

    accuracy=torch.sum(y_pred==y_true)/len(y_pred)
    return accuracy