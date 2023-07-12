import torch

def lasso_regularization(x: torch.tensor, lambda: float)->float:
    return  lambda*torch.norm(x, p=1)

def ridge_regularization(x: torch.tensor, lambda: float)->float:
    return  lambda*torch.norm(x, p=2)