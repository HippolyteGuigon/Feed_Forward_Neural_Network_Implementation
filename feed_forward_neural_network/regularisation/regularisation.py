import torch

def lasso_regularization(x: torch.tensor, lambda_coefficient: float)->float:
    return  lambda_coefficient*torch.norm(x, p=1)

def ridge_regularization(x: torch.tensor, mu_coefficient: float)->float:
    return  mu_coefficient*torch.norm(x, p=2)

