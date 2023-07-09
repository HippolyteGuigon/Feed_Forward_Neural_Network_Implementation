import torch
from typing import List


class gradient_descent:
    """
    The goal of his class 
    is to implement the 
    Gradient Descent algorihm
    for the optimization of the
    neural network algorithm
    
    Arguments:
        -lr: float: The learning 
        rate used during the optimization
        process
    Returns:
        -None
    """
    def __init__(self, lr: float, dropout: bool=False) -> None:
        self.lr = lr
        self.layer_list = None

    def get_new_layer_list(self, new_layer_list: List)->None:
        """
        The goal of this function
        is to get each new layer once
        is has been optimized
        
        Arguments:
            -new_layer_list: List: The
            list of layers once optimized
            to be given as attribute to the
            optimizer
        Returns:
            -None
        """
        self.layer_list = new_layer_list

    def step(self)->List:
        """
        The goal of this function
        is to optimize the current 
        weights and biases of each
        neuron in the network layers
        
        Arguments:
            -None
        Returns:
            -None
        """
        with torch.no_grad():
            for layer in self.layer_list[1:]:
                if self.dropout:
                    for neuron in layer.layer_neurons:                        
                        if neuron.dropout:
                            print("DROPOUT",neuron.dropout,neuron.dropout_weight)
                            neuron.dropout_weight -= self.lr * neuron.dropout_weight.grad
                            neuron.weight[neuron.dropout_index, :]=neuron.dropout_weight
                            neuron.dropout_weight.grad.zero_()
                            neuron.weight.grad.zero_()
                            neuron.bias.grad.zero_()
                        else:
                            neuron.weight -= self.lr * neuron.weight.grad
                            neuron.bias -= self.lr * neuron.bias.grad
                            neuron.weight.grad.zero_()
                            neuron.bias.grad.zero_()                            
                        
                else:
                    for neuron in layer.layer_neurons:
                        neuron.weight -= self.lr * neuron.weight.grad
                        neuron.bias -= self.lr * neuron.bias.grad
                        neuron.weight.grad.zero_()
                        neuron.bias.grad.zero_()

        return self.layer_list
        
class stochastic_gradient_descent:
     def __init__(self, lr)->None:
          pass