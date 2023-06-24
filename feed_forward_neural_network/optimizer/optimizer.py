import torch
from typing import List


class gradient_descent:
    def __init__(self, lr: float) -> None:
        self.lr = lr
        self.layer_list = None

    def get_new_layer_list(self, new_layer_list: List):
        self.layer_list = new_layer_list

    def step(self)->List:
        with torch.no_grad():
            for layer in self.layer_list[1:]:
                    for neuron in layer.layer_neurons:
                        neuron.weight -= self.lr * neuron.weight.grad
                        neuron.bias -= self.lr * neuron.bias.grad
                        neuron.weight.grad.zero_()
                        neuron.bias.grad.zero_()

        return self.layer_list
        
class stochastic_gradient_descent:
     def __init__(self, lr)->None:
          pass