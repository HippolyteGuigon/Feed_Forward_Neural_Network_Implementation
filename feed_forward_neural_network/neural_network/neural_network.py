import torch
from feed_forward_neural_network.layer.layer import layer

class neural_network:
    """
    The goal of this class
    is the full implementation
    of the neural_network pipeline
    
    Arguments:
        -input_data: torch.tensor:
        The input data tensor
    Returns:
        -None
    """

    def __init__(self, input_data: torch.tensor):
        self.input_data=input_data

    def forward(self, layer_1, layer_2)->torch.tensor:
        if layer_1.is_first_layer:
            layer_1_output=self.input_data
        else:
            layer_1_output=layer_1.get_all_outputs()
        
        for neuron in layer_2.layer_neurons:
            neuron.compute_output_value(layer_1_output)
            print(neuron.output_value)