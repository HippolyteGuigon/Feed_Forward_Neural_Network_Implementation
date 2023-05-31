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

    def forward(self, layer_1, layer_2)->None:
        """
        The goal of this function
        is to compute the values
        from one layer to the other

        Arguments:
            -layer_1: The first layer
            that will provide input
            values fed to the next layer
            -layer_2: The second layer
            that will compute output
            values 
        Returns:
            -None
        """
        
        if layer_1.is_first_layer:
            layer_1_output=self.input_data
            for neuron in layer_2.layer_neurons:
                neuron.compute_output_value(layer_1_output)
        else:
            layer_1.get_all_outputs()
            for neuron in layer_2.layer_neurons:
                neuron.compute_output_value(layer_1.all_outputs)
        
        
    def output(self, layer):
        """
        The goal of this function
        is to get the prediction 
        of the neural network once
        its final layer is reached
        
        Arguments:
            -layer: The last layer of
            the neuron network
        Returns:
            -output_value: torch.tensor(int):
            The predicion of the network
        """