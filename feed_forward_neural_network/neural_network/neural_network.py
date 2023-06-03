import torch
from feed_forward_neural_network.activation.activation import softmax
from feed_forward_neural_network.loss.loss import categorical_cross_entropy


class neural_network:
    """
    The goal of this class
    is the full implementation
    of the neural_network pipeline

    Arguments:
        -input_data: torch.tensor:
        The input data tensor
        -loss: str: The loss function
        that will be appplied at the
        end of the network
    Returns:
        -None
    """

    def __init__(
        self, input_data: torch.tensor, loss: str = "categorical_cross_entropy"
    ):
        self.input_data = input_data
        
    def forward(self, layer_1, layer_2) -> None:
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
            layer_1_output = self.input_data
            for neuron in layer_2.layer_neurons:
                neuron.compute_output_value(layer_1_output)
        else:
            layer_1.get_all_outputs()
            for neuron in layer_2.layer_neurons:
                neuron.compute_output_value(layer_1.all_outputs)

    def output(self, layer) -> torch.tensor:
        """
        The goal of this function
        is to get the prediction
        of the neural network once
        its final layer is reached

        Arguments:
            -layer: The last layer of
            the neuron network
        Returns:
            -output_value: torch.tensor:
            The predicion of the network
        """

        assert (
            layer.last_layer
        ), "Output can only be computed\
            on the last layer of the network !"

        layer.get_all_outputs()
        final_scores = layer.all_outputs.apply_(lambda x: softmax(torch.tensor(x)))
        
        return torch.argmax(final_scores)

    def loss_compute(self, layer, target: torch.tensor) -> torch.tensor:
        """
        The goal of this function
        is to compute the loss made
        by the moodel afer having
        produced its predictions

        Arguments:
            -layer: The final layer
            of the network
            -target: torch.tensor: The
            target tensor

        Returns:
            -loss: torch.tensor(float):
            The loss made by the model
            after its prediction
        """

        assert (
            layer.last_layer
        ), "Loss can only be computed\
            on the last layer of the network !"

        layer.get_all_outputs()
        final_scores = softmax(layer.all_outputs)
        loss = categorical_cross_entropy(final_scores, target)

        return loss
