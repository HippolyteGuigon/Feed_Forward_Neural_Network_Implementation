import torch
import logging
import numpy as np
from typing import List
from feed_forward_neural_network.activation.activation import softmax
from feed_forward_neural_network.loss.loss import categorical_cross_entropy
from feed_forward_neural_network.metrics.metrics import accuracy
from feed_forward_neural_network.logs.logs import main

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

main()


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
        -epochs: int: The number of
        epochs, the number of time
        training data will go throught
        the network
    
    Returns:
        -None
    """

    def __init__(
        self,
        input_data: torch.tensor,
        targets: torch.tensor,
        loss: str = "categorical_cross_entropy",
        epochs: int = 10,
        batch_size: int = 64,
    ):
        self.input_data = input_data
        self.targets = targets
        self.epochs = epochs
        self.batch_size = batch_size

    def forward(self, layer_1, layer_2, last_index: int = 0) -> None:
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
            -last_index: int: The last
            index from which data will be
            taken to get into batch
        Returns:
            -None
        """

        if layer_1.is_first_layer:
            layer_1_output = self.input_data[
                :, last_index : last_index + self.batch_size
            ]
            for neuron in layer_2.layer_neurons:
                neuron.compute_output_value(layer_1_output)
        else:
            layer_1.get_all_outputs()
            for neuron in layer_2.layer_neurons:
                neuron.compute_output_value(layer_1.all_outputs)
            layer_2.get_all_outputs()

    def fit(self, layer_list: List) -> None:
        """
        The goal of this function
        is to launch the overall
        training process with
        backpropagation

        Arguments:
            -layer_list: List: The
            list of layers composing
            the neural network
        Returns:
            -None
        """

        assert layer_list[
            0
        ].is_first_layer, "The first layer of the list\
            must be indicated as such in its arguments"

        assert layer_list[
            -1
        ].last_layer, "The last layer of the list\
            must be indicated as such in its arguments"

        for epoch in range(self.epochs):
            last_index = 0
            while last_index < self.input_data.size()[1]:
                for layer_index in range(0, len(layer_list)-1):
                    self.forward(
                        layer_list[layer_index],
                        layer_list[layer_index + 1],
                        last_index=last_index,
                    )

                    #for n, lay in enumerate(layer_list[1:]):
                    #    print(n)
                    #    lay.get_all_outputs()
                    #x, y = self.output(layer_list[-1])
                    #print(f"Loss: {self.loss_compute(layer_list[-1],self.targets[last_index:last_index+self.batch_size]):.2f}")
                last_index += self.batch_size

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
        layer.all_outputs = layer.all_outputs.T
        final_scores = torch.tensor([np.array(softmax(x)) for x in layer.all_outputs])
        final_results = torch.tensor([torch.argmax(x) for x in final_scores])

        return final_scores, final_results

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
        final_scores, final_results = self.output(layer)
        loss_values = torch.stack(
            [
                categorical_cross_entropy(y_true, y_pred)
                for y_true, y_pred in zip(final_scores, target)
            ]
        )
        loss = torch.mean(loss_values)
        logging.info(f"Loss computed: {loss:.2f}")

        return loss

    def get_metric(self, layer, metric: str = "accuracy"):
        """
        The goal of this function
        is to get the accuracy of
        the network

        Arguments:
            -metric: str: The metric
            that is wanted
            -layer: The last layer
            of the network
        Returns:
            -result: float: The metric
            computed
        """

        assert (
            layer.last_layer
        ), "Metric can only be computed\
            on the last layer of the network !"

        y_true = torch.tensor([torch.argmax(x) for x in self.targets])
        _, y_pred = self.output(layer)

        if metric == "accuracy":
            logging.info(f"Accuracy computed: {accuracy(y_true,y_pred):.2f}")
            return accuracy(y_true, y_pred)
