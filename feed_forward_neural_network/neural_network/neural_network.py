import torch
import logging
import numpy as np
import torch.nn.functional as F
import warnings
from typing import List
from feed_forward_neural_network.activation.activation import softmax
from feed_forward_neural_network.loss.loss import categorical_cross_entropy
from feed_forward_neural_network.metrics.metrics import accuracy
from feed_forward_neural_network.logs.logs import main
from feed_forward_neural_network.optimizer.optimizer import gradient_descent
from feed_forward_neural_network.layer.layer import layer
from feed_forward_neural_network.regularization.regularization import lasso_regularization, ridge_regularization, elastic_net_regularization

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

main()

optimizer = gradient_descent


class neural_network(optimizer):
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
        -batch_size: int: The number of
        data that will go throught the network
        at the same time
        -lr: float: The learning rate that
        will be used for the training of the
        network
        -lambda: float: The coefficient used
        for L1 regularization
        -mu: float: The coefficient used for
        L2 regularization

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
        lr: float = 0.1,
        dropout: bool = False,
        *args, 
        **kwargs
    ):
        super().__init__(lr, dropout)
        self.input_data = input_data
        self.targets = targets
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout=dropout

        if "dropout_rate" in kwargs.keys():
            self.dropout_rate=kwargs["dropout_rate"]

        if self.dropout and "dropout_rate" not in kwargs.keys():
            self.dropout_rate=0.25
            warnings.warn("As no dropout rate was precised, it was\
                automatically set to 0.25, if you want to set\
                it yourself, enter dropout_rate=p in the arguments")

        assert self.lr > 0, "The learning rate must be strictly positive"

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
            if self.dropout:
                for neuron in layer_2.layer_neurons:
                    neuron.compute_output_value(layer_1.all_outputs, dropout=True,dropout_index=layer_1.dropout_index)
            else:
                for neuron in layer_2.layer_neurons:
                    neuron.compute_output_value(layer_1.all_outputs)
            layer_2.get_all_outputs()

    def dropout_allocation(self, layer)->None:
        """
        The goal of this function is
        to have dropout being respected
        in all neurons in all layers
        being part of the learning 
        process
        
        Arguments:
            -layer: The layer whose neurons
            will be converted for dropout 
        Returns:
            -None
        """

        if not layer.last_layer:
            for neuron in layer.layer_neurons:
                neuron.dropout_param(self.dropout_rate)
        else:
            for neuron in layer.layer_neurons:
                neuron.dropout_param(0)

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

        self.layer_list = layer_list
        super().get_new_layer_list(layer_list)
        print(lasso_regularization(self.layer_list))
        print(ridge_regularization(self.layer_list))
        print(elastic_net_regularization(self.layer_list,alpha=0.5))
        for epoch in range(self.epochs):
            last_index = 0
            while last_index < self.input_data.size()[1]:

                if self.dropout:
                    for layer in layer_list:
                        self.dropout_allocation(layer)

                for layer_index in range(0, len(self.layer_list) - 1):
                    self.forward(
                        self.layer_list[layer_index],
                        self.layer_list[layer_index + 1],
                        last_index=last_index,
                    )

                loss = self.loss_compute(
                    self.layer_list[-1],
                    self.targets[last_index : last_index + self.batch_size],
                )
                loss.backward(retain_graph=True)

                self.layer_list=super().step()
                
                logging.info(
                    f"Epoch: {epoch+1} Loss: {loss.item():.2f}, Accuracy: {self.get_metric(self.layer_list[-1],last_index=last_index):.3f}"
                )
                last_index += self.batch_size

    def output(self, layer, final_predict:bool=False) -> torch.tensor:
        """
        The goal of this function
        is to get the prediction
        of the neural network once
        its final layer is reached

        Arguments:
            -layer: The last layer of
            the neuron network
            -final_predict: bool: Whether
            or not this prediction is made
            after the network has been fitted
        Returns:
            -output_value: torch.tensor:
            The predicion of the network
        """

        assert (
            layer.last_layer
        ), "Output can only be computed\
            on the last layer of the network !"
        layer.get_all_outputs()

        if final_predict:
            final_scores = softmax(layer.all_outputs.T)
            final_results = torch.argmax(final_scores)
        else:
            final_scores = F.softmax(layer.all_outputs.T, dim=1)
            final_results = torch.tensor([torch.argmax(x) for x in final_scores])

        return final_scores, final_results

    def loss_compute(self, layer, target: torch.tensor) -> torch.tensor:
        """
        The goal of this function
        is to compute the loss made
        by the model afer having
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

        self.final_scores = final_scores
        loss_values = []

        for y_true, y_pred in zip(self.final_scores, target):
            loss = categorical_cross_entropy(y_pred, y_true)
            loss_values.append(loss)
        loss = torch.stack(loss_values).mean()
        return loss

    def get_metric(
        self, layer, metric: str = "accuracy", batch_computation: bool = False, **kwargs
    ):
        """
        The goal of this function
        is to get the accuracy of
        the network

        Arguments:
            -metric: str: The metric
            that is wanted
            -layer: The last layer
            of the network
            -batch_computation: bool:
            Whether or not the metric
            is computed for a single batch
        Returns:
            -result: float: The metric
            computed
        """

        assert (
            layer.last_layer
        ), "Metric can only be computed\
            on the last layer of the network !"

        if "last_index" in kwargs.keys():
            y_true = torch.tensor([torch.argmax(x) for x in self.targets])[
                kwargs["last_index"] : kwargs["last_index"] + self.batch_size
            ]
        else:
            y_true = torch.tensor([torch.argmax(x) for x in self.targets])
        _, y_pred = self.output(layer)

        if metric == "accuracy":
            return accuracy(y_true, y_pred)
        
    def predict(self, x: torch.tensor)->torch.tensor:
        """
        The goal of this function is 
        to predict the outcome of a single
        input tensor 
        
        Arguments:
            -x: torch.tensor: The data to be 
            predicted
        Returns:
            -prediction: torch.tensor: The 
            computed prediction
        """

        assert self.layer_list[0].hidden_size==torch.flatten(x).size()[0],\
        f"The input size of the network is {self.layer_list[0].hidden_size} \
        whereas the input size of the input data is {torch.flatten(x).size()[0]}"

        self.input_data=x

        for layer_index in range(0, len(self.layer_list) - 1):
                    if self.layer_list[layer_index].is_first_layer:
                        layer_1_output = self.input_data
                        for neuron in self.layer_list[layer_index+1].layer_neurons:
                            neuron.compute_output_value(layer_1_output)
                    else:
                        self.layer_list[layer_index].get_all_outputs()
                        for neuron in self.layer_list[layer_index+1].layer_neurons:
                            neuron.compute_output_value(self.layer_list[layer_index].all_outputs)
                        self.layer_list[layer_index+1].get_all_outputs()

        self.layer_list[-1].get_all_outputs()
        final_scores, final_results = self.output(self.layer_list[-1],final_predict=True)

        return final_results

    def predict_proba(self, x:torch.tensor)->torch.tensor:
        """
        The goal of this function 
        is to predict with a probability
        the output for a given input
        
        Arguments:
            -x: torch.tensor: The tensor 
            to be predicted
        Returns:
            -proba_vector: torch.tensor: The
            tensor with proba for each class
        """

        pass