import argparse
import logging
import torch
import torchvision
import joblib
from feed_forward_neural_network.test.test import get_label
from feed_forward_neural_network.layer.layer import layer
from typing import List, Generator
from feed_forward_neural_network.neural_network.neural_network import neural_network
from feed_forward_neural_network.layer.layer import layer
from feed_forward_neural_network.logs.logs import main

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

main()

parser = argparse.ArgumentParser()

parser.add_argument(
    "--learning_rate",
    help="The learning rate that will\
         be used during the training\
        of the network",
    default=1e-2,
    type=float,
)

parser.add_argument(
    "--number_epochs",
    help="The number of epochs that\
                        will be used to train the model",
    default=10,
    type=int,
)

parser.add_argument(
    "--batch_size", help="The batch size for each", default=200, type=int
)

parser.add_argument(
    "--dropout", help="Whether or not dropout should be applied", default=False,type=bool
)

parser.add_argument(
    "--regularization", help="Whether or not regularization should be applied", default=False,type=bool
)

args = parser.parse_args()


def get_loader() -> List[Generator]:
    """
    The goal of this function
    is to get the train and test
    loader for the MNIST dataset

    Arguments:
        -None
    Returns:
        -None
    """

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "data/",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),
        ),
        batch_size=60000,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "data/",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),
        ),
        batch_size=10000,
        shuffle=True,
    )

    return train_loader, test_loader


def launch_mnist_pepeline(
    nb_epochs: int = args.number_epochs, batch_size: int = args.batch_size,lr:float=args.learning_rate
) -> str:
    """
    The goal of this function is to run
    the full mnist pipeline by training
    the neural network on the all dataset
    and return accuracte predictions
    with a score

    Arguments:
        -nb_epochs
        -batch_size

    Returns:
        -synthesis: str:
    """

    train_loader, test_loader = get_loader()
    
    logging.info("Train and test data successfully loaded !")

    data_train_iterator = iter(train_loader)
    data_test_loader=iter(test_loader)
    batch = next(data_train_iterator)

    targets_train = batch[-1]
    data_train = torch.stack([torch.flatten(x) for x in batch[0].squeeze()]).T
    data_test=[x for x in data_test_loader]
    targets_train = torch.tensor([get_label(x) for x in targets_train])

    network = neural_network(
            input_data=data_train, epochs=nb_epochs, targets=targets_train, batch_size=batch_size,lr=lr,
            dropout=args.dropout,regularization=args.regularization)
    layer_1 = layer(batch_size, 784, first_layer=True)
    layer_2 = layer(784, 16, first_layer=False)
    layer_3 = layer(16, 16, first_layer=False)
    layer_4 = layer(16, 10, last_layer=True)

    logging.info(f"Fitting of the model has begun !\
        Params are epochs: {nb_epochs}, batch_size: {batch_size},\
        learning rate: {lr}")
    
    network.fit(layer_list=[layer_1, layer_2, layer_3, layer_4])

    logging.warning("Fitting of the model has just ended ! Beginning the evaluation part...")
    predictions=torch.stack([network.predict(torch.flatten(x)) for x in data_test[0][0]])

    accuracy=(torch.sum(data_test[-1][-1]==predictions)/predictions.size()[0]).item()
    logging.info(f"The accuracy of the model on the overall test set is: {accuracy:.2f}")

if __name__ == "__main__":
    launch_mnist_pepeline()
