import argparse
import logging
import torch
import torchvision
from typing import List, Generator
from feed_forward_neural_network.neural_network.neural_network import neural_network
from feed_forward_neural_network.logs.logs import main

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

main()

parser = argparse.ArgumentParser()

parser.add_argument(
    "--number_epochs",
    help="The number of epochs that\
                        will be used to train the model",
    default=5,
    type=int,
)

parser.add_argument(
    "--batch_size", help="The batch size for each", default=200, type=int
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
    nb_epochs: int = args.number_epochs, batch_size: int = args.batch_size
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


if __name__ == "__main__":
    launch_mnist_pepeline()
