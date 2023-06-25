import argparse
import logging
import torch
from feed_forward_neural_network.neural_network.neural_network import neural_network

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

parser = argparse.ArgumentParser()

parser.add_argument(
    "number_epochs",
    help="The number of epochs that\
                        will be used to train the model",
    default=5,
)

parser.add_argument("batch_size", help="The batch size for each", default=200)

args = parser.parse_args()


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


if __name__ == "__main__":
    launch_mnist_pepeline()
