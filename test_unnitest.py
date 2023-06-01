import unittest
import logging
from feed_forward_neural_network.logs.logs import main


class Test(unittest.TestCase):
    """
    The goal of this class is to implement unnitest
    and check everything commited makes sense
    """

    logging.info("Unnitest session has begun")

    def test_single_forward_batch(self) -> None:
        """
        The goal of this function
        is to check the availability
        of the network to"""
        pass

    def test_loss_computation(self) -> None:
        pass


if __name__ == "__main__":
    main()
    unittest.main()
