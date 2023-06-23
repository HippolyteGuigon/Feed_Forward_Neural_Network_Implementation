from typing import List


class gradient_descent:
    def __init__(self, lr: float) -> None:
        self.lr = lr
        self.layer_list = None

    def get_new_layer_list(self, new_layer_list: List):
        self.layer_list = new_layer_list

    def optimizer(self):
        pass
