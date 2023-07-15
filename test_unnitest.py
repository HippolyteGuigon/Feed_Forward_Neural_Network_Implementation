import unittest
import torch
import torchvision
import subprocess
import numpy as np
from feed_forward_neural_network.neural_network.neural_network import neural_network
from feed_forward_neural_network.test.test import get_label
from feed_forward_neural_network.layer.layer import layer


class Test(unittest.TestCase):
    """
    The goal of this class is to implement unnitest
    and check everything commited makes sense
    """

    def test__forward_batch(self) -> None:
        """
        The goal of this function
        is to check the availability
        of the network to deal with
        multiple size batches

        Arguments:
            -None
        Returns:
            -None
        """

        batch_size = 64
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                "data/",
                train=True,
                download=True,
                transform=torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor()]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        data_iterator = iter(train_loader)
        batch = next(data_iterator)

        targets = batch[-1]
        data = torch.stack([torch.flatten(x) for x in batch[0].squeeze()]).T
        targets = torch.tensor([get_label(x) for x in targets])
        network = neural_network(input_data=data, targets=targets)
        layer_1 = layer(batch_size, 784, first_layer=True)
        layer_2 = layer(784, 16, first_layer=False)
        layer_3 = layer(16, 16, first_layer=False)
        layer_4 = layer(16, 10, last_layer=True)

        network.forward(layer_1, layer_2)
        network.forward(layer_2, layer_3)
        network.forward(layer_3, layer_4)
        prob, pred = network.output(layer_4)

        self.assertTrue(
            np.all(np.array([np.round(torch.sum(a).item(), 5) for a in prob]) == 1)
        )
        self.assertEquals(pred.size()[0], batch_size)

    def test_loss_computation(self) -> None:
        batch_size = 64
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                "data/",
                train=True,
                download=True,
                transform=torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor()]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        data_iterator = iter(train_loader)
        batch = next(data_iterator)

        targets = batch[-1]
        data = torch.stack([torch.flatten(x) for x in batch[0].squeeze()]).T
        targets = torch.tensor([get_label(x) for x in targets])
        network = neural_network(input_data=data, targets=targets)
        layer_1 = layer(batch_size, 784, first_layer=True)
        layer_2 = layer(784, 16, first_layer=False)
        layer_3 = layer(16, 16, first_layer=False)
        layer_4 = layer(16, 10, last_layer=True)

        network.forward(layer_1, layer_2)
        network.forward(layer_2, layer_3)
        network.forward(layer_3, layer_4)
        prob, pred = network.output(layer_4)
        loss = network.loss_compute(layer_4, targets)

        self.assertGreaterEqual(loss, 0)

    def test_full_fit_function(self)->None:
        batch_size = 64
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                "data/",
                train=True,
                download=True,
                transform=torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor()]
                ),
            ),
            batch_size=10000,
            shuffle=True,
        )

        data_iterator = iter(train_loader)
        batch = next(data_iterator)

        targets = batch[-1]
        data = torch.stack([torch.flatten(x) for x in batch[0].squeeze()]).T
        targets = torch.tensor([get_label(x) for x in targets])
        network = neural_network(
            input_data=data, epochs=2, targets=targets, batch_size=batch_size
        )

        layer_1 = layer(batch_size, 784, first_layer=True)
        layer_2 = layer(784, 16, first_layer=False)
        layer_3 = layer(16, 16, first_layer=False)
        layer_4 = layer(16, 10, last_layer=True)

        network.fit(layer_list=[layer_1, layer_2, layer_3, layer_4])

    def test_predict(self)->None:
        """
        The goal of this function is
        to verify that the neural network
        is able to deliver an accurate 
        prediction on the MNIST dataset
        
        Arguments:
            -None
            
        Returns:
            -None
        """

        batch_size = 300
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

        data_iterator = iter(train_loader)
        batch = next(data_iterator)

        targets = batch[-1]
        data = torch.stack([torch.flatten(x) for x in batch[0].squeeze()]).T
        targets = torch.tensor([get_label(x) for x in targets])
        network = neural_network(
            input_data=data, epochs=2, targets=targets, batch_size=batch_size
        )

        
        layer_1 = layer(batch_size, 784, first_layer=True)
        layer_2 = layer(784, 16, first_layer=False)
        layer_3 = layer(16, 16, first_layer=False)
        layer_4 = layer(16, 10, last_layer=True)

        network.fit(layer_list=[layer_1, layer_2, layer_3, layer_4])

        possible_pred=np.arange(10)

        to_predict= data[:,np.random.randint(60000)]
        prediction=network.predict(to_predict)

        self.assertIn(prediction.item(),possible_pred)

    def test_predict_proba(self)->None:
        """
        The goal of this function is
        to verify that the neural network
        is able to deliver an accurate 
        prediction in probability
        on the MNIST dataset
        
        Arguments:
            -None
            
        Returns:
            -None
        """

        batch_size = 300
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

        data_iterator = iter(train_loader)
        batch = next(data_iterator)

        targets = batch[-1]
        data = torch.stack([torch.flatten(x) for x in batch[0].squeeze()]).T
        targets = torch.tensor([get_label(x) for x in targets])
        network = neural_network(
            input_data=data, epochs=2, targets=targets, batch_size=batch_size
        )


        layer_1 = layer(batch_size, 784, first_layer=True)
        layer_2 = layer(784, 16, first_layer=False)
        layer_3 = layer(16, 16, first_layer=False)
        layer_4 = layer(16, 10, last_layer=True)

        network.fit(layer_list=[layer_1, layer_2, layer_3, layer_4])

        to_predict= data[:,np.random.randint(60000)]
        prediction_proba=network.predict_proba(to_predict)

        self.assertAlmostEquals(torch.sum(prediction_proba).item(),1,delta=1e-5)

    def test_full_fit_dropout_function(self)->None:
        """
        The goal of this function 
        is to check that the dropout 
        functionnality works and can be
        used during model training 
        
        Arguments:
            -None
        Returns:
            -None
        """
        
        batch_size = 64
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                "data/",
                train=True,
                download=True,
                transform=torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor()]
                ),
            ),
            batch_size=10000,
            shuffle=True,
        )

        data_iterator = iter(train_loader)
        batch = next(data_iterator)

        targets = batch[-1]
        data = torch.stack([torch.flatten(x) for x in batch[0].squeeze()]).T
        targets = torch.tensor([get_label(x) for x in targets])
        network = neural_network(
            input_data=data, epochs=2, targets=targets, batch_size=batch_size,
            dropout=True
        )


        layer_1 = layer(batch_size, 784, first_layer=True)
        layer_2 = layer(784, 16, first_layer=False)
        layer_3 = layer(16, 16, first_layer=False)
        layer_4 = layer(16, 10, last_layer=True)

        network.fit(layer_list=[layer_1, layer_2, layer_3, layer_4])

    def test_regularization(self)->None:
        """
        The goal of this function is to
        check if the neural network can be
        trained with L1/L2/Elastic-net regulrization
        
        Arguments:
            -None
        Returns:
            -None
        """

        
        batch_size = 64
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                "data/",
                train=True,
                download=True,
                transform=torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor()]
                ),
            ),
            batch_size=10000,
            shuffle=True,
        )

        data_iterator = iter(train_loader)
        batch = next(data_iterator)

        targets = batch[-1]
        data = torch.stack([torch.flatten(x) for x in batch[0].squeeze()]).T
        targets = torch.tensor([get_label(x) for x in targets])
        network = neural_network(
            input_data=data, epochs=2, targets=targets, batch_size=batch_size,
            dropout=True, regularization=True, lambda_regularization=1, mu_regularization=1
        )


        layer_1 = layer(batch_size, 784, first_layer=True)
        layer_2 = layer(784, 16, first_layer=False)
        layer_3 = layer(16, 16, first_layer=False)
        layer_4 = layer(16, 10, last_layer=True)

        network.fit(layer_list=[layer_1, layer_2, layer_3, layer_4])
    
    def test_image_build_run(self):
        # Exécuter la commande pour construire l'image Docker
        result = subprocess.run(['docker', 'build', '-t', 'ffnn:latest', '.'], capture_output=True, text=True)
        #result_run = subprocess.run(['docker', 'run', '-p', '8080:80', 'ffnn_image:latest'], capture_output=True, text=True)
        # Vérifier que la commande s'est terminée sans erreur
        self.assertEqual(result.returncode, 0)
        #self.assertEqual(result_run.returncode, 0)
        # Vérifier que la sortie de la commande contient le message attendu
        self.assertIn('Successfully built', result.stdout)
        #self.assertIn(result_run.stdout.strip(), subprocess.run(['docker', 'ps', '-q'], capture_output=True, text=True).stdout)


if __name__ == "__main__":
    unittest.main()
