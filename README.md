# Feed_Forward_Neural_Network_Implementation
The goal of this repository is the implementation of a Feed-Forward Neural Network in PyTorch from scratch

## Build Status

For the moment, the full network is ready. The next steps are:  

* To compute options that will make it possible for the user avoiding overfitting (L1/L2 regularization, dropout etc...)
* To put more user-friendly options such as more activation function
* To incorporate more optimizer options such as Stochastic Gradient Descent

Throughout its construction, if you see any improvements that could be made in the code, do not hesitate to reach out at 
Hippolyte.guigon@hec.edu. I will b delighted to get some insights !

## Code style 

The all project was coded under PEP-8 (https://peps.python.org/pep-0008/) and flake8 (https://pypi.org/project/flake8/) compliancy. Such compliance is verified during commits with pre-commits file ```.pre-commit-config.yaml```

## Installation

* This project uses a specific conda environment, to get it, run the following command: ```conda env create -f fnn_implementation_environment.yml``` 

* To install all necessary libraries, run the following code: ```pip install -r requirements.txt```

* This project has its own package that is used. To get it, run the following command: ```python install setup.py```

## Screenshot 

![alt text](https://github.com/HippolyteGuigon/Feed_Forward_Neural_Network_Implementation/blob/main/ressources/fnn.jpg)

Image of a vanilla Feed-Forward Neural Network

## How to use ? 

To have the network working on a single example (that is the MNIST dataset), you can have it running and evaluated with the following command: ```python main.py```