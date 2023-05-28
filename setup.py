from setuptools import setup, find_packages

setup(
    name="feed_forward_neural_network",
    version="0.1.0",
    packages=find_packages(
        include=["feed_forward_neural_network", "feed_forward_neural_network.*"]
    ),
    description="Python programm for the pytorch implementation\
        of a Feed Forward Neural Network",
    author="Hippolyte Guigon",
    author_email="Hippolyte.guigon@hec.edu",
    url="https://github.com/HippolyteGuigon/Feed_Forward_Neural_Network_Implementation",
)