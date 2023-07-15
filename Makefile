activate_environment:
	mamba init
	mamba activate feed_forward_neural_network_environment

build_docker_image:
	docker build -t ffnn_image:latest .

run_docker_image:
	docker run -p 8080:80 ffnn_image:latest

setup:
	python setup.py install