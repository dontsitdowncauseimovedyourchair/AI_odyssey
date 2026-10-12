from neural_network import NeuralNetwork
import numpy


def main():
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3

    learning_rate = 0.3

    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

if __name__ == '__main__':
    main()