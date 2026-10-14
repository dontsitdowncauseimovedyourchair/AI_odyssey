import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot

from neural_network import NeuralNetwork
import numpy


def main():
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    learning_rate = 0.2

    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    training_data_file = open("mnist_dataset/mnist_train.csv", "r")
    td_list = training_data_file.readlines()
    training_data_file.close()

    for record in td_list:
        entries = record.split(",")
        inputs = (numpy.asarray(entries[1:], dtype=float) / 255 * 0.99) + 0.01
        label = entries[0]
        targets = np.zeros(output_nodes) + 0.01
        targets[int(label)] = 0.99
        n.train(inputs_list=inputs, targets_list=targets)

    test_file = open("mnist_dataset/mnist_test.csv", "r")
    tf_list = test_file.readlines()
    test_file.close()

    scoreboard = []

    for record in tf_list:
        data = record.split(",")
        correct = int(data[0])
        print(correct, "correct answer")
        inputs = (numpy.asarray(data[1:], dtype=float) / 255 * 0.99) + 0.01
        outputs = n.query(inputs)
        label = numpy.argmax(outputs)
        print(label, "network's answer")
        if label == correct:
            scoreboard.append(1)
        else:
            scoreboard.append(0)

    print(scoreboard)

    sb_arr = numpy.asarray(scoreboard)
    print("Performance: " + (sb_arr.sum() / sb_arr.size * 100).__str__() + "%")

if __name__ == '__main__':
    main()