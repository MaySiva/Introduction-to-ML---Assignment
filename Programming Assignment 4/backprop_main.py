import backprop_data

import backprop_network
import numpy as np
import matplotlib.pyplot as plt

# training_data, test_data = backprop_data.load(train_size=10000,test_size=5000)
#
# net = backprop_network.Network([784, 40, 10])
#
# net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

# Q1b

rates = [0.001, 0.01, 0.1, 1, 10, 100]

training_acc_all = []
training_loss_all = []
test_acc_all = []

training_data, test_data = backprop_data.load(train_size=10000, test_size=5000)

for val in rates:
    network = backprop_network.Network([784, 40, 10])
    training_acc, training_loss, test_acc = network.SGD(training_data, 30, 10, val, test_data, 0)
    training_acc_all.append(training_acc)
    training_loss_all.append(training_loss)
    test_acc_all.append(test_acc)


def plot_Q1b(list_to_plot, num):
    for i, rate in enumerate(rates):
        plt.plot(range(30), list_to_plot[i])
    plt.xlabel('Epochs')
    if num == 0:
        plt.ylabel('Training Accuracy')
        plt.title('Training accuracy Vs. Epochs for each learning rate')
    if num == 1:
        plt.ylabel('Training Loss')
        plt.title('Training Loss Vs. Epochs for each learning rate')
    if num == 2:
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy Vs. Epochs for each learning rate')

    plt.legend(rates)
    plt.show()


plot_Q1b(training_acc_all, 0)
plot_Q1b(training_loss_all, 1)
plot_Q1b(test_acc_all, 2)


# Q1c
training_data, test_data = backprop_data.load(train_size=50000, test_size=10000)
net = backprop_network.Network([784, 40, 10])
net.SGD(training_data, 30, 10, 0.1, test_data, 1)


# Q1d - bonus
training_data, test_data = backprop_data.load(train_size=50000, test_size=10000)
net = backprop_network.Network([784, 800, 10])
net.SGD(training_data, 30, 10, 0.1, test_data, 1)
