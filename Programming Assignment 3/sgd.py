#################################
# Your name: May Siva
#################################
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
from scipy.special import expit
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    w_t = np.zeros(len(data[0]))
    for t in range(1, T + 1):
        i = np.random.randint(1, len(data))
        x_i = data[i]
        y_i = labels[i]
        curr_eta = eta_0 / t
        temp = np.dot(w_t, x_i)
        if y_i * temp < 1:
            w_t = (1 - curr_eta) * w_t + curr_eta * C * y_i * x_i
        else:
            w_t = (1 - curr_eta) * w_t
    return w_t


def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    global calc_norm
    w_t = np.zeros(len(data[0]))
    norm_of_w = []
    iter = []

    for t in range(1, T + 1):
        i = np.random.randint(1, len(data))
        x_i = data[i]
        y_i = labels[i]
        curr_eta = eta_0 / t
        pow_temp = np.dot(w_t, x_i)
        pow_y = np.array([])
        pow_y = np.append(pow_y, (y_i * pow_temp))
        temp_gradient = expit(pow_y)[0]
        gradient = -(y_i * x_i) * (1 - temp_gradient)
        w_t = w_t - (curr_eta * gradient)

        if calc_norm:
            norm_of_w.append(np.linalg.norm(w_t))
            iter.append(t)

    if calc_norm:
        plt.title("The norm of w as a function of the iteration")
        plt.xlabel("Iter")
        plt.ylabel("Norm of w")
        plt.plot(iter, norm_of_w)

        plt.show()

    return w_t


#################################

# Place for additional code

#################################

# Question 1a
def a1():
    etas = [10 ** val for val in range(-5, 5)]
    avg_accuracy_eta = []
    for eta in etas:
        curr_accuracy = 0
        for i in range(1, 11):
            w = SGD_hinge(train_data, train_labels, 1, eta, 1000)
            accuracy_w = calc_accuracy(validation_data, validation_labels, w)
            curr_accuracy += accuracy_w
        avg_accuracy_eta.append(curr_accuracy / 10)

    plt.title("Average accuracy as a function of eta - SGD for hinge loss")
    plt.xlabel("eta_0")
    plt.ylabel("Avg accuracy")
    plt.xscale("log")
    plt.plot(etas, avg_accuracy_eta)
    plt.show()
    return etas[np.argmax(avg_accuracy_eta)]  # ??????


# Question 1b
def b1(best_eta_0):
    C = [10 ** val for val in range(-5, 5)]
    avg_accuracy_C = []
    for c in C:
        curr_accuracy = 0
        for i in range(1, 11):
            w = SGD_hinge(train_data, train_labels, c, best_eta_0, 1000)
            accuracy_w = calc_accuracy(validation_data, validation_labels, w)
            curr_accuracy += accuracy_w
        avg_accuracy_C.append(curr_accuracy / 10)

    plt.title("Average accuracy as a function of c - SGD for hinge loss")
    plt.xlabel("C")
    plt.ylabel("Avg accuracy")
    plt.xscale("log")
    plt.plot(C, avg_accuracy_C)
    plt.show()
    return C[np.argmax(avg_accuracy_C)]  # ??????


# Question 1c
def c1(best_eta_0, best_c):
    w = SGD_hinge(train_data, train_labels, best_c, best_eta_0, 20000)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.colorbar()
    plt.show()


# Question 1d
def d1(best_eta_0, best_c):
    w = SGD_hinge(train_data, train_labels, best_c, best_eta_0, 20000)
    return calc_accuracy(test_data, test_labels, w)


def calc_accuracy(data, y_i, w):
    """This function gets data' lables y_i and a vector w, and calculates the accuracy on the data"""
    counter = 0
    for i in range(0, len(data)):
        predict_label = -1
        temp = np.dot(data[i], w)
        if temp > 0:
            predict_label = 1
        if y_i[i] == predict_label:
            counter += 1
    return counter / len(data)


# Question 2a
def a2():
    etas = [10 ** val for val in range(-5, 5)]
    avg_accuracy_eta = []
    for eta in etas:
        curr_accuracy = 0
        for i in range(1, 11):
            w = SGD_log(train_data, train_labels, eta, 1000)
            accuracy_w = calc_accuracy(validation_data, validation_labels, w)
            curr_accuracy += accuracy_w
        avg_accuracy_eta.append(curr_accuracy / 10)

    plt.title("Average accuracy as a function of eta - SGD for log loss")
    plt.xlabel("eta_0")
    plt.ylabel("Avg accuracy")
    plt.xscale("log")
    plt.plot(etas, avg_accuracy_eta)
    plt.show()
    return etas[np.argmax(avg_accuracy_eta)]


# Question 2b
def b2(eta_0):
    w = SGD_log(train_data, train_labels, eta_0, 20000)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.colorbar()
    plt.show()

    return calc_accuracy(test_data, test_labels, w)


# Question 2c
def c2(eta_0):
    global calc_norm
    calc_norm = True
    w = SGD_log(train_data, train_labels, eta_0, 20000)


train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
calc_norm = False

