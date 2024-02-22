import sklearn
from sklearn.datasets import fetch_openml
import numpy.random
import numpy as np
import matplotlib.pyplot as plt

"""Nearest Neighbor Q2"""


# Q2-a
# The KNN algorithm
def KNN(train_images, labels_vector, query_image, k):
    d_list = []
    for train_image, train_label in zip(train_images, labels_vector):
        d_list.append((np.linalg.norm(train_image - query_image, ord=2), train_label))  # Calculate the distance
    sorted_list = sorted(d_list, key=lambda x: x[0])[:k]  # Sort the list and save the first K elements
    labels = [int(val[1]) for val in sorted_list]  # Save the labels
    counter = [0 for i in range(0, 10)]
    for (value) in labels:
        counter[value] = counter[value] + 1  # Count the number of each label

    return np.argmax(counter)  # Return the most common label in the first K neighbors


def main():
    # Q2-b
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    idx = numpy.random.RandomState(0).choice(70000, 11000)
    train = data[idx[:10000], :].astype(int)
    train_labels = labels[idx[:10000]]
    test = data[idx[10000:], :].astype(int)
    test_labels = labels[idx[10000:]]
    count = 0
    for test_image, test_label in zip(test, test_labels):
        if KNN(train[:1000], train_labels[:1000], test_image, 10) == int(test_label):
            count += 1
    print("The accuracy of the prediction is: " + str((count / len(test_labels)) * 100) + " percents")

    # Q2-c
    counts = []
    for k in range(1, 101):
        count = 0
        for test_image, test_label in zip(test, test_labels):
            if KNN(train[:1000], train_labels[:1000], test_image, k) == int(test_label):
                count += 1
        counts.append((count / len(test_labels)) * 100)
    plt.plot(list(range(100)), counts)
    plt.title("Prediction accuracy as a function of K")
    plt.xlabel("K")
    plt.ylabel("Prediction accuracy")
    plt.show()

    # Q2- d
    counts = []
    for n in range(100, 5001, 100):
        count = 0
        for test_image, test_label in zip(test, test_labels):
            if KNN(train[:n], train_labels[:n], test_image, 1) == int(test_label):
                count += 1
        # plt.plot(n, (count / len(test_labels)) * 100)
        counts.append((count / len(test_labels)) * 100)
    plt.plot(list(range(100, 5001, 100)), counts)
    plt.title("Prediction accuracy as a function of n")
    plt.xlabel("n")
    plt.ylabel("Prediction accuracy")
    plt.show()


if __name__ == '__main__':
    main()
