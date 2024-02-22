import matplotlib.pyplot as plt
"""Calculus and Probability Q1-a - The plot """

n = [5, 10, 20, 50, 100]

for j in range(5):
    h = [n[j] * ((x / 100) ** (n[j] - 1)) for x in range(1, 101)]
    h_all = [0] * 100 + h + [0] * 100
    y = [((i / 100) - 1) for i in range(1, 301)]

    # Create a line plot
    plt.plot(y, h_all, label="n=" + str(n[j]))
plt.legend()
plt.show()


plt.show()
