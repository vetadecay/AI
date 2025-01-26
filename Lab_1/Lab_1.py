import numpy as np
import matplotlib.pyplot as plt

def plot_membership(x, y, title):
    plt.figure()
    plt.plot(x, y, label=title)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Membership Degree')
    plt.grid()
    plt.legend()
    plt.show()

x = np.linspace(0, 10, 100)

def trimf(x, params):
    a, b, c = params
    return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))

def trapmf(x, params):
    a, b, c, d = params
    return np.maximum(0, np.minimum(np.minimum((x - a) / (b - a), 1), (d - x) / (d - c)))

def gaussmf(x, mean, sigma):
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

def gbellmf(x, a, b, c):
    return 1 / (1 + (np.abs((x - c) / a) ** (2 * b)))

def sigmf(x, c, a):
    return 1 / (1 + np.exp(-a * (x - c)))

triangular = trimf(x, [3, 6, 9])
trapezoidal = trapmf(x, [3, 5, 7, 9])
plot_membership(x, triangular, "Triangular MF")
plot_membership(x, trapezoidal, "Trapezoidal MF")

simple_gaussian = gaussmf(x, 6, 2)
two_sided_gaussian = gaussmf(x, 4, 1.5) + gaussmf(x, 8, 1.5)
plot_membership(x, simple_gaussian, "Gaussian MF")
plot_membership(x, two_sided_gaussian, "Gaussian 2MF")

general_bell = gbellmf(x, 3, 5, 6)
plot_membership(x, general_bell, "Generalized Bell MF")

sigmoid_left = sigmf(x, 5, -1)
sigmoid_right = sigmf(x, 5, 1)
prod_sigmoid = sigmoid_left * sigmoid_right
sum_sigmoid = sigmf(x, 3, 1) + sigmf(x, 7, -1)
plot_membership(x, sigmoid_left, "L-Sigmoid MF")
plot_membership(x, sigmoid_right, "R-Sigmoid MF")
plot_membership(x, prod_sigmoid, "Prod of two Sigmoid MF")
plot_membership(x, sum_sigmoid, "Sum of two Sigmoid MF")

z_function = 1 - sigmf(x, 5, 1)
s_function = sigmf(x, 5, 1)
pi_function = z_function * s_function
plot_membership(x, z_function, "Z-Function MF")
plot_membership(x, s_function, "S-Function MF")
plot_membership(x, pi_function, "Pi-Function MF")

set_a = trimf(x, [3, 6, 9])
set_b = trapmf(x, [4, 6, 8, 9])
intersection = np.minimum(set_a, set_b)
union = np.maximum(set_a, set_b)
plot_membership(x, intersection, "Min")
plot_membership(x, union, "Max")

conjunctive = set_a * set_b
disjunctive = set_a + set_b - (set_a * set_b)
plot_membership(x, conjunctive, "Conjunctive")
plot_membership(x, disjunctive, "Disjunctive")

complementary = 1 - set_a
plot_membership(x, complementary, "Complementary")
