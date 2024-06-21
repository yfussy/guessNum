from data import get_mnist, get_input
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
import os


"""
w = weights
b = bias
i = input
h = hidden
o = output
l = label

ex w_i_h ==> weights from input layer to hidden layer
"""
weighBiasDir = os.path.dirname(os.path.realpath(__file__)) + '\\weight-bias\\'
images, labels = get_mnist()

# w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
# w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
# b_i_h = np.zeros((20, 1))
# b_h_o = np.zeros((10, 1))

w_i_h = loadtxt(weighBiasDir + 'w_i_h.csv', delimiter=',')
b_i_h = loadtxt(weighBiasDir + 'b_i_h.csv', delimiter=',')
w_h_o = loadtxt(weighBiasDir + 'w_h_o.csv', delimiter=',')
b_h_o = loadtxt(weighBiasDir + 'b_h_o.csv', delimiter=',')

b_i_h.shape += (1,)
b_h_o.shape += (1,)

learn_rate = 0.003
nr_correct = 0
epochs = 5
# 95.88%


while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(o.argmax())
    plt.show()


