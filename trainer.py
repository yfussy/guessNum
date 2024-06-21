from data import get_mnist, get_input
import numpy as np
import matplotlib.pyplot as plt
from numpy import savetxt, loadtxt
import os

weighBiasDir = os.path.dirname(os.path.realpath(__file__)) + '\\weight-bias\\'

"""
w = weights
b = bias
i = input
h = hidden
o = output
l = label

ex w_i_h ==> weights from input layer to hidden layer
"""
imagesMNIST, labelsMNIST = get_mnist()


# on start up
# w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
# w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
# b_i_h = np.zeros((20, 1))
# b_h_o = np.zeros((10, 1))
# ------------------------------ 

# on training
w_i_h = loadtxt(weighBiasDir + 'w_i_h.csv', delimiter=',')
b_i_h = loadtxt(weighBiasDir + 'b_i_h.csv', delimiter=',')
w_h_o = loadtxt(weighBiasDir + 'w_h_o.csv', delimiter=',')
b_h_o = loadtxt(weighBiasDir + 'b_h_o.csv', delimiter=',')

b_i_h.shape += (1,)
b_h_o.shape += (1,)
# ------------------------------ 


learn_rate = 0.0032
nr_correct = 0
epochs = 5
# 94.19

for epochs in range(epochs):
    for img, l in zip(imagesMNIST, labelsMNIST):
        # change vectors -> matrices
        img.shape += (1,)
        l.shape += (1,)

        # forward propagation i_h
        h_pre = b_i_h + w_i_h @ img
        h = 1 / (1 + np.exp(-h_pre))  # sigmoid function -> normalized
        # forward propagation h_o
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))

        # error calculation
        e = 1 / len(o) * np.sum((o - l)**2, axis=0)  # mean-square
        nr_correct += int(np.argmax(o) == np.argmax(l))

        # backward propagation o_h
        delta_o = o - l
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o
        # backward propagation h_i
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))  # derivative of sigmoid function
        w_i_h += -learn_rate * delta_h @ np.transpose(img)
        b_i_h += -learn_rate * delta_h

    # Show accuracy for this epoch
    print(f"Acc: {round((nr_correct / imagesMNIST.shape[0]) * 100, 2)}%")
    nr_correct = 0

savetxt(weighBiasDir + 'w_i_h.csv', w_i_h, delimiter=',')
savetxt(weighBiasDir + 'b_i_h.csv', b_i_h, delimiter=',')
savetxt(weighBiasDir + 'w_h_o.csv', w_h_o, delimiter=',')
savetxt(weighBiasDir + 'b_h_o.csv', b_h_o, delimiter=',')


print("Training Done!")
