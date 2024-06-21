import numpy as np
import matplotlib.pyplot as plt
from data import get_drawing
from numpy import loadtxt
import math
import drawing
import os

imgDir = os.path.dirname(os.path.realpath(__file__)) + '\\temp\\test.png'
print(imgDir)
weighBiasDir = os.path.dirname(os.path.realpath(__file__)) + '\\weight-bias\\'
img = get_drawing(imgDir)

w_i_h = loadtxt(weighBiasDir + 'w_i_h.csv', delimiter=',')
b_i_h = loadtxt(weighBiasDir + 'b_i_h.csv', delimiter=',')
w_h_o = loadtxt(weighBiasDir + 'w_h_o.csv', delimiter=',')
b_h_o = loadtxt(weighBiasDir + 'b_h_o.csv', delimiter=',')

b_i_h.shape += (1,)
b_h_o.shape += (1,)


# run
for i in range(1):
    
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    for i in o:
        print(i)


    plt.title(o.argmax())
    plt.show()

