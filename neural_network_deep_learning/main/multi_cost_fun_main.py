"""
===========================================
Cost Function Test
Author: Gordon Woo
Email:  wuguoning@gmail.com
Date:   Nov. 05, 2020
China University of Petroleum at Beijing

===========================================
"""
# import modules
import numpy as np
import matplotlib.pyplot as plt
import sys,os
from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib.animation import FuncAnimation

# system path append
sys.path.append('../')

# import local modules
from neuralsrc.perceptrons import CostFunction
from neuralsrc.animation import MultipleAnimation


if __name__ == "__main__":

    losses = []
    labels = []
    epoch = 300
    eta = 0.15
    epoch_num = np.arange(epoch)
    #---------------------------------------
    # Compute loss_h with weight and bias
    weight, bias = 0.6, 0.9
    x_input, y_output = 1.0, 0.0
    labels.append('GD w={}, b={}'.format(weight,bias))

    obj1 = CostFunction(weight, bias)
    obj1.GD(epoch, x_input, y_output, eta)
    loss_h = np.vstack((epoch_num,obj1.loss_h))
    losses.append(loss_h)

    #----------------------------------------
    # Compute loss_h with weight and bias
    weight, bias = 2.0, 2.0
    x_input, y_output = 1.0, 0.0
    labels.append('GD w={}, b={}'.format(weight,bias))

    obj2 = CostFunction(weight, bias)
    obj2.GD(epoch, x_input, y_output, eta)
    loss_h = obj2.loss_h
    loss_h = np.vstack((epoch_num,obj2.loss_h))
    losses.append(loss_h)

    #----------------------------------------
    # Compute loss_h with weight and bias using Cross-Entropy
    weight, bias = 0.6, 0.9
    x_input, y_output = 1.0, 0.0
    labels.append('CE w={}, b={}'.format(weight,bias))

    obj1 = CostFunction(weight, bias)
    obj1.CrossEntropy(epoch, x_input, y_output, eta)
    loss_h = obj1.loss_h
    loss_h = np.vstack((epoch_num,obj1.loss_h))
    losses.append(loss_h)

    #----------------------------------------
    # Compute loss_h with weight and bias using Cross-Entropy
    weight, bias = 2.0, 2.0
    x_input, y_output = 1.0, 0.0
    labels.append('CE w={}, b={}'.format(weight,bias))

    obj2 = CostFunction(weight, bias)
    obj2.CrossEntropy(epoch, x_input, y_output, eta)
    loss_h = obj2.loss_h
    loss_h = np.vstack((epoch_num,obj2.loss_h))
    losses.append(loss_h)


    # Animation
    epoch_num = np.arange(epoch)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlim([0,epoch])
    ax.set_ylim([0.0,1])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Lost')
    ax.grid()

    anim = MultipleAnimation(*losses, labels=labels, ax = ax)
    ax.legend(loc='upper left')
    HTML=(anim.to_jshtml())
    plt.show()
