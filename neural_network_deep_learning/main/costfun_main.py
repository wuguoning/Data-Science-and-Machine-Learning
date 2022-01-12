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
from neuralsrc.animation import SimpleAnimation


if __name__ == "__main__":

    weight, bias = 0.6, 0.9
    eta = 0.15
    epoch = 300
    x_input, y_output = 1.0, 0.0

    obj1 = CostFunction(weight, bias)
    obj1.GD(300, x_input, y_output, eta)
    loss_h = obj1.loss_h

    # Animation
    epoch_num = np.arange(epoch)
    fig, ax = plt.subplots(figsize=(16,9))
    ax.set_xlim([0,epoch])
    ax.set_ylim([0.0,1])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Lost')
    ax.grid()

    anim = SimpleAnimation(epoch_num, loss_h, label='GD', ax = ax)
    ax.legend(loc='upper left')
    HTML=(anim.to_jshtml())
    plt.show()
