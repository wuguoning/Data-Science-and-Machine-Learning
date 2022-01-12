# coding: utf-8
"""
Optimization comparison using naive function.
Author:     齋騰康毅
Revised:    Gordon Woo
Email:      wuguoning@gmail.com
Department: China University of Petroleum at Beijing
Date:       Nov.09, 2020
"""
#---------------------------------------------------
import sys, os
# import parents directory
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from collections import OrderedDict

from neuralsrc.optimizer.stocoptimizer import *
from neuralsrc.animation import MultipleAnimation


#---------------------------------------------------
# toy function
def f(x, y):
    return x**2 / 20.0 + y**2

# Gradients of the toy function
def df(x, y):
    return x / 10.0, 2.0*y

#---------------------------------------------------
if __name__ == "__main__":

    init_pos = np.array([-7.0, 2.0])
    grads = np.array([0.,0.])

    # dictionary of objects
    optimizers = OrderedDict()
    optimizers["SGD"] = SGD1(lr=0.95)
    optimizers["Momentum"] = Momentum1(lr=0.05)
    optimizers["AdaGrad"] = AdaGrad1(lr=0.95)
    optimizers["Adam"] = Adam1(lr=0.2)
    optimizers["Nesterov"] = Nesterov1(lr=0.05)
    optimizers["RMSprop"] = RMSprop1(lr=0.1)
    optimizers["AdaMax"] = AdaMax1(lr=0.25)
    optimizers["AdaDelta"] = AdaDelta1(lr=11.)

    labels = []
    paths = []

    for key in optimizers.keys():
        grads = np.array([0.0, 0.0])
        params = init_pos
        labels.append(key)
        optimizer = optimizers[key]
        x_history = []
        y_history = []

        for i in range(50):
            x_history.append(params[0])
            y_history.append(params[1])
            grads[0],grads[1] = df(params[0],params[1])
            params = optimizer.update(params, grads)
        paths.append(np.vstack((x_history, y_history)))


    fig, ax = plt.subplots(figsize=(6,4))
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # for simple contour line
    mask = Z > 100
    Z[mask] = 0
    cs = ax.contourf(X, Y, Z, cmap=plt.cm.jet)
    ax.contour(cs,colors='k')
    ax.plot([0.], 'r*', markersize=18)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    #Animation
    anim = MultipleAnimation(*paths, labels = labels, ax = ax)
    ax.legend(loc='upper left')
    HTML = (anim.to_jshtml)
    plt.show()
