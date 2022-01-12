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

from neuralsrc.optimizer.optimizer import *
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

    init_pos = (-7.0, 2.0)
    params = {}
    params['x'], params['y'] = init_pos[0], init_pos[1]
    grads = {}
    grads['x'], grads['y'] = 0, 0

    optimizers = OrderedDict()
    optimizers["SGD"] = SGD(lr=0.95)
    optimizers["Momentum"] = Momentum(lr=0.05)
    optimizers["AdaGrad"] = AdaGrad(lr=0.95)
    optimizers["Adam"] = Adam(lr=0.2)
    optimizers["Nesterov"] = Nesterov(lr=0.05)
    optimizers["RMSprop"] = RMSprop(lr=0.1)
    optimizers["AdaMax"] = AdaMax(lr=0.25)
    optimizers["AdaDelta"] = AdaDelta(lr=11.)

    labels = []
    paths = []

    for key in optimizers.keys():
        labels.append(key)
        optimizer = optimizers[key]
        x_history = []
        y_history = []
        params['x'], params['y'] = init_pos[0], init_pos[1]

        for i in range(50):
            x_history.append(params['x'])
            y_history.append(params['y'])
            grads['x'], grads['y'] = df(params['x'], params['y'])
            optimizer.update(params, grads)
        paths.append(np.vstack((x_history, y_history)))


    fig, ax = plt.subplots(figsize=(16,9))
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
    #fig.savefig('../data1/native_stoch_grad_comp.pdf',bbox_inches='tight')
    f = '../data1/native_stoch_grad_comp.gif'
    writergif = animation.PillowWriter(fps=30)
    anim.save(f, writer=writergif)
