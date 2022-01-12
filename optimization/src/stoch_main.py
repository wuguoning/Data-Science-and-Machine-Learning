"""
============================================
Stochastic Gradient Method and its Variants
Author: Gordon Woo
Email:  wuguoning@gmail.com
Date:   Oct,20, 2020
Institution: China University of Petroleum at Beijing

============================================
Reference:
    1. NiranjanKumar, Implementing different variants of Gradient Descent
       Optimization Algorithm in Python using Numpy
    2. SEBASTIAN RUDER, An overview of gradient descent optimization algorithms
============================================
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors
from matplotlib import animation, rc
from IPython.display import HTML
import numpy as np
import random

#import matplotlib
#matplotlib.rcParams['animation.embed_limit'] = 2**50

from stochGradOptim import Stochastic_Gradient_And_Variants
from visu_anim_opti_with_matplot import TrajectoryAnimation

if __name__ == "__main__":

    # Test for 1D toy data
    #Data
    X = np.asarray([3.5, 0.35, 3.2, -2.0, 1.5, -0.5])
    Y = np.asarray([0.5, 0.50, 0.5,  0.5, 0.1,  0.3])
    #Algo and parameter values
    algos = ['GD','SGD','MiniBatch','Momentum','NAG','Adagrad','AdaDelta', 'RMSprop', 'Adam', 'AdaMax', 'AMSGrad']
    w_init = 2.1
    b_init = 4.0
    #learning algorithum options
    epochs = 200
    mini_batch_size = 6
    gamma = 0.9
    eta = 5

    paths = [] # for animation

    for i, algo in enumerate(algos):
        ind = 'obj{}'.format(i+1)
        ind = Stochastic_Gradient_And_Variants(w_init, b_init)
        ind.fit(X, Y, algo, epochs=epochs, mini_batch_size=mini_batch_size)
        paths.append(np.vstack((ind.w_h, ind.b_h)))

        fig = 'fig{}'.format(i+1)
        ax = 'ax{}'.format(i+1)
        fig, ax = plt.subplots(figsize=(16,9))
        ax.plot(ind.e_h, 'r', lw=3)
        ax.plot(ind.w_h, 'b', lw=3)
        ax.plot(ind.b_h, 'g', lw=3)
        ax.legend(('error', 'weight', 'bias'))
        ax.set_title("Variation of Parameters and loss function: {}".format(algo))
        ax.set_xlabel("Epoch")
        #plt.savefig("fig{}.jpg".format(i), dpi = 2000)
        plt.show()

    # 2D animation

    w_min, w_max = -7., 5.
    b_min, b_max = -7., 5.
    ww = np.linspace(w_min, w_max, 256)
    bb = np.linspace(b_min, b_max, 256)
    WW, BB = np.meshgrid(ww, bb)
    err = ind.error(X, Y, WW, BB)
    fig, ax = plt.subplots(figsize=(16,9))
    cntr1 = ax.contourf(WW, BB, err,levels=20, cmap=plt.cm.jet)
    fig.colorbar(cntr1, ax=ax)
    ax.set_xlabel('w')
    ax.set_xlim(w_min, w_max)
    ax.set_ylabel('b')
    ax.set_ylim(b_min, b_max)
    ax.dist=12

    anim = TrajectoryAnimation(*paths, labels=algos, ax=ax)

    ax.legend(loc='upper left')
    HTML(anim.to_jshtml())
    #anim.save('../../data/stoch_anim.gif', writer='imagemagick', fps=5)
    plt.show()
