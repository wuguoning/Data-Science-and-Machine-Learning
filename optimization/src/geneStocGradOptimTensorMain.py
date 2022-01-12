"""
===========================================
Generalized Stochastic Gradient Method and its Variants
Implementation with Tensorflow
Author: Gordon Woo
Email:  wuguoning@gmail.com
Date:   Oct,28, 2020
China University of Petroleum at Beijing

===========================================
Reference:
    1. NiranjanKumar, Implementing different variants of Gradient
       Descent Optimization Algorithm in Python using Numpy
    2. SEBASTIAN RUDER, An overview of gradient descent
       optimization algorithms
    3. https://stackoverflow.com/questions/55552715/tensorflow-2-0-minimize-a-simple-function

===========================================
"""

import matplotlib.pyplot as plt
import autograd.numpy as np
import math

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import animation, rc
from IPython.display import HTML

from autograd import elementwise_grad, value_and_grad
from scipy.optimize import minimize
from collections import defaultdict
from itertools import zip_longest
from functools import partial

from visu_anim_opti_with_matplot import TrajectoryAnimation
from geneStocGradOptimTensor import General_Stoch_Grad_Tensor

if __name__ == '__main__':

    ##===========================
    ## Test functions:

    ## Test Beale's function
    def Bealefun(x,y):
        try:
            return ((1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2)
        except OverflowError:
            return math.exp(100)

    xmin, xmax, xstep = -4.5, 4.5, .2
    ymin, ymax, ystep = -4.5, 4.5, .2

    w_init = 4.0
    b_init = 1.0

    ## The Global minimum
    minima = np.array([3., .5])
    minima_ = minima.reshape(-1, 1)

    ##============================
    #def Boothfun(x,y):

    #    return (x + 2.*y - 7.)**2 + (2.*x + y - 5.)**2

    #xmin, xmax, xstep = -10, 10, .2
    #ymin, ymax, ystep = -10, 10, .2

    #w_init = 8.
    #b_init = 9.

    ## The Global minimum
    #minima = np.array([1., 3.])
    #minima_ = minima.reshape(-1, 1)
    #==============================


    f = Bealefun
    x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep),
                       np.arange(ymin, ymax + ystep, ystep))
    z = f(x, y)



    # Surf the test function
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection='3d', elev=50, azim=-50)

    ax.plot_surface(x, y, z, norm=LogNorm(), rstride=1, cstride=1,
                    edgecolor='none', alpha=.8, cmap=plt.cm.jet)
    ax.plot(*minima_, f(*minima_), 'r*', markersize=10)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    plt.show()

    # Compute the vector field, with Matplotlib's quiver method
    dz_dx = elementwise_grad(f, argnum=0)(x, y)
    dz_dy = elementwise_grad(f, argnum=1)(x, y)
    fig, ax = plt.subplots(figsize=(16, 9))

    ax.contourf(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
    ax.quiver(x, y, x - dz_dx, y - dz_dy, alpha=.8)
    ax.plot(*minima_, 'r*', markersize=18)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    # Animation
    # Test for 1D toy data
    #Data
    X = 0.005*np.random.random(10)
    Y = 0.005*np.random.random(10)
    #Algo and parameter values
    algos = ['SGD','Adagrad','AdaDelta', 'RMSprop', 'Adam', 'AdaMax']
    #learning algorithum options
    epochs = 50
    mini_batch_size = 10


    paths = [] # for animation
    for i, algo in enumerate(algos):
        ind = 'obj{}'.format(i+1)
        ind = General_Stoch_Grad_Tensor(w_init, b_init)
        ind.fit("Bealefun", algo, learning_rate=0.1, tol=1E-8, max_iter=5000)
        paths.append(np.vstack((ind.w_h, ind.b_h)))


    anim = TrajectoryAnimation(*paths, labels=algos, ax=ax)
    ax.legend(loc='upper left')
    HTML(anim.to_jshtml())
    plt.show()
