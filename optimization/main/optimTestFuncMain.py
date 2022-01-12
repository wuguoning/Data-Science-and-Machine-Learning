"""
===========================================
Optimization Test Functions
Author: Gordon Woo
Email:  wuguoning@gmail.com
Date:   2021-01-05
China University of Petroleum at Beijing

===========================================
Reference:
    1. https://www.sfu.ca/~ssurjano/ackley.html
    2. Adorio, E. P., & Diliman, U. P. MVF -
       Multivariate Test Functions Library in C
       for Unconstrained Global Optimization (2005).
       Retrieved June 2013, from
       http://http://www.geocities.ws/eadorio/mvf.pdf.
    3. Molga, M., & Smutnicki, C. Test functions for
       optimization needs (2005). Retrieved June 2013,
       from http://www.zsd.ict.pwr.wroc.pl/files/docs/functions.pdf.
===========================================
"""

#===========================================
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import cm
import matplotlib.colors
from matplotlib import animation, rc
from IPython.display import HTML

import numpy as np
import random
import sys,os

#===========================================
sys.path.append('../')
from src.optimTestFunc import OptimTestFunc

#===========================================
obj = OptimTestFunc()

#===========================================

xmin, xmax, xstep = -30, 30, 0.1
ymin, ymax, ystep = -30, 30, 0.1
x = np.arange(xmin, xmax, xstep)
y = np.arange(ymin, ymax, ystep)
[X, Y] = np.meshgrid(x,y)
Z = obj.AckleyFun([X, Y])
minima = [0., 0.]

# Surf the test function
fig = plt.figure(figsize=(16,9))
ax = plt.axes(projection='3d', elev=15, azim=-45)
Ys = Y/Y.max()
cmap = plt.cm.viridis
ax.plot_surface(X, Y, Z, facecolors=cmap(Ys))
#ax.plot_surface(X, Y, Z, norm=LogNorm(), rstride=2, cstride=2,
#                edgecolor='none', alpha=.8, cmap=plt.cm.coolwarm,
#                facecolors=cmap(Ys))
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
ax.set_xlim((-40, 40))
ax.set_ylim((-40, 40))
ax.set_zlim((0, 25))

plt.show()

#===========================================
