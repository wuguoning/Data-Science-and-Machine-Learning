import matplotlib.pyplot as plt
import autograd.numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import animation
from IPython.display import HTML

from autograd import elementwise_grad, value_and_grad
from scipy.optimize import minimize
from collections import defaultdict
from itertools import zip_longest
from functools import partial

from visu_anim_opti_with_matplot import TrajectoryAnimation

# Define a recursive function
def make_minimize_cb(path=[]):

    def minimize_cb(xk):
        # note that we make a deep copy of xk
        path.append(np.copy(xk))

    return minimize_cb

if __name__ == '__main__':
    # Test Beale's function
    #f  = lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

    #xmin, xmax, xstep = -4.5, 4.5, .2
    #ymin, ymax, ystep = -4.5, 4.5, .2

    f  = lambda x, y: x**2/20. + y**2

    xmin, xmax, xstep = -10, 10, .01
    ymin, ymax, ystep = -5, 5, .01
    x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep),
                       np.arange(ymin, ymax + ystep, ystep))

    z = f(x, y)

    # The Global minimum
    minima = np.array([0., .0])
    #minima = np.array([3., .5])
    minima_ = minima.reshape(-1, 1)


    ## Surf the test function
    #fig1 = plt.figure(figsize=(8, 6))
    #ax1 = plt.axes(projection='3d', elev=50, azim=-50)

    #ax1.plot_surface(x, y, z, norm=LogNorm(), rstride=1, cstride=1,
    #                edgecolor='none', alpha=.8, cmap=plt.cm.jet)
    #ax1.plot(*minima_, f(*minima_), 'r*', markersize=10)

    #ax1.set_xlabel('$x$')
    #ax1.set_ylabel('$y$')
    #ax1.set_zlabel('$z$')

    #ax1.set_xlim((xmin, xmax))
    #ax1.set_ylim((ymin, ymax))

    #plt.show()

    # Compute the vector field, with Matplotlib's quiver method
    dz_dx = elementwise_grad(f, argnum=0)(x, y)
    dz_dy = elementwise_grad(f, argnum=1)(x, y)
    #fig2, ax2 = plt.subplots(figsize=(8, 6))

    #ax2.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
    #ax2.quiver(x, y, x - dz_dx, y - dz_dy, alpha=.8)
    #ax2.plot(*minima_, 'r*', markersize=18)

    #ax2.set_xlabel('$x$')
    #ax2.set_ylabel('$y$')

    #ax2.set_xlim((xmin, xmax))
    #ax2.set_ylim((ymin, ymax))

    #plt.show()


    # Gradient method and their animation
    x0 = np.array([5., 5])

    path_ = [x0]
    func = value_and_grad(lambda args: f(*args))
    res = minimize(func, x0=x0, method='Newton-CG',
                   jac=True, tol=1e-20, callback=make_minimize_cb(path_))

    path = np.array(path_).T

    # The Gradient methods
    methods = [
        "CG",
        "BFGS",
        "Newton-CG",
        "L-BFGS-B",
        "TNC",
        "SLSQP",
    #   "dogleg",
    #   "trust-ncg"
    ]

    minimize_ = partial(minimize, fun=func, x0=x0, jac=True,
                        bounds=[(xmin, xmax), (ymin, ymax)], tol=1e-20)

    paths_ = defaultdict(list)

    for method in methods:
        paths_[method].append(x0)

    results = {method: minimize_(method=method, callback=
                                 make_minimize_cb(paths_[method])) for method in methods}

    paths = [np.array(paths_[method]).T for method in methods]
    zpaths = [f(*path) for path in paths]

    # Animation
    fig3, ax3 = plt.subplots(figsize=(16, 9))

    #ax3.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
    #ax3.plot(*minima_, 'r*', markersize=10)
    mask = z > 100
    z[mask] = 0
    cs = ax3.contourf(x, y, z, cmap=plt.cm.jet)
    ax3.contour(cs,colors='k')
    ax3.plot([0.], 'r*', markersize=18)
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$y$')
    ax3.set_xlim((xmin, xmax))
    ax3.set_ylim((ymin, ymax))

    print(*paths)
    anim = TrajectoryAnimation(*paths, labels=methods, ax=ax3)

    ax3.legend(loc='upper left')
    HTML(anim.to_jshtml())
    plt.show()
    fig3.savefig('../data1/mnist_L1_regu_5.pdf',bbox_inches='tight')
