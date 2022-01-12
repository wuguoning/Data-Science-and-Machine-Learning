import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Grid and test function
N = 29;
x,y = np.linspace(-1,1, N*2), np.linspace(-1,1, N)
X,Y = np.meshgrid(x,y)
F = lambda X,Y : np.sin(10*X)/(1+5*(X**2+Y**2))
Z = F(X,Y)

# 3D Surface plot
plt.figure(figsize = (5,6))
ax = plt.subplot(211, projection='3d')
# Normalise Y for calling in the cmap.
Ys = Y/Y.max()
cmap = plt.cm.viridis
ax.plot_surface(X, Y, Z, facecolors=cmap(Ys))

# 2D Plot of slice of 3D plot
# Normalise y for calling in the cmap.
ys = y/y.max()
plt.subplot(212)
plt.plot(x,Z[10,:], color=cmap(ys[10]))
plt.plot(x,Z[20,:], color=cmap(ys[20]))
plt.show()

plt.savefig('surfacePlotHighlight.png')
