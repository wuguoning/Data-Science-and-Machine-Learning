import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from IPython.display import HTML



class SimpleAnimation(animation.FuncAnimation):
    """
    Define a animation class.
    Author:Louis Tiao
    Revised by: Gordon Woo
    Email:wuguoning@gmail.com
    """

    def __init__(self, xdata, ydata, label, fig=None, ax=None, frames=None,
                 interval=60, repeat_delay=5, blit=True, **kwargs):

        if fig is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()
        else:
            if ax is None:
                ax = fig.gca()

        self.fig = fig
        self.ax = ax
        self.xdata = xdata
        self.ydata = ydata

        if frames is None:
            frames = len(xdata)

        self.line = self.ax.plot([], [], label=label, color='red', lw=2)[0]
        self.point = self.ax.plot([], [], 'o', color='red')[0]

        super(SimpleAnimation, self).__init__(fig, self.animate, init_func=self.init_anim,
                                                  frames=frames, interval=interval, blit=blit,
                                                  repeat_delay=repeat_delay, **kwargs)

    def init_anim(self):
        self.line.set_data([], [])
        self.point.set_data([], [])

        return self.line, self.point

    def animate(self, i):
        self.line.set_data(self.xdata[:i], self.ydata[:i])
        self.point.set_data(self.xdata[i-1:i], self.ydata[i-1:i])

        return self.line, self.point
