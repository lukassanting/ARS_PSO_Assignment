from re import L
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Animation:
    def __init__(self, opt_func, XY, title="", V=None):
        self.opt_func = opt_func
        self._XY = XY
        self._V = V
        self._title = title
        self._f, self._plot_xlow, self._plot_xhigh, self._plot_ylow, self._plot_yhigh = self.get_plot_vals()
        self._x, self._y, self._z, self._fig, self._ax, self._img, self._pcurrent, self._parrows = self.setup_plot()

    def get_plot_vals(self):
        if self.opt_func == "neg_rosenbrock":
            return self.neg_rosenbrock, -2, 2, -1, 3
        if self.opt_func == "rosenbrock":
            return self.rosenbrock, -2, 2, -1, 3
        else:    # self.opt_func === rastrigin
            return self.rastrigin, -5, 5, -5, 5

    def setup_plot(self):
        x, y = np.array(np.meshgrid(np.linspace(self._plot_xlow, self._plot_xhigh, 1000), np.linspace(self._plot_ylow, self._plot_yhigh, 1000)))
        z = self._f(x,y)
        XY = self._XY
        V = self._V
        x_min = x.ravel()[z.argmin()]
        y_min = y.ravel()[z.argmin()]

        fig, ax = plt.subplots(figsize=(8, 6))
        img = ax.imshow(z, extent=[self._plot_xlow, self._plot_xhigh, self._plot_ylow, self._plot_yhigh], origin='lower', cmap='Spectral', alpha=0.75)
        fig.colorbar(img, ax=ax)
        contours = ax.contour(x, y, z, 10, colors='black', alpha=0.4)
        ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")

        # Add particles to plot
        ax.plot([x_min], [y_min], marker='x', markersize=5, color="black")          # global minimum shown as 'x'
        p_current = ax.scatter(XY[0][0], XY[0][1], marker='o', color="blue")                # current particle locations
        if V is not None:
            p_arrows = ax.quiver(XY[0], XY[1], V[0], V[1], color='blue', width=0.005)     # vectors of particles
        else: p_arrows = None

        ax.set_xlim([self._plot_xlow, self._plot_xhigh])
        ax.set_ylim([self._plot_ylow, self._plot_yhigh])
        return x, y, z, fig, ax, img, p_current, p_arrows

    def animate(self):
        anim = FuncAnimation(self._fig, self.f_animate, frames=range(len(self._XY)), interval=150, blit=False, repeat=True)
        anim.save(f"ANIMATION_{self._title}.gif", dpi=120, writer="ffmpeg")
        print("SAVE SUCCESSFUL")

    def f_animate(self, i):
        if (i <= 100 ):
            title = 'Iteration {:02d}'.format(i)
            self._ax.set_title(title)
            XY = self._XY
            V = self._V
            # print(f"{i} = {XY[i]}")

            self._pcurrent.set_offsets(XY[i].T)
            if V is not None:
                self._parrows.set_offsets(XY[i].T)
                self._parrows.set_UVC(V[0], V[1])
        return self._ax, self._pcurrent#, self._parrows

    def neg_rosenbrock(self, x, y, a=0, b=150):
        return (-1) * ((a - x) ** 2) + b * ((y - (x ** 2)) ** 2)

    def rosenbrock(self, x, y, a=0, b=150):
        return ((a - x) ** 2) + b * ((y - (x ** 2)) ** 2)

    def rastrigin(self, x, y):
        return 20 + x ** 2 - 10 * np.cos(2 * np.pi * x) + y ** 2 - 10 * np.cos(2 * np.pi * y)


