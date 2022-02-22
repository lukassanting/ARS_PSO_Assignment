from re import L
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Animation:
    def __init__(self, opt_func, XY, V=None):
        self.opt_func = opt_func
        self._XY = XY
        self._V = V
        self._f, self._plot_xlow, self._plot_xhigh, self._plot_ylow, self._plot_yhigh = self.get_plot_vals()
        self._x, self._y, self._z, self._fig, self._ax, self._img, self._pcurrent, self._parrows = self.setup_plot()

    def get_plot_vals(self):
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
            p_arrows = ax.quiver(XY[0][0], XY[0][1], V[0][0], V[0][1], color='blue', width=0.005)     # vectors of particles
        else: p_arrows = None

        ax.set_xlim([self._plot_xlow, self._plot_xhigh])
        ax.set_ylim([self._plot_ylow, self._plot_yhigh])
        return x, y, z, fig, ax, img, p_current, p_arrows

    def animate(self):
        anim = FuncAnimation(self._fig, self.f_animate, frames=list(range(1, 10)), interval=150, blit=False, repeat=True)
        anim.save("ANIMATION.gif", dpi=120, writer="ffmpeg")

    def f_animate(self, i):
        title = 'Iteration {:02d}'.format(i)
        self._ax.set_title(title)
        XY = self._XY
        V = self._V

        self._pcurrent.set_offsets(XY[i])
        if V is not None:
            self._parrows.set_offsets(XY[i])
            self._parrows.set_UVC(V[i][0], V[i][1])
        return self._ax, self._pcurrent#, self._parrows

    def rosenbrock(self, x, y, a=0, b=150):
        return ((a - x) ** 2) + b * ((y - (x ** 2)) ** 2)

    def rastrigin(self, x, y):
        return 20 + x ** 2 - 10 * np.cos(2 * np.pi * x) + y ** 2 - 10 * np.cos(2 * np.pi * y)


# # TESTING CODE
# n_particles=20
# XY = np.random.rand(n_particles, 2) * 4
# XY[0] = XY[0] - 2  # Change range of particles on X-axis to be between -2 and 2
# XY[1] = XY[1] - 1  # Change range of particles on Y-axis to be between -1 and 3
# V = np.random.randn(n_particles,2) * 0.1
#
# new_anim = Animation("rastrigin", XY)
# anim = new_anim.animate()




# opt_func = "rosenbrock"  # Set to "rosenbrock" or to "rastigrin"
#
# # ---------------- FUNCTIONS -------------------
#
# def rosenbrock(x, y, a=0, b=150):
#     return ((a - x) ** 2) + b * ((y - (x ** 2)) ** 2)
#
# def rastigrin(x, y):
#     return 20 + x ** 2 - 10 * np.cos(2 * np.pi * x) + y ** 2 - 10 * np.cos(2 * np.pi * y)
#
# n_particles = 20
#
# # VALUES ARE SET DEPENDENT ON CHOSEN FUNCTION
# # Set values of variables for algorithm and plot for optimising Rosenbrock algorithm
# if opt_func == "rosenbrock":
#     f = rosenbrock
#     # Initialise location range of particles on X and Y-axes to be between 0 and 4
#     X = np.random.rand(2, n_particles) * 4
#     X[0] = X[0] - 2  # Change range of particles on X-axis to be between -2 and 2
#     X[1] = X[1] - 1  # Change range of particles on Y-axis to be between -1 and 3
#     # Set values used for defining plot size and location to fit specified function bounds
#     plot_xlow = -2
#     plot_xhigh = 2
#     plot_ylow = -1
#     plot_yhigh = 3
#
# # Set values of variables for algorithm and plot for optimising Rastigrin algorithm
# if opt_func == "rastigrin":
#     f = rastigrin
#     # Initialise location range of particles on X and Y-axes to be between 0 and 10
#     X = np.random.rand(2, n_particles) * 10
#     X = X - 5  # Change range of particles on X-axis and Y-axis to be between -5 and 5
#     # Set values used for defining plot size and location to fit specified function bounds
#     plot_xlow = plot_ylow = -5
#     plot_xhigh = plot_yhigh = 5
#
# V = np.random.randn(2, n_particles) * 0.1
#
# # change this with real function
# def update_position():
#     global X, V
#
#     V = np.array(20 * [.1, .1]).reshape(2, 20)
#     X = X + V
#
#     return X
#
# x, y = np.array(np.meshgrid(np.linspace(plot_xlow, plot_xhigh, 1000), np.linspace(plot_ylow, plot_yhigh, 1000)))
# z = f(x, y)
# x_min = x.ravel()[z.argmin()]
# y_min = y.ravel()[z.argmin()]
#
# fig, ax = plt.subplots(figsize=(8, 6))
# img = ax.imshow(z, extent=[plot_xlow, plot_xhigh, plot_ylow, plot_yhigh], origin='lower', cmap='Spectral', alpha=0.75)
# fig.colorbar(img, ax=ax)
# contours = ax.contour(x, y, z, 10, colors='black', alpha=0.4)
# ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
#
# # Add particles to plot
# ax.plot([x_min], [y_min], marker='x', markersize=5, color="black")          # global minimum shown as 'x'
# p_current = ax.scatter(X[0], X[1], marker='o', color="blue")                # current particle locations
# p_arrows = ax.quiver(X[0], X[1], V[0], V[1], color='blue', width=0.005)     # vectors of particles
# ax.set_xlim([plot_xlow, plot_xhigh])
# ax.set_ylim([plot_ylow, plot_yhigh])
#
# # Function to animate iterations
# def animate(i):
#     title = 'Iteration {:02d}'.format(i)
#     update_position()
#
#     ax.set_title(title)
#     p_current.set_offsets(X.T)
#     p_arrows.set_offsets(X.T)
#     p_arrows.set_UVC(V[0], V[1])
#     return ax, p_current, p_arrows
#
# max_iterations = 250
# anim = FuncAnimation(fig, animate, frames=list(range(1, 10)), interval=150, blit=False, repeat=True)
# anim.save("PSO_{}.gif".format(opt_func), dpi=120, writer="ffmpeg")



