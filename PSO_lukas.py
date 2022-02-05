# based on: https://machinelearningmastery.com/a-gentle-introduction-to-particle-swarm-optimization/

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# ---------------- DEFINE FUNCTIONS -------------------
#  Rosenbrock function to optimise
def rosenbrock(x, y, a=0, b=150):
    return ((a - x) ** 2) + b * ((y - (x ** 2)) ** 2)


# Other sample function to optimise
def egg_carton(x, y):
    return (x - 3.14) ** 2 + (y - 2.72) ** 2 + np.sin(3 * x + 1.41) + np.sin(4 * y - 1.73)


# Function to do an iteration of PSO
def iterate():
    global V, X, p_best, g_best, p_best_out, g_best_out
    r1, r2 = np.random.rand(2)
    V = w * V + c1 * r1 * (p_best - X) + c2 * r2 * (g_best.reshape(-1, 1) - X)
    X = X + V
    # Update p_best and g_best
    f_out = f(X[0], X[1])
    p_best[:, (p_best_out >= f_out)] = X[:, (p_best_out >= f_out)]
    g_best = p_best[:, p_best_out.argmin()]
    g_best_out = p_best_out.min()


# Function to animate iterations
def animate(i):
    title = 'Iteration {:02d}'.format(i)
    iterate()
    ax.set_title(title)
    p_current.set_offsets(X.T)
    p_arrows.set_offsets(X.T)
    p_arrows.set_UVC(V[0], V[1])
    p_g_best.set_offsets(g_best.reshape(1, -1))
    return ax, p_current, p_arrows, p_g_best

#  ---------------- VARIABLE SET-UP -------------------
n_particles = 20
max_iterations = 100
opt_func = 'rosenbrock'
if opt_func == 'rosenbrock':
    f = rosenbrock
    # values for plot size and position
    plot_x1 = -2
    plot_x2 = 2
    plot_y1 = -1
    plot_y2 = 3
    # initialise random locations and velocities for the particles
    np.random.seed(100)
    X = np.random.rand(2, n_particles) * 4
    X[0] = X[0] - 2
    X[1] = X[1] - 1
    V = np.random.randn(2, n_particles) * 0.1

if opt_func == 'egg_carton':
    f = egg_carton
    # values for plot size and position
    plot_x1 = plot_y1 = 0
    plot_x2 = plot_y2 = 5
    # initialise random locations and velocities for the particles
    np.random.seed(100)
    X = np.random.rand(2, n_particles) * 5
    V = np.random.randn(2, n_particles) * 0.1

#  Initialise p_best and g_best
p_best = X
p_best_out = f(X[0], X[1])
g_best = p_best[:, p_best_out.argmin()]
g_best_out = p_best_out.min()

# Set algorithm hyper-parameters
c1 = c2 = 0.1
w = 0.8

#  ---------------- PLOTTING AND ANIMATING ITERATIONS-------------------
# Set-up contour plot
xx, yy = np.array(np.meshgrid(np.linspace(plot_x1, plot_x2, 100), np.linspace(plot_y1, plot_y2, 100)))
z = f(xx, yy)

x_min = xx.ravel()[z.argmin()]
y_min = yy.ravel()[z.argmin()]

fig, ax = plt.subplots(figsize=(8, 6))
img = ax.imshow(z, extent=[plot_x1, plot_x2, plot_y1, plot_y2], origin='lower', cmap='Spectral', alpha=0.75)
fig.colorbar(img, ax=ax)
contours = ax.contour(xx, yy, z, 10, colors='black', alpha=0.4)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")

# Add particles
ax.plot([x_min], [y_min], marker='x', markersize=5, color="black")  # global minimum shown as 'x'
p_current = ax.scatter(X[0], X[1], marker='o', color="blue")
p_arrows = ax.quiver(X[0], X[1], V[0], V[1], color='blue', width=0.005)
p_g_best = plt.scatter([g_best[0]], [g_best[1]], marker='*', s=100, color='black', alpha=0.4)
ax.set_xlim([plot_x1, plot_x2])
ax.set_ylim([plot_y1, plot_y2])


# Animate the iterations
anim = FuncAnimation(fig, animate, frames=list(range(1, max_iterations)), interval=500, blit=False, repeat=True)
anim.save("PSO.gif", dpi=120, writer="imagemagick")

print("PSO found best solution at {}({})={}".format(opt_func, g_best, g_best_out))
print("Global optimal at {}({})={}".format(opt_func, [x_min, y_min], f(x_min, y_min)))