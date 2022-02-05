# based on: https://machinelearningmastery.com/a-gentle-introduction-to-particle-swarm-optimization/

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# ---------------- DEFINE FUNCTIONS -------------------
#  Rosenbrock function to optimise
def rosenbrock(x, y, a=0, b=150):
    return ((a - x) ** 2) + b * ((y - (x ** 2)) ** 2)

#  ---------------- VARIABLE SET-UP -------------------
# Set-up contour plot
x, y = np.array(np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-1, 3, 100)))
z = rosenbrock(x, y)
x_min = x.ravel()[z.argmin()]
y_min = y.ravel()[z.argmin()]

# Set algorithm hyper-parameters
c1 = c2 = 0.1
w = 0.8

# initialise random locations and velocities for the particles
n_particles = 20
np.random.seed(100)
X = np.random.rand(2, n_particles) * 4
# change values of X and Y for rosenbrock function (see image in slides)
X[0] = X[0] - 2
X[1] = X[1] - 1
V = np.random.randn(2, n_particles) * 0.1 # vector values

# Initialise p_best and g_best
#  ---- P_best = for each particle, the x and y co-ords of personal best so far
#  ---- P_best_out = Personal best values from function for each particle
#  ---- G_best = co-ords of the global best particle
#  ---- G_best_out = function value of the global best particle

p_best = X
p_best_out = rosenbrock(X[0], X[1])
g_best = p_best[:, p_best_out.argmin()]
g_best_out = p_best_out.min()

# Function to do an iteration of PSO
def iterate():
    global V, X, p_best, p_best_out, g_best, g_best_out
    r1, r2 = np.random.rand(2)
    V = w * V + (c1 * r1 * (p_best - X)) + (c2 * r2 * (g_best.reshape(-1, 1) - X))
    X = X + V
    # Update p_best and g_best
    f_out = rosenbrock(X[0], X[1])
    p_best[:, (p_best_out >= f_out)] = X[:, (p_best_out >= f_out)]
    p_best_out = np.array([p_best_out, f_out]).min(axis=0)
    g_best = p_best[:, p_best_out.argmin()]
    g_best_out = p_best_out.min()

#  ---------------- PLOTTING AND ANIMATING ITERATIONS-------------------

fig, ax = plt.subplots(figsize=(8, 6))
img = ax.imshow(z, extent=[-2, 2, -1, 3], origin='lower', cmap='Spectral', alpha=0.75)
fig.colorbar(img, ax=ax)
contours = ax.contour(x, y, z, 10, colors='black', alpha=0.4)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")

# Add particles
ax.plot([x_min], [y_min], marker='x', markersize=5, color="black")  # global minimum shown as 'x'
p_current = ax.scatter(X[0], X[1], marker='o', color="blue")
p_arrows = ax.quiver(X[0], X[1], V[0], V[1], color='blue', width=0.005)
p_g_best = plt.scatter([g_best[0]], [g_best[1]], marker='*', s=100, color='black', alpha=0.4)
ax.set_xlim([-2, 2])
ax.set_ylim([-1, 3])

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

max_iterations = 200
anim = FuncAnimation(fig, animate, frames=list(range(1, max_iterations)), interval=500, blit=False, repeat=True)
anim.save("PSO_rosenbrock.gif", dpi=120, writer="imagemagick")

print("PSO found best solution at rosenbrock({})={}".format(g_best, g_best_out))
print("Global optimal at rosenbrock({})={}".format([x_min, y_min], rosenbrock(x_min, y_min)))