import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# ***************** STEP 1 - DEFINE FUNCTION TO OPTIMISE HERE:
opt_func = "rosenbrock"  # Set to "rosenbrock" or to "rastigrin"


# ---------------- FUNCTIONS -------------------
#  Rosenbrock function to optimise
def rosenbrock(x, y, a=0, b=150):
    return ((a - x) ** 2) + b * ((y - (x ** 2)) ** 2)


def rastigrin(x, y):
    return 20 + x ** 2 - 10 * np.cos(2 * np.pi * x) + y ** 2 - 10 * np.cos(2 * np.pi * y)


#  ---------------- PARTICLE SWARM OPTIMIZATION -------------------
# Set algorithm hyper-parameters
c1_start = c2_start = 1  # Good initial value = 1
c1_end = c2_end = 0.1  # Good initial value = 0.5
# w = 0.8
inertia_start = 0.9
inertia_end = 0.4

# initialise random locations and velocities for the particles
n_particles = 20
np.random.seed()
V = np.random.randn(2, n_particles) * 0.1  # vector values

#  ---- SET VALUES THAT ARE DEPENDENT ON CHOSEN FUNCTION ----

# Set values of variables for algorithm and plot for optimising Rosenbrock algorithm
if opt_func == "rosenbrock":
    f = rosenbrock
    # Initialise location range of particles on X and Y-axes to be between 0 and 4
    X = np.random.rand(2, n_particles) * 4
    X[0] = X[0] - 2  # Change range of particles on X-axis to be between -2 and 2
    X[1] = X[1] - 1  # Change range of particles on Y-axis to be between -1 and 3
    # Set values used for defining plot size and location to fit specified function bounds
    plot_xlow = -2
    plot_xhigh = 2
    plot_ylow = -1
    plot_yhigh = 3

# Set values of variables for algorithm and plot for optimising Rastigrin algorithm
if opt_func == "rastigrin":
    f = rastigrin
    # Initialise location range of particles on X and Y-axes to be between 0 and 10
    X = np.random.rand(2, n_particles) * 10
    X = X - 5  # Change range of particles on X-axis and Y-axis to be between -5 and 5
    # Set values used for defining plot size and location to fit specified function bounds
    plot_xlow = plot_ylow = -5
    plot_xhigh = plot_yhigh = 5

# --------------------- Initialise p_best and g_best ---------------------------
#  ---- P_best = for each particle, the x and y co-ords of personal best so far
#  ---- P_best_out = Personal best values from function for each particle
#  ---- G_best = co-ords of the global best particle
#  ---- G_best_out = function value of the global best particle

p_best = X
p_best_out = f(X[0], X[1])
g_best = p_best[:, p_best_out.argmin()]
g_best_out = p_best_out.min()


# Function to do an iteration of PSO
def iterate(current_iter=1, inertia_start=0.9, inertia_end=0.9, c1_start=2, c1_end=0.1, c2_start=2, c2_end=0.1,
            max_iter=1):
    w = np.round(inertia_start - ((inertia_start - inertia_end) / max_iter) * current_iter, decimals=3)
    c1 = np.round(c1_start - ((c1_start - c1_end) / max_iter) * current_iter, decimals=3)
    c2 = np.round(c2_start - ((c2_start - c2_end) / max_iter) * current_iter, decimals=3)
    global V, X, p_best, p_best_out, g_best, g_best_out
    r1, r2 = np.random.rand(2)
    V = w * V + (c1 * r1 * (p_best - X)) + (c2 * r2 * (g_best.reshape(-1, 1) - X))
    X = X + V
    # Update p_best and g_best
    f_out = f(X[0], X[1])
    p_best[:, (p_best_out >= f_out)] = X[:, (p_best_out >= f_out)]
    p_best_out = np.array([p_best_out, f_out]).min(axis=0)
    g_best = p_best[:, p_best_out.argmin()]
    g_best_out = p_best_out.min()


#  ---------------- PLOTTING AND ANIMATING ITERATIONS-------------------

# Set-up contour plot
x, y = np.array(np.meshgrid(np.linspace(plot_xlow, plot_xhigh, 1000), np.linspace(plot_ylow, plot_yhigh, 1000)))
z = f(x, y)
x_min = x.ravel()[z.argmin()]
y_min = y.ravel()[z.argmin()]

fig, ax = plt.subplots(figsize=(8, 6))
img = ax.imshow(z, extent=[plot_xlow, plot_xhigh, plot_ylow, plot_yhigh], origin='lower', cmap='Spectral', alpha=0.75)
fig.colorbar(img, ax=ax)
contours = ax.contour(x, y, z, 10, colors='black', alpha=0.4)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")

# Add particles
ax.plot([x_min], [y_min], marker='x', markersize=5, color="black")  # global minimum shown as 'x'
p_current = ax.scatter(X[0], X[1], marker='o', color="blue")
p_arrows = ax.quiver(X[0], X[1], V[0], V[1], color='blue', width=0.005)
p_g_best = plt.scatter([g_best[0]], [g_best[1]], marker='*', s=100, color='black', alpha=0.4)
ax.set_xlim([plot_xlow, plot_xhigh])
ax.set_ylim([plot_ylow, plot_yhigh])


# Function to animate iterations
def animate(i, *fargs):
    title = 'Iteration {:02d}'.format(i)
    iterate(current_iter=i,
            inertia_start=fargs[0],
            inertia_end=fargs[1],
            c1_start=fargs[2],
            c1_end=fargs[3],
            c2_start=fargs[4],
            c2_end=fargs[5],
            max_iter=fargs[6])
    ax.set_title(title)
    p_current.set_offsets(X.T)
    p_arrows.set_offsets(X.T)
    p_arrows.set_UVC(V[0], V[1])
    p_g_best.set_offsets(g_best.reshape(1, -1))
    return ax, p_current, p_arrows, p_g_best


max_iterations = 250
anim = FuncAnimation(fig, animate, frames=list(range(1, max_iterations)), interval=150, blit=False, repeat=True, fargs=(
    inertia_start,
    inertia_end,
    c1_start,
    c1_end,
    c2_start,
    c2_end,
    max_iterations))
anim.save("PSO_{}.gif".format(opt_func), dpi=120, writer="ffmpeg")

print("PSO found best solution at ({})={}".format(g_best, g_best_out))
print("Global optimal at ({})={}".format([x_min, y_min], f(x_min, y_min)))
