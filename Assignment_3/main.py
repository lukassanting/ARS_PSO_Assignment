
import numpy as np
import sys
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ann import *
import helper
from evolutionary_algorithm import *

# ***************** STEP 1 - DEFINE FUNCTION TO OPTIMISE HERE:
opt_func = "rosenbrock"  # Set to "rosenbrock" or to "rastigrin"

# ---------------- FUNCTIONS -------------------
#  Rosenbrock function to optimise
def rosenbrock(x, y, a=0, b=150):
    return ((a - x) ** 2) + b * ((y - (x ** 2)) ** 2)

def neg_rosenbrock(x, y, a=0, b=150):
    return (-1)*((a - x) ** 2) + b * ((y - (x ** 2)) ** 2)


def rosenbrock_grad(x, y, a=1, b=150):
    return [(2 * (x - a) - 4 * b * x * (y - (x ** 2))), (2 * b * (y - (x ** 2)))]


def rastigrin(x, y):
    return 20 + x ** 2 - 10 * np.cos(2 * np.pi * x) + y ** 2 - 10 * np.cos(2 * np.pi * y)


def rastigrin_grad(x, y):
    return [2 * x + 20 * np.pi * np.sin(2 * np.pi * x), 2 * y + 20 * np.pi * np.sin(2 * np.pi * y)]

#  ---------------- EVOLUTION -------------------

# evolution(ann_layers=(2,12,4,2), bias_nodes=(False,True,True,False), pop_size=10, verbose=False)
population = Population(num_individuals=5, ann_layers=(2,12,4,2), bias_nodes=(False,True,True,False), fitness_func=neg_rosenbrock)
population.evolution(num_generations = 5, time_for_generation=5, get_ann_inputs=rosenbrock_grad, width=3, mutation_rate=0.0001)
fig = population._history.plot_fitness()
anim = population._history.animate_positions()

# def initialise_in_function(func:str = "rosenbrock", population_size:int = 10):
#     if func == "rosenbrock":
#         XY = np.random.rand(2, population_size) * 4
#         XY[0] = XY[0] - 2  # Change range of particles on X-axis to be between -2 and 2
#         XY[1] = XY[1] - 1  # Change range of particles on Y-axis to be between -1 and 3
#         # print(f"INIT = {XY[0]}, {XY[1]}")
#         return XY
#     else:    # func == "rastigrin"
#         XY = np.random.rand(2, population_size) * 10
#         XY = XY - 5  # Change range of particles on X-axis and Y-axis to be between -5 and 5
#         return XY
#
# def get_gradients(func, XY_coords, pop_size=10):
#     XY_grads = np.full_like(XY_coords, 1)
#     if func =="rosenbrock":
#         for i in range(pop_size):
#             out_x, out_y = rosenbrock_grad(XY_coords[0][i], XY_coords[1][i])
#             XY_grads[0][i] = out_x
#             XY_grads[1][i] = out_y
#     else:   # func == "rastigrin"
#         for i in range(pop_size):
#             out_x, out_y = rastigrin_grad(XY_coords[0][i], XY_coords[1][i])
#             XY_grads[0][i] = out_x
#             XY_grads[1][i] = out_y
#     return XY_grads
#
#
# def walk_around(networks, XY_coords, pop_size, step_size=1, verbose=False):
#     while step_size > 0:
#         for i in range(pop_size):
#             XY_grads = get_gradients(opt_func, XY_coords)
#
#             NN_coords = networks[i].prop_forward(input_sensors=np.array([XY_grads[0][i], XY_grads[1][i]]))
#             if verbose: print(f"Old coordinates = {XY_coords[0][i]}, {XY_coords[1][i]}")
#             XY_coords[0][i] += NN_coords[0]
#             XY_coords[1][i] += NN_coords[1]
#             if verbose: print(f"New coordinates = {XY_coords[0][i]}, {XY_coords[1][i]}")
#         step_size -= 1
#     return XY_coords


# def evolution(
#         ann_layers:Tuple[int],
#         bias_nodes:Tuple[bool],
#         pop_size:int = 10,
#         step_size:int = 1,
#         verbose:bool = False):
#     XY_coords = initialise_in_function(opt_func, 10)    # Will be used for animation later
#     # size = helper.get_network_size(ann_layers)
#     population = Population(num_individuals=pop_size, ann_layers=ann_layers, bias_nodes=bias_nodes, fitness_func=rosenbrock)
#     # population = [Individual(0, size) for i in range(population_size)]
#     # networks = [helper.array_to_network(individual.float_genotype, network_layers, bias) for individual in population]
#     # XY_coords = walk_around(networks, XY_coords, population_size, step_size, verbose)



# VALUES ARE SET DEPENDENT ON CHOSEN FUNCTION
# Set values of variables for algorithm and plot for optimising Rosenbrock algorithm
# if opt_func == "rosenbrock":
#     f = rosenbrock
#     # Initialise location range of particle on X and Y-axes to be between 0 and 4
#     X = np.random.rand(2) * 4
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
#     X = np.random.rand(2) * 10
#     X = X - 5  # Change range of particles on X-axis and Y-axis to be between -5 and 5
#     # Set values used for defining plot size and location to fit specified function bounds
#     plot_xlow = plot_ylow = -5
#     plot_xhigh = plot_yhigh = 5


# --------------------- Initialise p_best and g_best ---------------------------
#  ---- p_best      = for each particle, the x and y co-ords of personal best so far
#  ---- p_best_out  = Personal best values from function for each particle
#  ---- g_best      = co-ords of the global best particle
#  ---- g_best_out  = function value of the global best particle

# p_best = X
# p_best_out = f(X[0], X[1])
#
# best_x = p_best[0, p_best_out.argmin()]
# best_y = p_best[1, p_best_out.argmin()]
# g_best = np.array([best_x, best_y])
# g_best_out = p_best_out.min()


# Function to do an iteration of PSO
# def particle_swarm_optimization(current_iter=1, a_start=0.9, a_end=0.9, b_start=2, b_end=0.1, c_start=2, c_end=0.1,
#             max_iter=1):
#     global V, X, p_best, p_best_out, g_best, g_best_out
#     r1, r2 = np.random.rand(2)
#     # decay values of a (value for current direction)
#     a = np.round(a_start - ((a_start - a_end) / max_iter) * current_iter, decimals=3)
#     # decay values of b and c (values for personal best and global best directions)
#     b = np.round(b_start - ((b_start - b_end) / max_iter) * current_iter, decimals=3)
#     c = np.round(c_start - ((c_start - c_end) / max_iter) * current_iter, decimals=3)
#     V = a * V + (b * r1 * (p_best - X)) + (c * r2 * (g_best.reshape(-1, 1) - X))
#     X = X + V
#     # Update p_best and g_best
#     f_out = f(X[0], X[1])
#     for i in range(n_particles):
#         if(p_best_out[i] >= f_out[i]):
#             p_best[0,i] = X[0,i]
#             p_best[1,i] = X[1,i]
#     p_best_out = np.array([p_best_out, f_out]).min(axis=0)
#     g_best[0] = p_best[0, p_best_out.argmin()]
#     g_best[1] = p_best[1, p_best_out.argmin()]
#     g_best_out = p_best_out.min()

#  ---------------- PLOTTING AND ANIMATING ITERATIONS-------------------
#
#
# # Set-up contour plot of function
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
# # p_arrows = ax.quiver(X[0], X[1], V[0], V[1], color='blue', width=0.005)     # vectors of particles
# # p_g_best = plt.scatter([g_best[0]], [g_best[1]], marker='*', s=100, color='black', alpha=0.4) # global best
# ax.set_xlim([plot_xlow, plot_xhigh])
# ax.set_ylim([plot_ylow, plot_yhigh])
#
# # Function to animate iterations
# def animate(i, *fargs):
#     title = 'Iteration {:02d}'.format(i)
#     evolution()
#     ax.set_title(title)
#     p_current.set_offsets(X.T)
#     # p_arrows.set_offsets(X.T)
#     # p_arrows.set_UVC(V[0], V[1])
#     # p_g_best.set_offsets(g_best.reshape(1, -1))
#     return ax, p_current, #p_arrows, p_g_best
#
# max_iterations = 250
# anim = FuncAnimation(fig, animate, frames=list(range(1, max_iterations)), interval=150, blit=False, repeat=True, fargs=(max_iterations))
# anim.save("PSO_{}.gif".format(opt_func), dpi=120, writer="ffmpeg")

# print("Global best found at: {}".format(g_best))