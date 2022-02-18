# Who did what:
#  -- Original set-up: Lukas Santing: i6298143
#  -- Edited and refined by all three of us

import numpy as np
# from matplotlib import animation
from vpython import *
from motion_model import *

# ---- SET UP GLOBAL VARIABLES ----
#  Set up vPython graphics environment
scene = canvas(title='Robot Simulator', width=400, height=400, center=vector(0, 0, 0), background=color.white)
ball = sphere(pos=vector(0, 0, 0), radius=1, color=color.green)

# Set up environment variables
wall_length = 40
wall_width = 0.001
wall_height = 2

right_wall = box(pos=vector(20, 0, 0), size=vector(wall_width, wall_length + wall_width, wall_height))
left_wall = box(pos=vector(-20, 0, 0), size=vector(wall_width, wall_length + wall_width, wall_height))

upper_wall = box(pos=vector(0, 20, 0), size=vector(wall_length - wall_width, wall_width, wall_height))
lower_wall = box(pos=vector(0, -20, 0), size=vector(wall_length - wall_width, wall_width, wall_height))

# Define variables and functions for sensor initialization and update
num_sensors = 8
animation_sensors = []
sensor_labels = []

for i in range(num_sensors):
    # just instantiation here, sensors & sensor labels get values in the update function below
    animation_sensors.append(curve(ball.pos, ball.pos))         # vPython sensors (for the animation)
    sensor_labels.append(label(pos=ball.pos, text="Label"))     # vPython labels

def update_all_sensors_pos(robot):
    rays = robot.get_rays_vpython()
    dists = robot.get_distance_to_walls()
    for index, anim_sens in enumerate(animation_sensors):
        anim_sens.clear()       # clear previous sensor
        start = rays[index][0]
        end = rays[index][1]
        anim_sens.append(start, end)
        if index==0:
            anim_sens.append(pos=[start, end], color=color.red)

        # Add distances to labels for sensors
        sensor_labels[index].pos = end
        sensor_labels[index].text = f'{index}: {str(dists[index])}'

# Add details to the dashboard
dashboard = wtext()

def update_distance_dashboard():
    dashboard.text = ''
    distances = bot.get_distance_to_walls()
    #print(distances)
    for i, dist in enumerate(distances):
        dashboard.text += f'{i}: {dist} <br>'
    dashboard.text += f'Speed right wheel: {bot.vel_right} <br>'
    dashboard.text += f'Speed left wheel: {bot.vel_left} <br>'
    dashboard.text += "<br> CONTROLS: <br>"
    dashboard.text += "Accel/Decel left wheel: 'w', 's' <br>"
    dashboard.text += "Accel/Decel right wheel: 'o', 'l' <br>"
    dashboard.text += "Accel/Decel both wheels: 'y', 'h' <br>"
    dashboard.text += "Stop bot in place: 'x' <br>"
    dashboard.text += "Reset bot position: 'r'"

edge_north = [(-20, 20), (20, 20)]
edge_south = [(20, -20), (-20, -20)]
edge_west = [(-20, -20), (-20, 20)]
edge_east = [(20, 20), (20, -20)]
edges = [edge_north, edge_south, edge_west, edge_east]

# initialize instance of Robot class
bot = Robot([0, 0, 0],
            acceleration=0.5,
            num_sensors=num_sensors,
            obstacle_edges=edges,
            wall_distance=20,
            collision_check=True,
            slide_collision_check=False)

# Function that defines a simulation & updates the scene
def simulation(animation_rate):
    i=0
    while (True):
        rate(animation_rate)

        k = keysdown()
        if 'w' in k: bot.accel_left()
        if 's' in k: bot.decel_left()
        if 'o' in k: bot.accel_right()
        if 'l' in k: bot.decel_right()
        if 'y' in k: bot.accel_both()
        if 'h' in k: bot.decel_both()
        if 'r' in k: bot.reset()
        if 'x' in k: bot.stop()
        if 'q' in k: return

        bot.timestep(1/animation_rate)

        # change robot position
        ball.pos = bot.get_pos_vpython()
        
        # change sensor position (to update points coordinates: remove the current points and add the updated values)
        update_all_sensors_pos(bot)
        
        # change distance dashboard
        update_distance_dashboard()

        i += 1


simulation(30)