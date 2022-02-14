import numpy as np
from matplotlib import animation
from vpython import *
from motion_model import robot

scene = canvas(title='Robot Simulator', width=400, height=400, center=vector(0, 0, 0), background=color.white)
ball = sphere(pos=vector(0, 0, 0), radius=1, color=color.green)


# create all the sensors
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
        anim_sens.clear()   # clear previous sensor
        start = rays[index][0]
        end = rays[index][1]
        anim_sens.append(start, end)
        # Add distances to labels to sensors
        sensor_labels[index].pos = end
        sensor_labels[index].text = f'{index}: {str(dists[index])}'

dashboard = wtext()

wall_length = 40
wall_width = 0.001
wall_height = 2

right_wall = box(pos=vector(20, 0, 0), size=vector(wall_width, wall_length + wall_width, wall_height))
left_wall = box(pos=vector(-20, 0, 0), size=vector(wall_width, wall_length + wall_width, wall_height))

upper_wall = box(pos=vector(0, 20, 0), size=vector(wall_length - wall_width, wall_width, wall_height))
lower_wall = box(pos=vector(0, -20, 0), size=vector(wall_length - wall_width, wall_width, wall_height))

bot = robot([0, 0, 0], acceleration=0.5, num_sensors=num_sensors)

def update_distance_dashboard():
    dashboard.text = ''
    distances = bot.get_distance_to_walls()
    print(distances)
    for i, dist in enumerate(distances):
        dashboard.text += f'{i}: {dist} <br>'

def simulation(animation_rate):
    i=0
    while (i<3000):
        rate(animation_rate)
        
        k = keysdown()
        if 'w' in k: bot.accel_left(verbose=True)
        if 's' in k: bot.decel_left(verbose=True)
        if 'o' in k: bot.accel_right(verbose=True)
        if 'l' in k: bot.decel_right(verbose=True)
        if 'y' in k: bot.accel_both(verbose=True)
        if 'h' in k: bot.decel_both(verbose=True)
        if 'r' in k: bot.reset()
        if 'x' in k: bot.stop()

        bot.timestep(1/animation_rate)

        # change robot position
        ball.pos = bot.get_pos_vpython()
        
        # change sensor position (to update points coordinates: remove the current points and add the updated values)
        update_all_sensors_pos(bot)
        
        # change distance dashboard
        update_distance_dashboard()

        i += 1


simulation(30)
