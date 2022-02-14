import numpy as np
from os import XATTR_SIZE_MAX
from matplotlib import animation
from vpython import *
from motion_model import robot, distance_sensor

scene = canvas(title='Robot Simulator', width=400, height=400, center=vector(0, 0, 0), background=color.white)
ball = sphere(pos=vector(0, 0, 0), radius=1, color=color.green)

SENSOR_RADIUS = 10

# initialize start and end points of walls

wall_north = [(-20, 20), (20, 20)]
wall_east = [(20, 20), (20, -20)]
wall_south = [(20, -20), (-20, -20)]
wall_west = [(-20, -20), (-20, 20)]

# create all the sensors

def degrees_to_radians(angle):
    return angle*(np.pi/180)

animation_sensors = []
model_sensors = []
for alpha in np.linspace(0, 360, 12, endpoint=False):
    alpha = degrees_to_radians(alpha)
    animation_sensors.append(curve(ball.pos, ball.pos + vector(SENSOR_RADIUS * np.cos(alpha), SENSOR_RADIUS * np.sin(alpha), 0)))
    model_sensors.append(distance_sensor(alpha, wall_north, wall_east, wall_south, wall_west, 10))

def update_all_sensors_pos(bot_pos):
    for anim_sens, model_sens in zip(animation_sensors, model_sensors):
        anim_sens.clear()
        start, end = model_sens.get_pos_vpython(bot_pos)
        anim_sens.append(start, end)

        # object detection
        model_sens.object_detected(bot_pos)

wall_length = 40
wall_width = 4
wall_height = 2

right_wall = box(pos=vector(20, 0, 0), size=vector(wall_width, wall_length + wall_width, wall_height))
left_wall = box(pos=vector(-20, 0, 0), size=vector(wall_width, wall_length + wall_width, wall_height))

upper_wall = box(pos=vector(0, 20, 0), size=vector(wall_length - wall_width, wall_width, wall_height))
lower_wall = box(pos=vector(0, -20, 0), size=vector(wall_length - wall_width, wall_width, wall_height))

sensor_1 = distance_sensor(0, wall_north, wall_east, wall_south, wall_west, 10)
sensor_2 = distance_sensor(30, wall_north, wall_east, wall_south, wall_west, 10)

bot = robot([0, 0, 0], 1, acceleration=0.5)

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
        update_all_sensors_pos(bot.pos)
        

        i += 1


simulation(30)
