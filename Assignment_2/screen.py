from os import XATTR_SIZE_MAX
from matplotlib import animation
from vpython import *
from motion_model import robot, distance_sensor

scene = canvas(title='Robot Simulator', width=400, height=400, center=vector(0, 0, 0), background=color.white)
ball = sphere(pos=vector(0, 0, 0), radius=1, color=color.green)
sens = curve(ball.pos, ball.pos + vector(10, 0, 0))

wall_length = 40
wall_width = 4
wall_height = 2

right_wall = box(pos=vector(20, 0, 0), size=vector(wall_width, wall_length + wall_width, wall_height))
left_wall = box(pos=vector(-20, 0, 0), size=vector(wall_width, wall_length + wall_width, wall_height))

upper_wall = box(pos=vector(0, 20, 0), size=vector(wall_length - wall_width, wall_width, wall_height))
lower_wall = box(pos=vector(0, -20, 0), size=vector(wall_length - wall_width, wall_width, wall_height))

# initialize start and end points of walls

wall_north = [(-20, 20), (20, 20)]
wall_east = [(20, 20), (20, -20)]
wall_south = [(20, -20), (-20, -20)]
wall_west = [(-20, -20), (-20, 20)]

sensor = distance_sensor(0, wall_north, wall_east, wall_south, wall_west, 10)

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
        sens.clear()
        s, e = sensor.get_pos_vpython(bot.pos)
        sens.append(s, e)

        i += 1


simulation(30)
