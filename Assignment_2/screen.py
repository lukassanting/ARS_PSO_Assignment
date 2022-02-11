from vpython import *
from motion_model import robot

scene = canvas(title='Robot Simulator', width=400, height=400, center=vector(0, 0, 0), background=color.white)
ball = sphere(pos=vector(0, 0, 0), radius=1, color=color.green)

wall_length = 40
wall_width = 4
wall_height = 2

right_wall = box(pos=vector(20, 0, 0), size=vector(wall_width, wall_length + wall_width, wall_height))
left_wall = box(pos=vector(-20, 0, 0), size=vector(wall_width, wall_length + wall_width, wall_height))

upper_wall = box(pos=vector(0, 20, 0), size=vector(wall_length - wall_width, wall_width, wall_height))
lower_wall = box(pos=vector(0, -20, 0), size=vector(wall_length - wall_width, wall_width, wall_height))

bot = robot([0, 0, 0], 1)

def simulation():
    i = 0
    while (i<3000):
        rate(30)
        k = keysdown()
        if 'w' in k: bot.accel_left()
        if 's' in k: bot.decel_left()
        if 'o' in k: bot.accel_right()
        if 'l' in k: bot.decel_right()
        if 'y' in k:
            bot.accel_left()
            bot.accel_right()
        if 'h' in k:
            bot.decel_left()
            bot.decel_right()
        bot.timestep()
        ball.pos = bot.get_pos_vpython()
        i += 1


simulation()
