import pymunk
import pygame
import math
from motion_model import *
from pymunk_classes import *

# -------- Pymunk Animation -------- #
# ---- UPDATED FOR ASSIGNMENT 3 ---- #

# --- Initialise pygame and make a display appear ---
pygame.init()
display = pygame.display.set_mode((600, 600))

# --- Set global values
space = pymunk.Space()  # Holds our pymunk physics space
clock = pygame.time.Clock()

# positional value variables
# left = top = 10
# right = bottom = 580
# middle_x = middle_y = 300
# wall_size = 10
bot_radius = 20

# set colours for display
white = (255, 255, 255)
black = (0, 0, 0)
display.fill(white)

def simulation(FPS=30):
    motion_model_robot = Robot([0, 0, 1], distance_between_wheels=bot_radius*2, robot_body_radius=bot_radius, acceleration=5, num_sensors=8, wall_distance=280, collision_check=False)
    bot = Pymunk_Bot(robot=motion_model_robot, pygame_display=display, pymunk_space=space, radius=bot_radius, color=black)
    wall_right=Pymunk_Obstacle(pygame_display=display, pymunk_space=space, radius=10, color=black, p1=[580, 10], p2=[580, 580])
    wall_left=Pymunk_Obstacle(pygame_display=display, pymunk_space=space, radius=10, color=black, p1=[10, 10], p2=[10, 580])
    wall_bottom=Pymunk_Obstacle(pygame_display=display, pymunk_space=space, radius=10, color=black, p1=[10, 580], p2=[580, 580])
    wall_top=Pymunk_Obstacle(pygame_display=display, pymunk_space=space, radius=10, color=black, p1=[10, 10], p2=[580, 10])

    while(True):
        key = "none"
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # pygame.QUIT gets called when we press the 'x' button on the window
                return  # exit out of the function call to end the display
            if event.type == pygame.KEYDOWN:
                key = event.key

        bot.move(key)
        display.fill(white)
        bot.draw()
        wall_right.draw()
        wall_left.draw()
        wall_bottom.draw()
        wall_top.draw()

        # update the display - using clock object to set frame-rate
        pygame.display.update()
        # pass some time in the simulation
        space.step(1 / FPS)  # basis: correlate with FPS. Low val = more accurate simulation, but slower program




# call the function simulation to keep the display running until we quit
simulation(FPS=30)

# End the pygame display
pygame.quit()

