import pymunk
import pygame
import math
from motion_model import *
import numpy as np
from pymunk_classes import *

# -------- Pymunk Animation -------- #
# ---- UPDATED FOR ASSIGNMENT 3 ---- #


# Define the locations of obstacle edges, used both pygame/pymunk, & for sensors (for intersection & collision checks)
# edge_north = [(10, 5), (990, 5)]
# edge_south = [(200, 595), (990, 595)]
# edge_west = [(5, 0), (5, 400)]
# edge_east = [(990, 0), (990, 600)]
# edge_sw = [(5, 400), (200, 595)]
# edge_mid_n = [(350, 10), (350, 100)]
# edge_mid_s = [(350, 250), (350, 595)]
# edges = [edge_north, edge_south, edge_east, edge_west, edge_sw, edge_mid_s, edge_mid_n]


def simulation(bot, walls, display, space, FPS=30, walk_time=5000):

    # set colours for display
    white = (255, 255, 255)
    black = (0, 0, 0)
    display.fill(white)

    # Main loop of the simulation
    while(walk_time>0):
        # Move robot based on keyboard input
        key = "none"
        for event in pygame.event.get():
            if event.type == pygame.QUIT:   # pygame.QUIT gets called when we press the 'x' button on the window
                return                      # exit out of the function call to end the display
            if event.type == pygame.KEYDOWN:
                key = event.key
        bot.move(key)

        # Update the pygame display color & draw the elements
        display.fill(white)
        bot.draw()
        bot.draw_sensors()
        for wall in walls:
            wall.draw()
        pygame.display.update()

        # pass some time in the simulation
        space.step(1 / FPS)  # basis: correlate with FPS. Low val = more accurate simulation, but slower program
        walk_time = walk_time - 1

# # Make the motion-model Robot
# motion_model_robot = Robot([0, 0, 0],
#               distance_between_wheels=bot_radius*2,
#               robot_body_radius=bot_radius,
#               acceleration=30,
#               num_sensors=8,
#               sensor_measuring_distance=30,
#               obstacle_edges=edges,
#               wall_distance=280,          # !! doesn't match edge locations: expects square room from old impl !!
#               collision_check=True,      # !! if not using pymunk collision, wall_distance needs to be correct before setting collision to True !!
#               slide_collision_check=False,
#               pymunk_offset=[100,400,0])  # x, y position offset, as backend logic is based on 0,0
#
# # Make the pymunk-pygame Bot, taking the motion_model Robot as an argument
# pymunk_bot = Pymunk_Bot(robot=motion_model_robot,
#                  pygame_display=display,
#                  pymunk_space=space,
#                  radius=bot_radius,
#                  color=black,
#                  dust_grid=dust_grid,
#                  pymunk_collision=True)

# Use earlier defined obstacle edges to create walls for Pymunk Visuals & Pygame Graphics
# pymunk_walls = []
# for edge in edges:
#     pymunk_walls.append(Pymunk_Obstacle(pygame_display=display, pymunk_space=space, radius=10, color=black, p=edge))

# call the function simulation to keep the display running until we quit
# simulation(pymunk_bot, pymunk_walls, FPS=30, walk_time=1000)
# print(fitness_func(pymunk_bot))

# End the pygame display
# pygame.quit()

