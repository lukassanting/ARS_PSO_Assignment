import pymunk
import pygame
from motion_model import *

class Pymunk_Bot:
    """ Class for a movable robot (circle) in PyMunk using the motion_model Robot class
    """
    def __init__(self, robot, pygame_display, pymunk_space, radius, color):
        self.bot = robot
        self.pygame_display = pygame_display
        self.pymunk_space = pymunk_space
        self.radius = radius
        self.color = color

        self.body = pymunk.Body()
        self.body.position = self.to_pygame(self.bot.get_pos_pygame()[0]), self.to_pygame(self.bot.get_pos_pygame()[1])
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.density = 1
        self.shape.elasticity = 0
        self.pymunk_space.add(self.body, self.shape)
        self.shape.collision_type = 1

    def move(self, key=None):
        if key:
            if key == pygame.K_w: self.bot.accel_left()
            if key == pygame.K_s: self.bot.decel_left()
            if key == pygame.K_o: self.bot.accel_right()
            if key == pygame.K_l: self.bot.decel_right()
            if key == pygame.K_y: self.bot.accel_both()
            if key == pygame.K_h: self.bot.decel_both()
            if key == pygame.K_x: self.bot.stop()
            if key == pygame.K_r: self.bot.reset()
        bot_velocity = self.bot.get_velocity()
        self.body.velocity = bot_velocity[0], bot_velocity[1]

    def draw(self):
        pygame.draw.circle(self.pygame_display, self.color, self.body.position, self.radius)

    def to_pygame(self, value):
        return value + 300


class Pymunk_Obstacle:
    """ Class for ...
    """
    def __init__(self, pygame_display, pymunk_space, radius, color, p1, p2):
        self.pygame_display = pygame_display
        self.pymunk_space = pymunk_space
        self.radius = radius
        self.color = color
        self.p1 = p1
        self.p2 = p2

        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.shape = pymunk.Segment(self.body, self.p1, self.p2, self.radius)
        self.shape.elasticity = 1
        pymunk_space.add(self.body, self.shape)

    def draw(self):
        pygame.draw.line(self.pygame_display, self.color, self.p1, self.p2, self.radius)
