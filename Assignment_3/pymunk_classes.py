import pymunk
import pygame

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

    def move(self, key):
        return

    def draw(self):
        pygame.draw.circle(self.pygame_display, self.color, self.body.position, self.radius)

    def to_pygame(self, value):
        return value + 300


class Pymunk_Obstacle:
    """ Class for ...
    """
    def __init__(self):
        return

    def draw(self):
        return
