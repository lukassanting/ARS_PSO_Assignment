import pymunk
import pygame
import math
from motion_model import *

# --- Initialise pygame and make a display appear ---

pygame.init()
display = pygame.display.set_mode((600, 600))

# --- Set global values

# Add pymunk.Space() object, which holds our pymunk physics world
space = pymunk.Space()
# Create clock object for setting the frame-rate for display update
clock = pygame.time.Clock()
FPS = 30
# Grab image for bot showing direction (isn't implemented correctly yet)
# image = pygame.image.load("circle.png")

# set positional value variables
left = top = 10
right = bottom = 590
middle_x = middle_y = 300
bot_radius = 20
wall_size = 10

# set colours for display
white = (255, 255, 255)
black = (0, 0, 0)
display.fill(white)

# initialize start and end points of walls
wall_north = [(left, top), (right, top)]
wall_east = [(right, top), (right, bottom)]
wall_south = [(left, bottom), (right, bottom)]
wall_west = [(left, top), (left, bottom)]


# --- Define classes for pymunk implementation of bot and wall ---

class Bot:
    """" Class for a movable robot (circle) in PyMunk using the motion_model Robot class
    """

    def __init__(self):
        self.robot = robot([middle_x, middle_y, 0], bot_radius * 2, acceleration=10)
        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.body.position = self.robot._pos[0], self.robot._pos[1]
        self.shape = pymunk.Circle(self.body, bot_radius)
        self.shape.density = 1
        self.shape.elasticity = 0;  # no 'bounce'
        space.add(self.body, self.shape)
        self.shape.collision_type = 1

    def draw(self):
        x, y = self.body.position
        angle = math.degrees(self.robot._pos[2])
        # display.blit(pygame.transform.rotate(image, angle), [(int(x) - 15), (int(y)- 15)])
        pygame.draw.circle(display, black, (int(x), int(y)), bot_radius)

    def move(self, key=None):
        if key:
            if key == pygame.K_w: self.robot.accel_left()
            if key == pygame.K_s: self.robot.decel_left()
            if key == pygame.K_o: self.robot.accel_right()
            if key == pygame.K_l: self.robot.decel_right()
            if key == pygame.K_y: self.robot.accel_both()
            if key == pygame.K_h: self.robot.decel_both()
            if key == pygame.K_x: self.robot.stop()
            if key == pygame.K_r: self.robot.reset()
        self.robot.timestep(1 / FPS)
        self.body.position = self.robot._pos[0], self.robot._pos[1]


class Wall:
    """" Class for a static wall in PyMunk
    """

    def __init__(self, p1, p2, collision_number=None):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.shape = pymunk.Segment(self.body, p1, p2, wall_size)
        self.shape.elasticity = 0
        space.add(self.body, self.shape)
        if collision_number:
            self.shape.collision_type = collision_number

    def draw(self):
        pygame.draw.line(display, black, self.shape.a, self.shape.b, wall_size)


class Sensor: # NOTE: Right now I think that the pymunk physics of the location doesn't get updated
    # Can't find how to update the position of a line segment in Pymunk - might need to do it just with pyGame logic
    def __init__(self, s_alpha, s_radius):
        # self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        s_x = (sensor_radius * np.cos(s_alpha)) + middle_x
        s_y = (sensor_radius * np.sin(s_alpha)) + middle_y
        self.start = (middle_x, middle_y)
        self.end = (middle_x + s_x, middle_y + s_y)
        # self.shape = pymunk.Segment(self.body, self.start, self.end, 2)
        # self.shape.sensor = True
        # space.add(self.body, self.shape)
        # self.shape.collision_type = 1

    def draw(self, start, end):
        # self.body = pymunk.Body()
        self.start = start
        self.end = end
        # self.shape = pymunk.Segment(self.body, self.start, self.end, 2)
        # space.add(self.body, self.shape)
        pygame.draw.line(display, (0, 128, 0), self.start, self.end, 2)


# --- Setting up sensors

# set up sensors
sensor_radius = 200
animation_sensors = []
model_sensors = []


def degrees_to_radians(angle):
    return angle * (np.pi / 180)


for alpha in np.linspace(0, 360, 12, endpoint=False):
    alpha = degrees_to_radians(alpha)
    sensor = Sensor(alpha, sensor_radius)
    animation_sensors.append(sensor)
    model_sensors.append(distance_sensor(alpha, wall_north, wall_east, wall_south, wall_west, sensor_radius))


def update_all_sensors_pos(bot):
    bot_pos = bot._pos
    for anim_sens, model_sens in zip(animation_sensors, model_sensors):
        start, end = model_sens.get_pos_vpython(bot_pos)
        anim_sens.draw((start.x, start.y), (end.x, end.y))

        # object detection
        model_sens.object_detected(bot_pos)


# sensor_1 = distance_sensor(0, wall_north, wall_east, wall_south, wall_west, 10)
# sensor_2 = distance_sensor(30, wall_north, wall_east, wall_south, wall_west, 10)


# --- The method where the simulation happens ---
def simulation():
    # set events that can happen while in the display window
    bot = Bot()
    wall_left = Wall([left, top], [left, bottom], 1)
    wall_right = Wall([right, top], [right, bottom], 1)
    wall_top = Wall([left, top], [right, top])
    wall_bottom = Wall([left, bottom], [right, bottom])

    while True:
        key = "none"
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # pygame.QUIT gets called when we press the 'x' button on the window
                return  # exit out of the function call to end the display
            if event.type == pygame.KEYDOWN:
                key = event.key

        bot.move(key)

        display.fill(white)
        bot.draw()
        wall_left.draw()
        wall_right.draw()
        wall_top.draw()
        wall_bottom.draw()
        update_all_sensors_pos(bot.robot)

        # Add here: edited version of 'update sensors'

        # update the display - using clock object to set frame-rate
        pygame.display.update()
        # clock.tick(FPS)
        # pass some time in the simulation
        space.step(1 / FPS)  # basis: correlate with FPS. Low val = more accurate simulation, but slower program


# call the function simulation to keep the display running until we quit
simulation()

# End the pygame display
pygame.quit()
