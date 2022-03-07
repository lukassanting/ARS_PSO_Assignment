import pymunk
import pygame
from motion_model import *


class Pymunk_Bot:
    """ Class for a movable robot (circle) in PyMunk using the motion_model Robot class
    """

    def __init__(self, robot, pygame_display, pymunk_space, radius, color, dust_grid, ann=None, pymunk_collision=True):
        self.bot = robot
        self.pygame_display = pygame_display
        self.pymunk_space = pymunk_space
        self.radius = radius
        self.color = color
        self.pymunk_collision = pymunk_collision
        self.hitting_wall = False
        self.dust_grid = dust_grid
        self.ann = ann
        self.movement_type = 'keys'
        if ann is not None: self.movement_type = 'ann'
        self.collision_counter = 0

        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        if self.pymunk_collision:
            self.body = pymunk.Body()
        self.body.position = self.bot.get_pos_pygame()[0] + 100, self.bot.get_pos_pygame()[1] + 400
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.density = 1
        self.shape.elasticity = 0
        self.pymunk_space.add(self.body, self.shape)
        self.shape.collision_type = 1

    def move(self, key=None, FPS=30):
        # Case 1: Using pymunk auto collision, not our own collision
        if self.pymunk_collision:
            # Case 1.1: Manual movement (using keys)
            if self.movement_type == 'keys':
                if key:
                    if key == pygame.K_w: self.bot.accel_left()
                    if key == pygame.K_s: self.bot.decel_left()
                    if key == pygame.K_o: self.bot.accel_right()
                    if key == pygame.K_l: self.bot.decel_right()
                    if key == pygame.K_y: self.bot.accel_both()
                    if key == pygame.K_h: self.bot.decel_both()
                    if key == pygame.K_x: self.bot.stop()
                    if key == pygame.K_r: self.bot.reset()
                bot_velocity = self.bot.get_xy_velocity(1 / FPS)
            # Case 1.2: Auto movement (using ANN)
            else:
                distances = [sensor._activations for sensor in self.bot._sensors]
                ann_velocity = self.ann.prop_forward(distances)
                if (ann_velocity[0] > 0 and ann_velocity[1] > 0): self.bot.accel_both()
                elif (ann_velocity[0] < 0 and ann_velocity[1] < 0): self.bot.decel_both()
                else:
                    if ann_velocity[0] > 0: self.bot.accel_left()
                    if ann_velocity[0] < 0: self.bot.decel_left()
                    if ann_velocity[1] > 0: self.bot.accel_right()
                    if ann_velocity[1] < 0: self.bot.decel_right()
                bot_velocity = self.bot.get_xy_velocity(1 / FPS)
            bot_velocity = self.cap_velocity(bot_velocity)  # cap the velocity to be between -25 and 25
            self.body.velocity = bot_velocity[0], bot_velocity[1]  # set body (pymunk) velocity
            self.bot.pymunk_position_update(self.body.position)  # update position in the motion_model

            # Check if there is a collision, update counter for collisions
            sensor_check = False
            for sensor in self.bot._sensors:
                sensor.object_detected(self.bot._pymunk_position, verbose=False)
                if sensor._hitting_wall:
                    self.hitting_wall = True
                    sensor_check = True
            if sensor_check == False:
                if self.hitting_wall:
                    self.hitting_wall = False
                    self.collision_counter = self.collision_counter + 1

            # Update dust_particles grid based on bot location
            grid_position = self.body.position / 20
            grid_x, grid_y = math.ceil(grid_position[0]), math.ceil(grid_position[1])
            if self.dust_grid[grid_x, grid_y] == 0:
                self.dust_grid[grid_x, grid_y] = 1

        # Case 2: Using our own collision, not pymunk collison (NOT USED)
        else:
            self.bot.timestep(1 / 30)
            self.body.position = (self.bot._pos[0] + 100, self.bot._pos[1] + 400)
            self.bot.pymunk_position_update(self.body.position)

    def cap_velocity(self, velocity):
        if velocity[0] <= -25: velocity[0] = -25
        if velocity[0] >= 25: velocity[0] = 25
        if velocity[1] <= -25: velocity[1] = -25
        if velocity[1] >= 25: velocity[1] = 25
        return velocity

    def get_dust_count(self):
        dust_count = 0
        for x in self.dust_grid:
            for y in x:
                if y == 1: dust_count = dust_count + 1
        return dust_count

    def get_fitness(self):
        return self.get_dust_count() - (5 * self.collision_counter)

    def draw(self):
        pygame.draw.circle(self.pygame_display, self.color, self.body.position, self.radius)

    def draw_sensors(self):
        sensors = self.bot._sensors
        for sensor in sensors:
            # Case 1: Using pymunk auto collision, not our own collision
            if self.pymunk_collision:
                start, end = sensor.get_sensor_position(self.bot._pymunk_position)
                pygame.draw.line(self.pygame_display, self.color, (start[0], start[1]), (end[0], end[1]), 5)
                if sensor._offset == 0:
                    pygame.draw.line(self.pygame_display, (255, 0, 0), (start[0], start[1]), (end[0], end[1]), 5)
            # Case 2: Using our own collision, not pymunk collison
            else:
                start, end = sensor.get_sensor_position(self.bot._pos)
                pygame.draw.line(self.pygame_display, self.color, (start[0] + 100, start[1] + 400),
                                 (end[0] + 100, end[1] + 400), 5)
                if sensor._offset == 0:
                    pygame.draw.line(self.pygame_display, (255, 0, 0), (start[0] + 100, start[1] + 400),
                                     (end[0] + 100, end[1] + 400), 5)

    # def to_pygame(self, value):
    #     return value + 300


class Pymunk_Obstacle:
    """ Class for a wall in pymunk, for pygame. Soon to be extended for more than just line segments.
    """

    def __init__(self, pygame_display, pymunk_space, radius, color, p):
        self.pygame_display = pygame_display
        self.pymunk_space = pymunk_space
        self.radius = radius
        self.color = color
        self.p1 = p[0]
        self.p2 = p[1]

        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.shape = pymunk.Segment(self.body, self.p1, self.p2, self.radius)
        self.shape.elasticity = 1
        pymunk_space.add(self.body, self.shape)

    def draw(self):
        pygame.draw.line(self.pygame_display, self.color, self.p1, self.p2, self.radius)
