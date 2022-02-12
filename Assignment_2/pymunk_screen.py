import pymunk
import pygame
import motion_model
from motion_model import robot

# --- Initialise pygame and make a display appear
pygame.init()
display = pygame.display.set_mode((600, 600))

# --- Set global values
# Add pymunk.Space() object, which holds our pymunk physics world
space = pymunk.Space()
# Create clock object for setting the frame-rate for display update
clock = pygame.time.Clock()
FPS = 50

left = top = 10
right = bottom = 590
middle_x = middle_y = 300
bot_radius = 15
wall_size = 10

white = (255, 255, 255)
black = (0, 0, 0)

display.fill(white)


class Bot():
    """" Class for a moveable robot (circle) in PyMunk using the motion_model Robot class
    """
    def __init__(self):
        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.robot = robot([middle_x, middle_y, 0], bot_radius * 2)
        self.body.position = self.robot._pos[0], self.robot._pos[1]
        self.body.velocity = self.robot._vel_left, self.robot._vel_right
        self.shape = pymunk.Circle(self.body, bot_radius)
        self.shape.density = 1
        self.shape.elasticity = 0;  # no 'bounce'
        space.add(self.body, self.shape)
        self.shape.collision_type = 1

    def draw(self):
        x, y = self.body.position
        pygame.draw.circle(display, black, (int(x), int(y)), bot_radius)

    def move(self, key):
        if key == pygame.K_w: self.robot.accel_left()
        if key == pygame.K_s: self.robot.decel_left()
        if key == pygame.K_o: self.robot.accel_right()
        if key == pygame.K_l: self.robot.decel_right()
        if key == pygame.K_y: self.robot.accel_both()
        if key == pygame.K_h: self.robot.decel_both()
        if key == pygame.K_x: self.robot.reset()
        self.body.position = self.robot._pos[0], self.robot._pos[1]
        self.body.velocity = self.robot._vel_left, self.robot._vel_right


class Wall():
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

# --- the method where the simulation happens
def simulation():
    # set events that can happen while in the display window
    bot = Bot()
    wall_left = Wall([left, top], [left, bottom], 2)
    wall_right = Wall([right, top], [right, bottom], 2)
    wall_top = Wall([left, top], [right, top], 2)
    wall_bottom = Wall([left, bottom], [right, bottom])
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # pygame.QUIT gets called when we press the 'x' button on the window
                return  # exit out of the function call to end the display
            if event.type == pygame.KEYDOWN:
                bot.move(event.key)

        bot.robot.timestep(1/FPS)

        display.fill(white)
        bot.draw()
        wall_left.draw()
        wall_right.draw()
        wall_top.draw()
        wall_bottom.draw()
        # update the display - using clock object to set frame-rate
        pygame.display.update()
        clock.tick(FPS)
        # pass some time in the simulation
        space.step(1 / FPS)  # basis: correlate with FPS. Low val = more accurate simulation, but slower program


# call the function simulation to keep the display running until we quit
simulation()

# End the pygame display
pygame.quit()
