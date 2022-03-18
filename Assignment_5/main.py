import pygame
from pygame_robot import *

pygame.init()
# Create clock object for setting the frame-rate for display update
clock = pygame.time.Clock()
FPS = 50

display_width = 600
display_height = 800
pygame_display = pygame.display.set_mode((display_width, display_height))

bot_radius = 20
white = (255, 255, 255)
black = (0,0,0)
pygame_display.fill(white)

robot = Robot(pygame_display=pygame_display,
              radius=bot_radius,
              color=black,
              position=(50,50,0),
              distance_between_wheels=bot_radius*2,
              acceleration=10,
              angular_acceleration=1,
              FPS=FPS,
              current_time=0)

def simulation(display, bot, FPS=50):
    while True:
        key = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # pygame.QUIT gets called when we press the 'x' button on the window
                return  # exit out of the function call to end the display
            if event.type == pygame.KEYDOWN:
                key = event.key
        bot.move(key=key, time_elapsed=1/FPS)
        display.fill(white)
        bot.draw()

        # update the display - using clock object to set frame-rate
        pygame.display.update()
        clock.tick(FPS)

# call the function simulation to keep the display running until we quit
simulation(pygame_display, robot, FPS)

# End the pygame display
pygame.quit()
