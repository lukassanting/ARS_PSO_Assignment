import pygame
import numpy as np

class Robot:
    def __init__(self,
                 pygame_display,
                 radius,
                 color,
                 position,
                 distance_between_wheels,
                 acceleration=10,
                 angular_acceleration=5,
                 FPS=1/50,
                 current_time=0):
        self._acc = acceleration
        self._angular_acc = angular_acceleration
        self._color = color
        self._display = pygame_display
        self._l = distance_between_wheels
        self._pos = position
        self._radius = radius
        self._rot_rate = 0 # corresponds to omega in slides
        self._rot_radius = 0 # corresponds to (uppercase) R in slides
        self._start = position
        self._time = current_time
        self._time_step_size = 1/FPS
        self._vel_right = 0
        self._vel_left = 0

    def draw(self):
        pygame.draw.circle(self._display, self._color, (self._pos[0], self._pos[1]), self._radius)

    def move(self, key, time_elapsed, verbose=False):
        if key is not None:
            if key == pygame.K_w: self.accelerate(verbose)
            if key == pygame.K_s: self.decelerate(verbose)
            if key == pygame.K_a: self.increase_angle(verbose)
            if key == pygame.K_d: self.decrease_angle(verbose)
            if key == pygame.K_x: self.stop(verbose)
            if key == pygame.K_r: self.reset(verbose)

        vel_forward = np.round((self._vel_right+self._vel_left)/2, decimals=8)
        deriv_x = (vel_forward)*np.cos(self._pos[2])
        deriv_y = (vel_forward)*np.sin(self._pos[2])
        deriv_theta = 0.1*(1/self._l)*(self._vel_right-self._vel_left)
        self._pos = self._pos + time_elapsed*np.array([deriv_x, deriv_y, deriv_theta])

        self._time += time_elapsed

    def accelerate(self, verbose=False):
        self._vel_right = np.round(self._vel_right+self._acc, decimals=8)
        self._vel_left = np.round(self._vel_left+self._acc, decimals=8)
        self.update_rotate_rate()
        self.update_rotate_radius(verbose)
        if verbose:
            print(f'Accelerating right: {self._vel_right}, Accelerating left: {self._vel_left}, rotation rate: {self._rot_rate}, rotation radius: {self._rot_radius}')

    def decelerate(self, verbose=False):
        self._vel_right = np.round(self._vel_right-self._acc, decimals=8)
        self._vel_left = np.round(self._vel_left-self._acc, decimals=8)
        self.update_rotate_rate()
        self.update_rotate_radius(verbose)
        if verbose:
            print(f'Decelerating right: {self._vel_right}, Decelerating left: {self._vel_left}, rotation rate: {self._rot_rate}, rotation radius: {self._rot_radius}')

    def increase_angle(self, verbose=False):
        self._pos[2] += self._angular_acc
        if verbose:
            print(f'Increasing angular position: {self._pos[2]}')

    def decrease_angle(self, verbose=False):
        self._pos[2] -= self._angular_acc
        if verbose:
            print(f'Decreasing angular position: {self._pos[2]}')

    def stop(self, verbose=False):
        self._vel_right = 0
        self._vel_left = 0
        self.update_rotate_rate()
        self.update_rotate_radius()
        if verbose:
            print('Stopping the robot. Set velocity of wheels to zero')

    def reset(self, verbose=False):
        self.stop()
        self._pos = self._start
        if verbose:
            print(f'Position of robot has been reset to the initial starting position of {self._start}')

    def update_rotate_rate(self):
        self._rot_rate = np.round(((self._vel_right - self._vel_left)/self._l), decimals=8)

    def update_rotate_radius(self, verbose=False):
        if self._vel_right == self._vel_left:
            self._rot_radius = np.Inf
        elif self._vel_right == 0:
            self._rot_radius = -1*np.round(self._l/2, decimals=8)
        elif self._vel_left==0:
            self._rot_radius = np.round(self._l/2, decimals=8)
        else:
            self._rot_radius = np.round((self._vel_right - self._vel_left)/self._l, decimals=8)

        if verbose:
            print(f'Updated R: {self._rot_radius}')