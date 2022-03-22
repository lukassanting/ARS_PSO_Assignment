import pygame
import numpy as np
import math
from kalman_filter import Kalman_filter, KF_no_rot_rate, initial_covariance_matrix

class Robot:
    """
    Class for a differential drive robot in a pygame simulation. Includes
        - a function (with supporting functions) for calculating movement, given keyboard inputs
        - a function to draw the robot to a provided pygame display
    """
    def __init__(self,
                 pygame_display: pygame.display,
                 radius: int,
                 color: tuple,
                 position: tuple,
                 distance_between_wheels: int,
                 acceleration: int = 10,
                 angular_acceleration: int = 5,
                 FPS: float = 1 / 50,
                 current_time: int = 0,
                 noisy_velocity_scale: float = 1):
        self._acc = acceleration
        self._angular_acc = angular_acceleration
        self._color = color
        self._display = pygame_display
        self._l = distance_between_wheels
        self._pos = position
        self._radius = radius
        self._rot_rate = 0  # corresponds to omega in slides
        self._rot_radius = 0  # corresponds to (uppercase) R in slides
        self._start = position
        self._time = current_time
        self._time_step_size = 1 / FPS
        self._vel_right = 0
        self._vel_left = 0
        self._belief_vel = 0
        self._positions = [(position[0], position[1])]
        self._belief_positions = [(position[0], position[1])]
        self._belief_angle = [position[2]]
        self._belief_covariance_matrix = [initial_covariance_matrix()]
        self._change_in_theta = 0
        self._ellipses = []
        self._update_lines = []
        self._noisy_velocity_scale = noisy_velocity_scale

    # ---------------------------------------------------------------------------------
    # --------------------------- DRAWING FUNCTIONS -----------------------------------
    # ---------------------------------------------------------------------------------

    def draw(self):
        pygame.draw.circle(self._display, self._color, (self._pos[0], self._pos[1]), self._radius)

    def draw_track(self, color: pygame.Color):
        pygame.draw.lines(self._display, color, False, self._positions)

    def draw_updated_lines(self):
        for (old, new) in self._update_lines:
            pygame.draw.line(self._display, (0, 200, 15), old, new)

    def draw_active_beacon(self, beacon_pos):
        pygame.draw.line(self._display, (0, 0, 255), (self._pos[0], self._pos[1]), beacon_pos)

    # ---------------------------------------------------------------------------------
    # Code for draw_dashed_line() and draw_dashed_lines() copied from user Rabbid76 on StackOverflow:
    # https://stackoverflow.com/questions/66943011/how-to-draw-a-dashed-curved-line-with-pygame?noredirect=1&lq=1
    def draw_dashed_line(self, p1, p2, prev_line_len, dash_length=8):
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        if dx == 0 and dy == 0:
            return
        dist = math.hypot(dx, dy)
        dx /= dist
        dy /= dist

        step = dash_length*2
        start=(int(prev_line_len) // step) * step
        end=(int(prev_line_len + dist)// step+1) * step
        for i in range(start, end, dash_length*2):
            s = max(0, start-prev_line_len)
            e = min(start-prev_line_len+dash_length, dist)
            if s < e:
                ps = p1[0] + dx * s, p1[1] + dy * s
                pe = p1[0] + dx * e, p1[1] + dy * e
                pygame.draw.line(self._display, self._color, pe, ps)

    def draw_dashed_lines(self, points, dash_length=8):
        line_len=0
        for i in range(1, len(points)):
            p1, p2 = points[i-1], points[i]
            dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            self.draw_dashed_line(p1, p2, line_len, dash_length)
            line_len += dist
    # ---------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------
    # -------------------------- MOVEMENT FUNCTIONS -----------------------------------
    # ---------------------------------------------------------------------------------
    def move(self, key, time_elapsed, noisy_velocity=False, noisy_angles=False, verbose=False):
        self._change_in_theta = 0
        if key is not None:
            if key == pygame.K_w: self.accelerate(add_noise=noisy_velocity, verbose=verbose)
            if key == pygame.K_s: self.decelerate(add_noise=noisy_velocity, verbose=verbose)
            if key == pygame.K_a: self.increase_angle(add_noise=noisy_angles, verbose=verbose)
            if key == pygame.K_d: self.decrease_angle(add_noise=noisy_angles, verbose=verbose)
            if key == pygame.K_x: self.stop(verbose=verbose)
            if key == pygame.K_r: self.reset(verbose=verbose)

        vel_forward = np.round((self._vel_right + self._vel_left) / 2, decimals=8)
        deriv_x = (vel_forward) * np.cos(self._pos[2])
        deriv_y = (vel_forward) * np.sin(self._pos[2])
        deriv_theta = 0.1 * (1 / self._l) * (self._vel_right - self._vel_left)
        self._pos = self._pos + time_elapsed * np.array([deriv_x, deriv_y, deriv_theta])
        self._positions.append((self._pos[0], self._pos[1]))
        self._time += time_elapsed

    def accelerate(self, add_noise=False, verbose=False):
        if add_noise:
            self._vel_right = np.round(self._vel_right + self._acc + np.random.normal(scale=self._noisy_velocity_scale), decimals=8)
            self._vel_left = np.round(self._vel_left + self._acc + np.random.normal(scale=self._noisy_velocity_scale), decimals=8)
        else:
            self._vel_right = np.round(self._vel_right + self._acc, decimals=8)
            self._vel_left = np.round(self._vel_left + self._acc, decimals=8)

        self._belief_vel = np.round(self._belief_vel + self._acc, decimals=8)
        self.update_rotate_rate()
        self.update_rotate_radius(verbose)
        if verbose:
            print(
                f'Accelerating right: {self._vel_right}, Accelerating left: {self._vel_left}, rotation rate: {self._rot_rate}, rotation radius: {self._rot_radius}')

    def decelerate(self, add_noise=False, verbose=False):
        if add_noise:
            self._vel_right = np.round(self._vel_right - self._acc + np.random.normal(), decimals=8)
            self._vel_left = np.round(self._vel_left - self._acc + np.random.normal(), decimals=8)
        else:
            self._vel_right = np.round(self._vel_right - self._acc, decimals=8)
            self._vel_left = np.round(self._vel_left - self._acc, decimals=8)

        self._belief_vel = np.round(self._belief_vel - self._acc, decimals=8)
        self.update_rotate_rate()
        self.update_rotate_radius(verbose)
        if verbose:
            print(
                f'Decelerating right: {self._vel_right}, Decelerating left: {self._vel_left}, rotation rate: {self._rot_rate}, rotation radius: {self._rot_radius}')

    def increase_angle(self, add_noise=False, verbose=False):
        if add_noise:
            self._pos[2] += self._angular_acc+np.random.normal(scale=0.01)
        else:
            self._pos[2] += self._angular_acc
        self._change_in_theta += self._angular_acc
        if verbose:
            print(f'Increasing angular position: {self._pos[2]}')

    def decrease_angle(self, add_noise=False, verbose=False):
        if add_noise:
            self._pos[2] -= self._angular_acc+np.random.normal(scale=0.01)
        else:
            self._pos[2] -= self._angular_acc
        self._change_in_theta -= self._angular_acc
        if verbose:
            print(f'Decreasing angular position: {self._pos[2]}')

    def stop(self, verbose=False):
        self._belief_vel = 0
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
        self._rot_rate = np.round(((self._vel_right - self._vel_left) / self._l), decimals=8)

    def update_rotate_radius(self, verbose=False):
        if self._vel_right == self._vel_left:
            self._rot_radius = np.Inf
        elif self._vel_right == 0:
            self._rot_radius = -1 * np.round(self._l / 2, decimals=8)
        elif self._vel_left == 0:
            self._rot_radius = np.round(self._l / 2, decimals=8)
        else:
            self._rot_radius = np.round((self._vel_right - self._vel_left) / self._l, decimals=8)

        if verbose:
            print(f'Updated R: {self._rot_radius}')

    # ---------------------------------------------------------------------------------
    # -------------------------- Kalman Filter functions -----------------------------------
    # ---------------------------------------------------------------------------------

    def update_beliefs(self, trilateration_pos, delta_t:float):
        if trilateration_pos is not None:
            if isinstance(trilateration_pos, tuple):
                trilateration_pos = np.asarray(trilateration_pos)
            trilateration_pos = trilateration_pos.reshape((3,1))
        prior_belief = np.append(np.asarray(self._belief_positions[-1]), self._belief_angle[-1]).reshape((3,1))
        print(f'rotation rate is {self._rot_rate}')
        u = np.array([self._belief_vel, self._rot_rate]).reshape((2,1))
        pos, cov_matrix = Kalman_filter(
                                    mean_t_minus_1=prior_belief,
                                    cov_matrix_t_minus_1=self._belief_covariance_matrix[-1],
                                    u_t=u,
                                    z_t=trilateration_pos,
                                    delta_t=delta_t
                                    )
        self._belief_positions.append(tuple(pos.flatten()[:2]))
        self._belief_angle.append(pos.flatten()[2])
        self._belief_covariance_matrix.append(cov_matrix)


    def update_beliefs_no_rot_rate(self, trilateration_pos, delta_t:float):
        if trilateration_pos is not None:
            if isinstance(trilateration_pos, tuple):
                trilateration_pos = np.asarray(trilateration_pos)
            if trilateration_pos.shape != (3,1):
                trilateration_pos = trilateration_pos.reshape((3,1))
        prior_belief = np.append(np.asarray(self._belief_positions[-1]), self._belief_angle[-1]).reshape((3,1))
        u = np.array([self._belief_vel, self._change_in_theta]).reshape((2,1))
        pos, cov_matrix, update_line = KF_no_rot_rate(
                                    mean_t_minus_1=prior_belief,
                                    cov_matrix_t_minus_1=self._belief_covariance_matrix[-1],
                                    u_t=u,
                                    z_t=trilateration_pos,
                                    delta_t=delta_t
                                    )
        if update_line:
            self._update_lines.append([self._belief_positions[-1], pos.flatten()[:2]])
        self._belief_positions.append(tuple(pos.flatten()[:2]))
        self._belief_angle.append(pos.flatten()[2])
        self._belief_covariance_matrix.append(cov_matrix)

    def generate_ellipse(self):
        width = 100*np.diagonal(self._belief_covariance_matrix[-1])[0] # scaled by 1000 to make it better visible
        height = 100*np.diagonal(self._belief_covariance_matrix[-1])[1] # scaled by 1000 to make it better visible
        left = self._belief_positions[-1][0]-width/2 # subtracting width/2 to center the ellipse at the center of the robot
        top = self._belief_positions[-1][1]-height/2 # subtracting height/2 to center the ellipse at the center of the robot
        ellipse_boundaries = (left, top, width, height)
        self._ellipses.append(pygame.Rect(ellipse_boundaries))

    def draw_ellipses(self):
        for ellipse in self._ellipses:
            pygame.draw.ellipse(surface=self._display, color=(169,169,169), rect=ellipse, width = 1)