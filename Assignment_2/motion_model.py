from audioop import mul
from turtle import shape
import numpy as np
from sklearn import multiclass
from sympy import multiplicity
from vpython import *

# ---------- Motion Model ---------- #

class robot():
    """Class for the two-wheeled robot
    """
    def __init__(self, pos, distance_between_wheels, current_time=0) -> None:
        assert distance_between_wheels>0, 'Distance between wheels must be positive'
        self._pos = pos # position should be given in the form [x,y,theta] with theta given in radians not degrees
        self._time = current_time
        self._l = distance_between_wheels
        self._vel_right = 0
        self._vel_left = 0
        self._rot_rate = 0 # corresponds to omega in slides
        self._rot_radius = 0 # corresponds to (uppercase) R in slides

    @property
    def pos(self):
        return self._pos

    def get_pos_vpython(self) -> np.ndarray:
        """returns the position of the center of the robot as it is used in VPython

        Returns:
            np.ndarray: 3D-coordinates of center of robot, where y-coordinate is 0 to simulate 2D
        """           
        return vector(self._pos[0], self._pos[1], 0)

    @property
    def time(self):
        return self._time
        
    def timestep(self):
        self._time += 1
        self.move()

    @property
    def vel_right(self):
        return self._vel_right

    def accel_left(self):
        self._vel_left += 1
        self.update_rot_rate()
        self.update_rot_radius()           

    def decel_left(self):
        self._vel_left -= 1
        self.update_rot_rate()
        self.update_rot_radius()

    @property
    def vel_left(self):
        return self._vel_right

    def accel_right(self):
        self._vel_right += 1
        self.update_rot_rate()
        self.update_rot_radius()

    def decel_right(self):
        self._vel_right -= 1
        self.update_rot_rate()
        self.update_rot_radius()

    @property
    def rot_rate(self):
        return self._rot_rate

    def update_rot_rate(self):
        self._rot_rate = (self._vel_right - self._vel_left)/self._l

    @property
    def rot_radius(self):
        return self._rot_radius

    def update_rot_radius(self):
        if (self._vel_right == 0) & (self._vel_left==0):
            self._rot_radius = np.Inf
        elif self._vel_right == self._vel_left:
            self._rot_radius = self._l/2
        else:
            self._rot_radius = (self._vel_right - self._vel_left)/self._l

    def move(self):
        """Method that performs moving the robot one time-step forward.
        The time step is defined to be delta*t = 1 to calculate the rotation matrix.
        """
        pos_icc = np.array([
            self._pos[0]-self._rot_radius*np.sin(self._pos[2]), 
            self._pos[1]+self._rot_radius*np.cos(self._pos[2])])
        print(f'position of ICC: {pos_icc}')
        print(f'shape of ICC: {pos_icc.shape}')
        print()
        rot_matrix = np.array([
            [np.cos(self._rot_rate), -np.sin(self._rot_rate), 0],
            [np.sin(self._rot_rate), np.cos(self._rot_rate), 0],
            [0, 0, 1]
        ], dtype='float')
        print(f'rotation matrix: {rot_matrix}')
        print(f'shape of matrix: {rot_matrix.shape}')
        print()
        multiplier = np.array([self._pos[0]-pos_icc[0], self._pos[1]-pos_icc[1], self._pos[2]], dtype='float')
        print(f'multiplier: {multiplier}')
        print(f'shape of multiplier: {multiplier.shape}')
        print()
        self._pos = np.dot(rot_matrix, np.transpose(multiplier))+np.append(pos_icc, self._rot_rate)
        print(f'new position: {self._pos}')
        print()
