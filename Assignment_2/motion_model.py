import numpy as np
from vpython import *
from sensor import distance_sensor, degrees_to_radians

# ---------- Motion Model ---------- #


class robot():
    """Class for the two-wheeled robot
    Assumes the radius of the tires to be equal to 1
    """
    def __init__(self, pos, distance_between_wheels=1, robot_body_radius=1, current_time=0, acceleration=1, num_sensors=8, sensor_measuring_distance=10) -> None:
        assert distance_between_wheels>0, 'Distance between wheels must be positive'
        self._start = pos
        self._pos = pos # position should be given in the form [x,y,theta] with theta given in radians not degrees
        self._time = current_time
        self._l = distance_between_wheels
        self._body_r = robot_body_radius
        self._vel_right = 0
        self._vel_left = 0
        self._rot_rate = 0 # corresponds to omega in slides
        self._rot_radius = 0 # corresponds to (uppercase) R in slides
        self._acc = acceleration
        self._num_sens = num_sensors
        self._sens_dist = sensor_measuring_distance
        self._sensors = []

        for i in range(num_sensors):
            offset = np.linspace(0, 360, self._num_sens, endpoint=False)[i]
            offset = degrees_to_radians(offset)
            self._sensors.append(distance_sensor(offset, self._body_r, self._sens_dist))

    @property
    def pos(self):
        return self._pos

    def get_pos_vpython(self) -> np.ndarray:
        """returns the position of the center of the robot as it is used in VPython

        Returns:
            np.ndarray: 3D-coordinates of center of robot, where z-coordinate is 0 to simulate 2D
        """           
        return vector(self._pos[0], self._pos[1], 0)

    @property
    def time(self):
        return self._time
        
    def timestep(self, time_elapsed=1):
        self.move_mouhknowsbest(time_elapsed)
        self.update_sensors()
        # self.move(time_elapsed)
        self._time += time_elapsed

    def stop(self, verbose=False):
        self._vel_right = 0
        self._vel_left = 0
        self.update_rot_rate()
        self.update_rot_radius()
        if verbose:
            print('Stoppping the robot. Set velocity of wheels to zero')

    def reset(self, verbose=False):
        self.stop(verbose=verbose)
        self._pos = self._start
        if verbose:
            print(f'Position of robot has been reset to the initial starting poisiton of {self._start}')

    @property
    def vel_right(self):
        return self._vel_right  

    def accel_right(self, verbose=False):
        self._vel_right = np.round(self._vel_right+self._acc, decimals=8)
        self.update_rot_rate()
        self.update_rot_radius()
        if verbose:
            print(f'Accelerating right: {self._vel_right}, rotation rate: {self._rot_rate}, rotation radius: {self._rot_radius}')        

    def decel_right(self, verbose=False):
        self._vel_right = np.round(self._vel_right-self._acc, decimals=8)
        self.update_rot_rate()
        self.update_rot_radius()
        if verbose:
            print(f'Decelerating right: {self._vel_right}, rotation rate: {self._rot_rate}, rotation radius: {self._rot_radius}')

    @property
    def vel_left(self):
        return self._vel_right

    def accel_left(self, verbose=False):
        self._vel_left = np.round(self._vel_left+self._acc, decimals=8)
        self.update_rot_rate()
        self.update_rot_radius()    
        if verbose:
            print(f'Accelerating left: {self._vel_left}, rotation rate: {self._rot_rate}, rotation radius: {self._rot_radius}')           

    def decel_left(self, verbose=False):
        self._vel_left = np.round(self._vel_left-self._acc, 8)
        self.update_rot_rate()
        self.update_rot_radius()
        if verbose:
            print(f'Decelerating left: {self._vel_left}, rotation rate: {self._rot_rate}, rotation radius: {self._rot_radius}')

    def accel_both(self, verbose=False):
        self._vel_right = np.round(self._vel_right+self._acc, decimals=8)
        self._vel_left = np.round(self._vel_left+self._acc, decimals=8)
        self.update_rot_rate()
        self.update_rot_radius()
        if verbose:
            print(f'Accelerating right: {self._vel_right}, rotation rate: {self._rot_rate}, rotation radius: {self._rot_radius}')
            print(f'Accelerating left: {self._vel_left}, rotation rate: {self._rot_rate}, rotation radius: {self._rot_radius}')

    def decel_both(self, verbose=False):
        self._vel_right = np.round(self._vel_right-self._acc, decimals=8)
        self._vel_left = np.round(self._vel_left-self._acc, decimals=8)
        self.update_rot_rate()
        self.update_rot_radius()
        if verbose:
            print(f'Decelerating left: {self._vel_left}, rotation rate: {self._rot_rate}, rotation radius: {self._rot_radius}')
            print(f'Decelerating right: {self._vel_right}, rotation rate: {self._rot_rate}, rotation radius: {self._rot_radius}')

    @property
    def rot_rate(self):
        return self._rot_rate

    def update_rot_rate(self):
        self._rot_rate = np.round(((self._vel_right - self._vel_left)/self._l), decimals=8)

    @property
    def rot_radius(self):
        return self._rot_radius

    def update_rot_radius(self, verbose=False):
        if self._vel_right == self._vel_left:
            self._rot_radius = np.Inf
        elif self._vel_right == 0:
            self._rot_radius = -1*np.round(self._l/2, decimals=8)
        elif self._vel_left==0:
            self._rot_radius = np.round(self._l/2, decimals=8)
        else:
            self._rot_radius = np.round((self._vel_right - self._vel_left)/self._l, decimals=8)

        if verbose:
            print(f'Updated R: {self.rot_radius}')

    def update_sensors(self):
        for sensor in self._sensors:
            sensor.update(self._pos)

    def get_rays_vpython(self):
        rays = []
        for sensor in self._sensors:
            rays.append(sensor.get_ray_vpython(self._pos))
        #print(f'rays are the following: {rays}')
        return rays

    def get_distance_to_walls(self):
        distances = []
        for sensor in self._sensors:
            distances.append(sensor.dist_to_wall)
        return distances


    def move_mouhknowsbest(self, time_elapsed):
        # new attempt at the move function, as the old one has issues
        # https://www.youtube.com/watch?v=aE7RQNhwnPQ
        # define radius of the wheel to be 1:
        vel_forward = np.round((self._vel_right+self._vel_left)/2, decimals=8)
        deriv_x = 0.5*(vel_forward)*np.cos(self._pos[2])
        deriv_y = 0.5*(vel_forward)*np.sin(self._pos[2])
        deriv_theta = (1/self._l)*(self._vel_right-self._vel_left)
        self._pos = self._pos + time_elapsed*np.array([deriv_x, deriv_y, deriv_theta])


    # old move method that was not working properly

    # def move(self, time_elapsed, verbose=False) -> None:
    #     """Method that performs moving the robot one time-step forward.
    #     The time step is defined to be delta*t = time_elapsed to calculate the rotation matrix.
    #     """
    #     if (self._vel_right == 0) and (self._vel_left==0):
    #         return
    #     if self._rot_radius == np.Inf:
    #         print(f'old position: {self._pos}')
    #         print(f'Norm of directional vector: {np.linalg.norm(self._pos[:-1])}')
    #         vel_forward = np.round(self._vel_right, decimals=8)
    #         move_x = np.round(np.cos(self.pos[2])*np.abs(vel_forward), decimals=8)
    #         move_y = np.round(np.sin(self.pos[2])*np.abs(vel_forward), decimals=8)
    #         self._pos = self._pos + [move_x, move_y, 0]
    #         print(f'movement along x-axis: {move_x}')
    #         print(f'movement along y-axis: {move_y}')
    #         print(f'new position: {self._pos}')            
    #         return

    #     dt = time_elapsed
    #     pos_icc = np.array([
    #         self._pos[0]-self._rot_radius*np.sin(self._pos[2]), 
    #         self._pos[1]+self._rot_radius*np.cos(self._pos[2])])
    #     rot_matrix = np.array([
    #         [np.cos(self._rot_rate*dt), -np.sin(self._rot_rate*dt), 0],
    #         [np.sin(self._rot_rate*dt), np.cos(self._rot_rate*dt), 0],
    #         [0, 0, 1]
    #     ], dtype='float')
    #     multiplier = np.array([self._pos[0]-pos_icc[0], self._pos[1]-pos_icc[1], self._pos[2]], dtype='float')
    #     self._pos = np.dot(rot_matrix, np.transpose(multiplier))+np.append(pos_icc, self._rot_rate*dt)
    #     if verbose:
    #         print(f'position of ICC: {pos_icc}')
    #         print(f'shape of ICC: {pos_icc.shape}')
    #         print()
    #         print(f'rotation matrix: {rot_matrix}')
    #         print(f'shape of matrix: {rot_matrix.shape}')
    #         print()
    #         print(f'multiplier: {multiplier}')
    #         print(f'shape of multiplier: {multiplier.shape}')
    #         print()
    #         print(f'new position: {self._pos}')
    #         print()
