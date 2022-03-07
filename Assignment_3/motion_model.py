import math

import numpy as np
from vpython import *
from sensor import *

# ---------- Motion Model ---------- #
# ---- UPDATED FOR ASSIGNMENT 3 ---- #

class Robot():
    """Class for the two-wheeled robot
    Assumes the radius of the tires to be equal to 1
    """
    def __init__(self,
                 pos,
                 distance_between_wheels=1,
                 robot_body_radius=1,
                 current_time=0,
                 acceleration=1,
                 num_sensors=8,
                 sensor_measuring_distance=10,
                 obstacle_edges=None,
                 wall_distance=20,
                 collision_check=False,
                 slide_collision_check=True,
                 pymunk_offset=[0,0,0]) -> None:
        assert distance_between_wheels>0, 'Distance between wheels must be positive'
        self._start = pos
        self._pos = pos # position should be given in the form [x,y,theta] with theta given in radians not degrees
        self._time = current_time
        self._time_step_size = 1/30 # careful!! Hard coded for now
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
        self._obstacle_edges = obstacle_edges
        self._collision_sensor = DistanceSensor(0, self._body_r, np.round((self._vel_right+self._vel_left)/2, decimals=8), self._obstacle_edges)
        self._forward_sensor = None
        self._wall_distance = wall_distance
        self._collision_check = collision_check
        self._slide_collision_check = slide_collision_check
        self._theta = self._pos[2]                          # used as angle for pymunk
        self._pymunk_position = self._pos + pymunk_offset   # keeps track of pymunk position

        for i in range(num_sensors):
            offset = np.linspace(0, 360, self._num_sens, endpoint=False)[i]
            offset = degrees_to_radians(offset)
            sensor = DistanceSensor(offset, self._body_r, self._sens_dist, self._obstacle_edges)
            if offset == 0:
                self._forward_sensor = sensor
            self._sensors.append(sensor)

    # -------------------------------------------------------------
    # ---------------------- 'GET' FUNCTIONS ----------------------
    # -------------------------------------------------------------

    @property
    def pos(self):
        return self._pos

    @property
    def time(self):
        return self._time

    @property
    def vel_right(self):
        return self._vel_right

    @property
    def vel_left(self):
        return self._vel_left

    @property
    def rot_rate(self):
        return self._rot_rate

    @property
    def rot_radius(self):
        return self._rot_radius

    def get_pos_vpython(self) -> np.ndarray:
        """"Returns:
            np.ndarray: 3D-coordinates of center of robot, where z-coordinate is 0 to simulate 2D
        """           
        return vector(self._pos[0], self._pos[1], 0)

    def get_pos_pygame(self):
        return self._pos

    def get_rays_vpython(self):
        rays = []
        for sensor in self._sensors:
            rays.append(sensor.get_ray_vpython(self._pos))
        #print(f'rays are the following: {rays}')
        return rays

    def get_distance_to_walls(self):
        distances = []
        for sensor in self._sensors:
            distances.append(sensor._dist_to_wall)
        return distances

    # ------------------------------------------------------
    # --------- MANIPULATING & UPDATING VELOCITY -----------
    # ------------------------------------------------------

    def stop(self, verbose=False):
        """ Function that gets called on pressing a key to stop the robot movement
        """
        self._vel_right = 0
        self._vel_left = 0
        self.update_rot_rate()
        self.update_rot_radius()
        if verbose:
            print('Stopping the robot. Set velocity of wheels to zero')

    def reset(self, verbose=False):
        """ Function that gets called on pressing a key to reset the robot position
        """
        self.stop(verbose=verbose)
        self._pos = self._start
        if verbose:
            print(f'Position of robot has been reset to the initial starting position of {self._start}')

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

    def update_rot_rate(self):
        self._rot_rate = np.round(((self._vel_right - self._vel_left)/self._l), decimals=8)

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

    # ------------------------------------------------------
    # ------------- CALCULATING NEW POSITION ---------------
    # ------------------------------------------------------

    def timestep(self, time_elapsed=1):
        """ Main method that gets called for updating calculating and updating position of robot
        """
        # Following commented code is for 'advanced' collision detection:
        # self._prev_time = self._time
        # if self.check_for_immediate_collision() is not None:
        # pass
        # else:
        self.move_mouhknowsbest(time_elapsed)
        self.update_sensors()
        # self.move(time_elapsed)
        self._time += time_elapsed

    def move_mouhknowsbest(self, time_elapsed):
        # new attempt at the move function, as the old one has issues
        # https://www.youtube.com/watch?v=aE7RQNhwnPQ
        # define radius of the wheel to be 1:
        vel_forward = np.round((self._vel_right+self._vel_left)/2, decimals=8)
        deriv_x = (vel_forward)*np.cos(self._pos[2]) # not sure why it is 0.5 anymore, maybe this depends on the distance between wheels or the radius of the robot or sth. Rewatch the video for that.
        deriv_y = (vel_forward)*np.sin(self._pos[2])
        deriv_theta = 0.1*(1/self._l)*(self._vel_right-self._vel_left)

        # Case 1: New collision type (doesn't work yet)
        if self._slide_collision_check:
            intersec_coords = self._forward_sensor.intersection_coordinates(self._pos)
            if intersec_coords is not None:
                deriv_x, deriv_y = self.collision_slide(intersec_coords, deriv_x, deriv_y)
                print(intersec_coords)
            self._pos = self._pos + time_elapsed*np.array([deriv_x, deriv_y, deriv_theta])
        # Case 2: Previous collision type (Only works for 4 square hardcoded walls)
        elif self._collision_check:
            # test collision movement
            temp_pos = self._pos + time_elapsed*np.array([deriv_x, deriv_y, deriv_theta])
            self._pos = np.append(np.clip(temp_pos[:-1], a_min=-1*self._wall_distance+self._body_r, a_max=self._wall_distance-self._body_r), temp_pos[2])  # careful, boundaries are hard coded for now!!!
        # Case 3: No collision check
        else:
            # end test collision movement
            self._pos = self._pos + time_elapsed*np.array([deriv_x, deriv_y, deriv_theta])

    def update_sensors(self):
        for sensor in self._sensors:
            sensor.update(self._pos)
        self._collision_sensor.update(self._pos)
        self._collision_sensor._sens_dist = np.round(0.5*(self._vel_left+self._vel_right)*self._time_step_size, decimals=8)

    def get_xy_velocity(self, dt=1 / 30):

        vel_forward = np.round((self._vel_right + self._vel_left) / 2, decimals=8)
        vel_x = dt * (np.round(vel_forward * np.cos(self._theta), decimals=8))
        vel_y = dt * (np.round(vel_forward * np.sin(self._theta), decimals=8))
        self._theta += dt * 0.1 * (1 / self._l) * (self._vel_right - self._vel_left)

        return [vel_x, vel_y]

    def get_vel_ann(self, vel_right, vel_left, dt=1 / 30):
        self._vel_right = vel_right
        self._vel_left = vel_left
        self.update_rot_rate()
        self.update_rot_radius()

        vel_forward = np.round((self._vel_right + self._vel_left) / 2, decimals=8)
        vel_x = dt * (np.round(vel_forward * np.cos(self._theta), decimals=8))
        vel_y = dt * (np.round(vel_forward * np.sin(self._theta), decimals=8))
        self._theta += dt * 0.1 * (1 / self._l) * (self._vel_right - self._vel_left)
        return [vel_x, vel_y]

    def pymunk_position_update(self, coords):
        self._pymunk_position[0] = coords[0]
        self._pymunk_position[1] = coords[1]
        self._pymunk_position[2] = self._theta
        # theta value gets updated in get_xy_velocity

    # ----------------------------------------------------------
    # ------------- ADVANCED COLLISION DETECTION ---------------
    # ----------------------------------------------------------

    def collision_slide(self, intersec_coords, x_vel, y_vel):
        nx = intersec_coords[0]
        ny = intersec_coords[1]
        magnitude = math.sqrt((nx*nx)+(ny*ny))
        if magnitude == 0: magnitude = 1
        nx = nx / magnitude
        ny = ny / magnitude
        dotprod = (nx * x_vel) + (ny * y_vel)
        new_x_vel = x_vel - (dotprod * nx)
        new_y_vel = y_vel - (dotprod * ny)
        return [new_x_vel, new_y_vel]

    '''
    def check_for_immediate_collision(self):
        return self._collision_sensor._dist_to_wall

    def collision_movement(self):
        pos_collision = self._collision_sensor.intersection_coordinates()
        vel_forward = (self._vel_left+self._vel_right)/2

        # implement that the robot collides with the outside of its shape and not with its center
        fraction_to_wall = np.round(np.linalg.norm(pos_collision-np.array(self._pos[0], self._pos[1]))/vel_forward, decimals=8)
        self.move_alongside_wall(frac_vel_remaining=1-fraction_to_wall)  # use the remaining velocity vector
        # get fraction of velocity vector that we still need to travel until we hit the wall
        pass

    def move_alongside_wall(self, frac_vel_remaining):
        x_component = 0.5*(self._vel_right+self._vel_right) * np.cos(self._pos[2])
        y_component = 0.5*(self._vel_right+self._vel_right) * np.sin(self._pos[2])
        pass


    def test(self, boundary=20):
        vel_forward = np.round((self._vel_right+self._vel_left)/2, decimals=8)
        deriv_x = 0.5*(vel_forward)*np.cos(self._pos[2])  # not sure why it is 0.5 anymore, maybe this depends on the distance between wheels or the radius of the robot or sth. Rewatch the video for that.
        deriv_y = 0.5*(vel_forward)*np.sin(self._pos[2])
        intermediate_pos = self._pos + time_elapsed*np.array([deriv_x, deriv_y, 0])
        np.clip(intermediate_pos[:-1], a_min=-1*boundary, a_max=boundary)
    '''


    # ----------------------------------------------
    # old move method that was not working properly
    # ----------------------------------------------


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
