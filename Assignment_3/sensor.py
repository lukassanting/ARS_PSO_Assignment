import numpy as np
from shapely.geometry import LineString
from vpython import *

# ------------- Sensors ------------ #
# ---- UPDATED FOR ASSIGNMENT 3 ---- #

def degrees_to_radians(angle):
    return angle*(np.pi/180)


class DistanceSensor():
    def __init__(self, offset, radius_robot, sensor_measuring_distance, obstacle_edges) -> None:
        # offset is the angle (counter-clockwise) the sensor faces away from the direction of the robot
        # offset should be given in radians
        # "wall" parameters should be a list of two tuples, indicating the end-points of the wall
        self._offset = offset   # direction in which the sensor is pointing
        self._sens_dist = sensor_measuring_distance
        self._radius_rob = radius_robot
        self._dist_to_wall = None
        if obstacle_edges:
            self._obstacle_edges = self.get_obstacle_edges(obstacle_edges)

    # -------------------------------------------------------------
    # ---------------------- 'GET' FUNCTIONS ----------------------
    # -------------------------------------------------------------

    def get_obstacle_edges(self, coords):
        edges = []
        for coord in coords:
            # coords are expected to be start and endpoints of obstacle edges (walls) in form: [(10, 5), (990, 5)]
            edges.append(LineString(coord))
        return edges

    @property
    def get_dist_to_wall(self):
        return self._dist_to_wall

    def get_ray_vpython(self, pos_robot) -> tuple:
        """ Returns the position of the sensor:
                tuple: tuple of vectors for start and end points, where z-coordinate is 0 to simulate 2D
        """
        start, end = self.get_sensor_position(pos_robot)
        return vector(start[0], start[1], 0), vector(end[0], end[1], 0)
    
    def get_sensor_position(self, pos_robot):
        theta = pos_robot[2] + self._offset
        start_pos_sensor = np.array([pos_robot[0], pos_robot[1], theta])
        end_pos_sensor = np.add(start_pos_sensor, np.array([(self._sens_dist) * np.cos(theta), (self._sens_dist) * np.sin(theta), 0]))
        
        # print(f'start_pos_sensor: {start_pos_sensor}\n end_pos_sensor: {end_pos_sensor}')

        return [start_pos_sensor, end_pos_sensor]


    def update(self, pos_robot, verbose=False):
        self.object_detected(pos_robot, verbose=verbose)

    def object_detected(self, pos_robot, verbose=False):
        
        walls = self._obstacle_edges

        sensor_start, sensor_end = self.get_sensor_position(pos_robot)
        sensor_line = LineString([tuple(sensor_start), tuple(sensor_end)])

        # check if there is intersection with the 4 walls
        for w in walls:
            int_pt = sensor_line.intersection(w)  # point of intersection with the wall
            int_pt = list(int_pt.coords)
            if not sensor_line.intersection(w).is_empty:
                # if LineString get first point
                if isinstance(int_pt, LineString):
                    int_pt = int_pt.xy[0]

                dis = self.distance_detected_object(sensor_line.__geo_interface__.get('coordinates')[0][:-1], (int_pt[0][0], int_pt[0][1]))
                self._dist_to_wall = dis
                if verbose:
                    print(f"Wall with coordinates {w} intersection!")
                    print(f"Distance from wall with coordinates {w}: {dis}")
                return
            else:
                # print(f"Distance from {w} out of sensor range")
                pass
        self._dist_to_wall = None

    def intersection_coordinates(self, pos_robot):
        walls = self._obstacle_edges

        sensor_start, sensor_end = self.get_sensor_position(pos_robot)
        sensor_line = LineString([tuple(sensor_start), tuple(sensor_end)])

        # check if there is intersection with the 4 walls
        for w in walls:
            int_pt = sensor_line.intersection(w) # point of intersection with the wall
            if not sensor_line.intersection(w).is_empty:
                return np.array([int_pt.x, int_pt.y])
        return None

    def distance_detected_object(self, sensor_start, intersection_point, verbose=False):
        # finds the distance between sensor starting point and intersection point
        
        sensor_start = np.array(sensor_start)
        intersection_point = np.array(intersection_point)
        
        if verbose:
            print(sensor_start, intersection_point)

        return np.round(np.linalg.norm(sensor_start-intersection_point), 4)


    # def dist_to_wall(self, pos_robot):
    #     # equation of "sensor line", i.e. a line that simulates the infrared beam that the distance sensor sends out
    #     # note that the line is not just the infrared beam, but a geometric line that extends in both directions, so in the direction of the
    #     # infrared beam as well as out of the "back" of the robot
    #
    #     # determining the slope given a point and an angle: https://math.stackexchange.com/questions/105770/find-the-slope-of-a-line-given-a-point-and-an-angle
    #     # slope = np.tan(np.arctan(pos_robot[1]/pos_robot[2]) - self.radians_to_degrees(pos_robot[2]))
    #
    #     # calculating points of intersection
    #     # still to be implemented
    #     pass