import numpy as np
import matplotlib.pyplot as plt
from typing import List

'''
    A beacon is set on each end-point between the walls and measures the euclidean distance from itself
    to the center of the robot, if the latter is inbetween the radius range of the beacon.
'''

class Beacon:
    def __init__(self, x: float, y: float, radius: float = 50.0, active: int = 0) -> None:
        self._x = x
        self._y = y
        self._radius = radius
        self._active = active
    
    def __str__(self) -> str:
        return f"Beacon {id(self)}, active: {self._active}, coordinates: {format(self._x, '.4f')} {format(self._y, '.4f')}"

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y
    
    @property
    def radius(self):
        return self._radius

    @property
    def active(self):
        return self._active

def verbose_all_beacons(beacons: list) -> None:
    for b in beacons:
        print(b)

def nr_active_beacons(beacons: List[Beacon], x: float, y: float) -> int:
    '''
        returns the number of the beacons that are active. A beacon is considered active when the robot
        center position is inside the range circle of the beacon (the laser beam reaches the center of the )
        this method is a prerequisite of the trilateration method, if there are at least 3 active beacons,
        the trilateration algorithm can be performed to estimate the robot's position
    '''
    active_beacons = 0
    
    # check if the robot is in the slant range of the beacons
    for b in beacons:
        # find euclidean distance betweeen beacon and center of robot
        robot_pos, beacon_pos = np.array((x, y)), np.array((b.x, b.y))
        dist = np.linalg.norm(robot_pos - beacon_pos)
        print(f'Distance: {dist}')

        # check if dist is in beacon range
        if dist <= b.radius:
            active_beacons += 1
            b._active = 1

    return active_beacons

def trilateration(beacon1: Beacon, beacon2: Beacon, beacon3: Beacon) -> tuple:
    pass


#### TESTING ####

NR_BEACONS = 5
ENV_LENGTH, ENV_WIDTH = 250, 250

beacons = [Beacon(np.random.uniform(0, ENV_LENGTH), np.random.uniform(0, ENV_WIDTH)) for x in range(NR_BEACONS)]
verbose_all_beacons(beacons)

bot_pos = (np.random.uniform(0, ENV_LENGTH), np.random.uniform(0, ENV_WIDTH))
active = nr_active_beacons(beacons, *bot_pos)
print(f'Active beacons: {active}')

verbose_all_beacons(beacons)

# scatter points
figure, axes = plt.subplots()
plt.axis("equal")

for b in beacons:
    plt.scatter(b.x, b.y, c='k')
    cc = plt.Circle((b.x, b.y), b.radius, facecolor='none', edgecolor='b')
    axes.add_patch(cc)

plt.scatter(bot_pos[0], bot_pos[1], c='r')
plt.show()
