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

def nr_active_beacons(beacons: List[Beacon], x: float, y: float) -> list:
    '''
        returns the beacons that are active. A beacon is considered active when the robot center
        position is inside the range circle of the beacon (the laser beam reaches the center of the robot)
        this method is a prerequisite of the trilateration method, if there are at least 3 active beacons,
        the trilateration algorithm can be performed to estimate the robot's position
    '''
    active_beacons = []
    
    # check if the robot is in the slant range of the beacons
    for b in beacons:
        # find euclidean distance betweeen beacon and center of robot
        robot_pos, beacon_pos = np.array((x, y)), np.array((b.x, b.y))
        dist = np.linalg.norm(robot_pos - beacon_pos)
        print(f'Distance from robot: {dist}')

        # check if dist is in beacon range
        if dist <= b.radius:
            active_beacons.append(b)
            b._active = 1

    return active_beacons

# reference: https://www.101computing.net/cell-phone-trilateration-algorithm/
def trilateration(robot_pos: tuple, beacons: list) -> tuple:
    '''
        estimates the bot position in relation to the position of exactly 3 beacons
        for simplification purposes we calculate the exact position of robot from the
        beams within sensor range and add noise afterwards to mimic sensor noise
    '''

    assert len(beacons) == 3, "provide at least 3 beacons"

    # calculate distance of robot from eaach beam (the distance corresponds to signal strength from the beamer)
    beam_robot_dist = []
    for b in beacons:
        beam_robot_dist.append(np.linalg.norm(np.array([robot_pos]) - np.array((b.x, b.y))))

    # trilateration algorithm
    A = 2 * beacons[1].x - 2 * beacons[0].x
    B = 2 * beacons[1].y - 2 * beacons[0].y
    C = 2 * beacons[2].x - 2 * beacons[1].x
    D = 2 * beacons[2].y - 2 * beacons[1].y
    E = beam_robot_dist[0] ** 2 - beam_robot_dist[1] ** 2 - beacons[0].x ** 2 + beacons[1].x ** 2 - beacons[0].y ** 2 + beacons[1].y ** 2
    F = beam_robot_dist[1] ** 2 - beam_robot_dist[2] ** 2 - beacons[1].x ** 2 + beacons[2].x ** 2 - beacons[1].y ** 2 + beacons[2].y ** 2

    x = (E*D - F*B) / (D*A - B*C)
    y = (E*C - A*F) / (B*C - A*D)

    return x, y


#### TESTING ####

NR_BEACONS = 5
ENV_LENGTH, ENV_WIDTH = 100, 100

bot_pos = (np.random.uniform(0, ENV_LENGTH), np.random.uniform(0, ENV_WIDTH))
beacons = [Beacon(np.random.uniform(0, ENV_LENGTH), np.random.uniform(0, ENV_WIDTH)) for x in range(NR_BEACONS)]
# verbose_all_beacons(beacons)

active = nr_active_beacons(beacons, *bot_pos)
print(f'Active beacons: {len(active)}')

if len(active) >= 3:
    print(f'At least 3 active beams found. Performing trilateration...')
    beam_robot_dist = trilateration(bot_pos, active[:3])
    print(f'Estimated bot position: {beam_robot_dist}')
    print(f'Real bot position: {bot_pos}')

verbose_all_beacons(beacons)


#### VISUALIZATION ####

figure, axes = plt.subplots()
plt.axis("equal")

for b in beacons:
    plt.scatter(b.x, b.y, c='k')
    cc = plt.Circle((b.x, b.y), b.radius, facecolor='none', edgecolor='b')
    if b.active:
        plt.plot(np.array([b.x, bot_pos[0]]), np.array([b.y, bot_pos[1]]), 'g--')
    axes.add_patch(cc)

plt.scatter(bot_pos[0], bot_pos[1], c='r')
plt.show()
