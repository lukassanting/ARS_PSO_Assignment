import numpy as np
from typing import List
from sqlalchemy import func


def rosenbrock(x:float, y:float, a:float=0, b:float=1):
    return (a-x)**2+b*(y-x**2)**2


def rastrigin(x:float, y:float):
    return 20+x**2-10*np.cos(2*np.pi*x)+y**2-10*np.cos(2*np.pi*y)


class particle:
    pos:np.ndarray # position
    vel:float # velocity; change so velocity includes an angle 
    perf:float # performance
    best_personal_pos:np.ndarray
    best_personal_perf:float
    best_global_pos:np.ndarray
    best_global_perf:float
    fitness_func:func
    inertia:float

    def __init__(self, position, velocity, function, inertia=1):
        self.pos = position
        self.vel = velocity
        self.perf = function(position[0], position[1])
        self.best_personal_pos = position
        self.best_personal_perf = self.perf
        self.best_global_pos = np.array([10,10])
        self.best_global_perf = np.Inf
        self.fitness_func = function
        self.inertia = inertia

    def evaluate(self):
        return self.fitness_func(self.pos[0], self.pos[1])

    def decrease_inertia(self, start:float, end:float, max_iter:int):
        self.inertia -= (self.inertia-end)/max_iter

    def move(self):
        b = 2
        c = 2
        r1 = np.random.uniform(0,1)
        r2 = np.random.uniform(0,1)
        self.pos = self.pos + self.vel
        self.vel = self.inertia*self.vel + b*r1*(self.best_personal_pos-self.pos)+c*r2*(self.best_global_pos-self.pos)

    def update_personal_best(self):
        current_perf = self.evaluate()
        if current_perf < self.best_personal_perf:
            self.best_personal_pos = self.pos
            self.best_personal_perf = current_perf
            if current_perf < self.best_global_perf:
                return [self.pos, current_perf]
        else: return None

    def update_global_best(self, position, performance):
        self.best_global_pos = position
        self.best_global_perf = performance


class swarm:
    particles:List[particle]
    fitness_func:func
    global_best_pos:np.ndarray
    global_best_perf:float
    time:int
    
    def __init__(self, num_particles, fitness_func):
        self.fitness_func = fitness_func
        self.particles = []
        self.global_best_pos = np.zeros(shape=2)
        self.global_best_perf = np.Inf
        self.time = 0
        for index in range(num_particles):
            pos_x = np.random.uniform(10, -10)
            pos_y = np.random.uniform(10, -10)
            vel_x = np.random.normal()
            vel_y = np.random.normal()
            position = np.array([pos_x,pos_y])
            velocity = np.array([vel_x, vel_y])
            self.particles.append(particle(position, velocity, fitness_func))

    def time_step(self, max_iter, verbose=False, inertia_start=1, inertia_end=1):
        # make one time-step, i.e. let all particles move and update their parameters

        for particle in self.particles:
            particle.move()
            update = particle.update_personal_best()
            if update:
                self.global_best_pos = update[0]
                self.global_best_perf = update[1]
            if inertia_end != inertia_start:
                particle.decrease_inertia(inertia_start, inertia_end, max_iter)

        for particle in self.particles:
            particle.update_global_best(self.global_best_pos, self.global_best_perf)

        self.time += 1
        if verbose:
            print(f'Time step {self.time}')
            print(f'Best global position: {self.global_best_pos}')
            print(f'Performance on that position: {self.global_best_perf}')
            print()

    def search(self, max_iter, verbose=False, inertia_start=1, inertia_end=1):
        for i in range(max_iter):
            self.time_step(max_iter, verbose=verbose)
        print('Swarm optimization finished.')
        print(f'Best location {self.global_best_pos} with performance: {self.global_best_perf}')


swarm = swarm(20, rastrigin)
swarm.search(1000, inertia_start=0.9, inertia_end=0.9)