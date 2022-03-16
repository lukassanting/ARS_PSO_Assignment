import numpy as np

class Beacon:
    def __init__(self, x: float, y: float, radius: float = 20, active: int = 0) -> None:
        self._x = x
        self._y = y
        self._radius = radius
        self._active = active
    
    def __str__(self) -> str:
        return f'Beacon {id(self)}, coordinates: {self._x}{self._y}, active: {self._active}'

    