
import habitat_sim

from abc import abstractmethod, ABC
from enum import Enum
from typing import List

class SensorPos:
    UP = habitat_sim.geo.UP
    LEFT = habitat_sim.geo.LEFT
    RIGHT = habitat_sim.geo.RIGHT



def _setup_habitat_camera(uuid: str,
                          resolution: List[int],
                          position: SensorPos,
                          type):
    camera = habitat_sim.CameraSensorSpec()
    camera.uuid = uuid
    camera.resolution = resolution
    camera.position = position
    camera.sensor_type = type

    return camera

class BaseSensor:
    '''
    dummy wrapper of sensor subclasses,
    '''
    def __init__(self):
        super().__init__()
    def __call__(self):
        return self.init() 
    
    def init(self):
        raise NotImplementedError()

class RGBCamera(BaseSensor):
    def __init__(self, name: str,
                resolution: List[int],
                position: SensorPos):
        self.name = name
        self.resolution = resolution
        self.position = position
    def init(self):
        return _setup_habitat_camera(
            self.name, self.resolution,
            self.position,
            habitat_sim.SensorType.COLOR
        )

class DepthCamera(BaseSensor):
    def __init__(self, name: str,
                resolution: List[int],
                position: SensorPos):
        self.name = name
        self.resolution = resolution
        self.position = position
    
    def init(self):
        return _setup_habitat_camera(
            self.name, self.resolution,
            self.position,
            habitat_sim.SensorType.DEPTH
        )

class SemanticCamera(BaseSensor):
    def __init__(self, name: str,
                 resolution: List[int],
                 position: SensorPos):
        self.name = name
        self.resolution = resolution
        self.position = position
    def init(self):
        return _setup_habitat_camera(
            self.name, self.resolution,
            self.position,
            habitat_sim.SensorType.SEMANTIC
        )