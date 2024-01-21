import habitat_sim
import numpy as np

from typing import List

class SensorPos:
    UP = habitat_sim.geo.UP
    LEFT = habitat_sim.geo.LEFT
    RIGHT = habitat_sim.geo.RIGHT

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

class CameraSensor(BaseSensor):
    def __init__(self, name: str,
                 camera_type: str,
                 resolution: List[int],
                 position: SensorPos):
        super().__init__()
        self.name, self.camera_type = name, camera_type

        h, w = resolution
        assert (h == w), 'currently only support camera of square size'
        self.resolution = resolution
        self.position = position

        # hfov is default to 90 degree for habitat CameraSensor
        # focal is in unit of pixel number
        self.focal = w/2 * 1/ np.tan((np.pi/2)/2) 


    def init(self):
        camera = habitat_sim.CameraSensorSpec()
        camera.uuid = self.name
        camera.resolution = self.resolution
        camera.position = self.position
        camera.sensor_type = self.camera_type
        return camera
    
    

class RGBCamera(CameraSensor):
    def __init__(self, name: str,
                resolution: List[int],
                position: SensorPos):

        super().__init__(
            name, habitat_sim.SensorType.COLOR, resolution, position
        )


class DepthCamera(CameraSensor):
    def __init__(self, name: str,
                resolution: List[int],
                position: SensorPos):
        super().__init__(
            name, habitat_sim.SensorType.DEPTH, resolution, position
        )
    

class SemanticCamera(CameraSensor):
    def __init__(self, name: str,
                 resolution: List[int],
                 position: SensorPos):
        super().__init__(
            name, habitat_sim.SensorType.SEMANTIC, resolution, position
        )