import habitat_sim



from abc import ABC, ABCMeta, abstractmethod

from .sensor import BaseSensor
from explorer.simulation.dispatcher import RayResources

class BaseActor(ABC):
    '''
    standard protocols of explorer actors
    '''
    def __init__(self):
        super().__init__()
        self._ray_resources = RayResources()
        self._moves = None
        self._sensors = None

    
    @abstractmethod
    def step(self, *args, **kwargs):
        '''
        main entry point of explorer actor
        '''
        pass

    def register_resources(self, num_cpu, num_gpu=None):
        self._ray_resources = RayResources(num_cpu, num_gpu)
        return self
    
    def register_moves(self, *args):
        self._moves = args
        return self
    
    def register_sensors(self, *args):
        self._sensors = []
        for sensor_setter in args:
            assert issubclass(type(sensor_setter), BaseSensor), \
            'Your own implementation of sensor class should subclass `BaseSensor`'
            self._sensors.append(sensor_setter())
        self._sensors = tuple(self._sensors)
        
        return self



class Actor(BaseActor):
    def __init__(self):
        super().__init__()
    
    def step(self):
        pass


class SemanticActor(BaseActor):
    pass
