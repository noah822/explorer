import habitat_sim


from .sensor import BaseSensor
from explorer.simulation.dispatcher import RayResources

class BaseActor:
    '''
    standard protocols of explorer actors
    '''
    def __init__(self):
        super().__init__()
        self._ray_resources = RayResources()
        self._moves = None
        self._sensors = None

    
    def step(self, *args, **kwargs):
        '''
        main entry point of explorer actor
        '''
        raise NotImplementedError()

    def register_resources(self, num_cpu, num_gpu=None):
        self._ray_resources = RayResources(num_cpu, num_gpu)
        return self
    
    def register_moves(self, *args):
        self._moves = args
        return self
    
    def register_sensors(self, *args):
        '''
        Instantiation of habitat sensor should be postponed to remote ray worker,
        because habitat's sensor class can not be serialized under pickle protocol,
        which means that it can not be distributed over ray nodes
        '''
        self._sensors = []
        for sensor_setter in args:
            assert issubclass(type(sensor_setter), BaseSensor), \
            'Your own implementation of sensor class should subclass `BaseSensor`'
            self._sensors.append(sensor_setter)
        self._sensors = tuple(self._sensors)
        
        return self



class Actor(BaseActor):
    def __init__(self):
        super().__init__()
    
    def step(self):
        pass


class SemanticActor(BaseActor):
    pass
