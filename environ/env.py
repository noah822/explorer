

from abc import ABC, abstractmethod

'''
@register_sensors(...)
is the only way of attaching sensors to clients

'''

class BaseEnv(ABC):
    def __init__(self):
        pass
        self._shared_sim = None

    def update(self):
        '''
        
        '''
        observes = []
    
    @abstractmethod
    def step(self):
        pass

class Env:
    def __init__(self):
        pass
    def step(self):
        pass