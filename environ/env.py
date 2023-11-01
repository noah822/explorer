
from typing import Dict, final


class BaseEnv:
    def __init__(self, path):
        self._shared_sim = None
        self._env_path = path

    def _update(self, action: str) -> Dict:
        '''
        update internal sim engine with action from actor
        '''
        observes = self._shared_sim.step(action)
        return observes
    
    def post_process(self, observes: Dict):
        '''
        free pass by default, no processing of received raw sensor data
        '''
        return observes

    @final
    def step(self, action: str):
        observes = self._update(action)
        observes= self.post_process(observes)
        return observes
        

class Env(BaseEnv):
    def __init__(self, path):
        super().__init__(path)
