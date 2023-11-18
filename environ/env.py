import habitat_sim.utils.common as sim_utils
from typing import Dict, final


class BaseEnv:
    def __init__(self, path, config=None):
        self._shared_sim = None
        self._env_path = path
        self._config_path = config

    def _update(self, action: str) -> Dict:
        '''
        update internal sim engine with action from actor
        '''
        observes = self._shared_sim.step(action)
        if type(self).post_process != BaseEnv.post_process:
            # allow incremental configuration of observations
            observes = super(self.__class__, self).post_process(observes)
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
    TARGET_AGENT = 0
    def __init__(self, path, config=None):
        super().__init__(path, config)
    def post_process(self, observes: Dict):
        '''
        besides sensor data, by default, Env will also expose agent's
        - position: [x, y, z]
        - rotation: rotated angle in radian about y-axis
        '''
        agent_position = self._shared_sim.agents[Env.TARGET_AGENT].state.position
        agent_quat = self._shared_sim.agents[Env.TARGET_AGENT].state.rotation
        radian, rotate_vec = sim_utils.quat_to_angle_axis(agent_quat)
        if rotate_vec[1] < 0:
            radian = -radian
        observes['position'] = agent_position
        observes['rotation'] = radian
        return observes


try:
    import numpy as np
    import copy
    from habitat.utils.visualizations import maps
    class BirdEyeEnv(Env):
        def __init__(self,
                     path: str, config: str=None,
                     map_resolution: int=512,
                     agent_radius_px: int=15,
                     track_trajectory: bool=False):
            super().__init__(path, config)
            self.map_resolution = map_resolution
            self.agent_radius_px = agent_radius_px

            self._cached_topdown_map = None
            self._track_trajectory = track_trajectory
            self.trajectory = []
        
        @property
        def cached_topdown_map(self):
            if self._cached_topdown_map is None:
                self._cached_topdown_map = self._init_topdown_map()
            return copy.deepcopy(self._cached_topdown_map)
        
        
        def _init_topdown_map(self):
            recolor_map =  np.array(
                [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
            )
            topdown_map = maps.get_topdown_map_from_sim(self._shared_sim, self.map_resolution)
            topdown_map = recolor_map[topdown_map]
            return topdown_map

        def post_process(self, observes: Dict):
            '''
            Inherit from Env class,
            which provides agent's real world position and orientation in observes
            '''
            topdown_map = self.cached_topdown_map
            realworld_pos, rotation = observes['position'], observes['rotation']
            grid_dims = (topdown_map.shape[0], topdown_map.shape[1])
            agent_grid_pos = maps.to_grid(realworld_pos[2], realworld_pos[0], grid_dims, self._shared_sim)
            if self._track_trajectory:
                self.trajectory.append(agent_grid_pos)
                maps.draw_path(topdown_map, [agent_grid_pos], thickness=2)
            maps.draw_agent(topdown_map, agent_grid_pos, np.pi+rotation, self.agent_radius_px)
            observes['topdown_map'] = topdown_map
            return observes


except ImportError:
    pass

    

