import habitat_sim
import numpy as np


from dataclasses import dataclass


@dataclass
class SceneInfo:
    nav_bbox: np.ndarray

def inspect_environ(path: str) -> SceneInfo:
    '''
    look into basic statistics about a glb scene 
    this is achieved by preloading habitat sim environment
    invoke of this function is heavy-weight, 
    one should avoid calling it for too many times
    '''
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = path
    agent_config = habitat_sim.AgentConfiguration()
    simulator = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_config]))

    scene_info = SceneInfo(
        nav_bbox=list(zip(*simulator.pathfinder.get_bounds()))
    )
    return scene_info