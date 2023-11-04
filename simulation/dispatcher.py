import habitat_sim
import ray


from dataclasses import dataclass
from typing import List, Union, Dict


Scalar = Union[int, float, None]
Actor = List
Env = List

def _valid_dict_key(d: Dict, valid_names):
    return all([i in valid_names for i in d])

@dataclass
class Context:
    actor: Actor
    environ: Env

@dataclass
class RayResources:
    num_cpu: Scalar=None
    num_gpu: Scalar=None

    def has_not_set_cpu(self):
        return self.num_cpu is None

    def has_not_set_gpu(self):
        return self.num_gpu is None



class ExecEngine:
    def __init__(self, context: Context):
        self.environ, self.actor = context.environ, context.actor

        self._dependency = []

    def _report_actor_progress(self):
        return self.actor.report_progress()

    def _at_entry(self):
        # initiate habitat sensors, refer to doc in BaseActor for more info
        self.actor._sensors = [sensor_setter() for sensor_setter in self.actor._sensors]
    
    def exec(self):
        self._at_entry()
        # postpone context bind at worker end
        self._bind(self.actor, self.environ)
        action = 'turn_left'
        
        self.frame_cnt = 0
        while self.frame_cnt < 4:
            observe = self.environ.step(action)
            action = self.actor.step(observe)
            self.frame_cnt += 1

    def _bind(self, actor, environ):
        '''
        use current configuration of actor and environ to build a shared
        habitat sim under the hood
        [attribute of interest]
        actor: sensors, init_state
        environ: path
        '''
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = environ._env_path

        agent_config = habitat_sim.AgentConfiguration()
        agent_config.sensor_specifications = actor._sensors

        actor._shared_sim = environ._shared_sim = \
        habitat_sim.Simulator(
            habitat_sim.Configuration(backend_cfg, [agent_config])
        )

    def _config_dependency(self, futures: List):
        self._dependency = futures

    def _sync_barrier(self):
        res = ray.get(self._dependency)




class BaseDispatcher:
    def __init__(self,
                 contexts: List[Context],
                 global_resources: Dict[str, Scalar]={'num_cpu':1, 'num_gpu':0}):
        
        self.contexts = contexts
        assert _valid_dict_key(global_resources, ['num_gpu', 'num_cpu']), \
        'Invalid keys in @param global_resources, which only accepts `num_cpu` and `num_gpu`'
        self.global_resources = global_resources

        self._distribute_resources()

    def launch(self):
        # wrap into ray actors
        worker_procs = self._init_worker_procs()
        futures = [proc.exec.remote() for proc in worker_procs]
        res = ray.get(futures)

    def reap_futures(self):
        pass

    def _distribute_resources(self):
        num_cpu_to_dist = self.global_resources['num_cpu']
        num_gpu_to_dist = self.global_resources['num_gpu']

        non_allocate_cpu_actor = []
        non_allocate_gpu_actor = []

        for context in self.contexts:
            actor = context.actor
            preset = actor._ray_resources
            if preset.has_not_set_cpu():
                non_allocate_cpu_actor.append(actor)
            else:
                num_cpu_to_dist -= preset.num_cpu
            
            if preset.has_not_set_gpu():
                non_allocate_gpu_actor.append(actor)
            else:
                num_gpu_to_dist -= preset.num_gpu
        
        assert num_cpu_to_dist >=0, 'hardcoded cpu exceeds the avaliable global cpu resources'
        assert num_gpu_to_dist >=0, 'hardcoded gpu exceeds the avaliable global gpu resources'

        if len(non_allocate_cpu_actor):
            avg_cpu = int(num_cpu_to_dist / len(non_allocate_cpu_actor))
            for actor in non_allocate_cpu_actor:
                actor._ray_resources.num_cpu = avg_cpu
        
        if len(non_allocate_gpu_actor):
            avg_gpu = int(num_gpu_to_dist / len(non_allocate_gpu_actor))
            for actor in non_allocate_gpu_actor:
                actor._ray_resources.num_gpu = avg_gpu
    
    def _init_worker_procs(self):
        worker_procs = []
        for context in self.contexts:
            num_cpu = context.actor._ray_resources.num_cpu
            num_gpu = context.actor._ray_resources.num_gpu
            worker_procs.append(
                BaseDispatcher._wrap_into_remote_actor(
                    ExecEngine, {'num_cpus':num_cpu, 'num_gpus':num_gpu}
                ).remote(context)
            )
        return worker_procs

    @classmethod
    def _wrap_into_remote_actor(cls, engine, config):
        ray_remote_engine = ray.remote(**config)(engine)
        return ray_remote_engine

    @classmethod
    def _check_registeration():
        pass

from .comm import Comm

class GroupDispatcher(BaseDispatcher):
    def __init__(self,
                 contexts: List[Context],
                 global_resources: Dict[str, Scalar]={'num_cpu':1, 'num_gpu':0}):
        super().__init__(contexts, global_resources)

    def launch(self):
        worker_procs = super()._init_worker_procs()
        comm = Comm(worker_procs)
        Comm.broadcast_futures(worker_procs)
        Comm.sync(worker_procs)


'''
input:
    a list of actors
    a list of environs

    [entry_point] run_simulation()
prepare/group those contexts
            |
            |
        Dispatcher
dispatch those contexts to [ray nodes]---collect------
            |                            futures     |
            |                                        |                   
    Context Exec Engine + [Model Executor]           |
            |                                        | 
            |________________________________________|      

There are two possible simulation paradigm
    1. Every engine has their own copy of models
    and can train on their own accord
    In this case, model needs to be synced after each back-prop

    2. Two types of workers involved [usually used when model is too large]
    - stand-alone model runner
    - context simulator
'''
