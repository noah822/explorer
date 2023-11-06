import numpy as np


from functools import partial
from typing import List, Dict, Any


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



ActorLocalProgress = Any

class GroupActor:
    '''
    standardize protocols for actors that work as part of a community
    - report_progress:
      report local progress to others in the community
    - recv_progress:
      recv progress of others in the community in the lastest iteration
    '''
    def __init__(self):
        pass
    
    def report_progress(self) -> ActorLocalProgress:
        '''
        report local progress to other actors in the same community
        '''
        raise NotImplementedError()
    
    def recv_progress(self, comm_synced: List[ActorLocalProgress]):
        '''
        operates on progress reported by every actors in the community, self included
        '''
        pass

try:
    import torch
    import torch.nn as nn

    import copy
    import threading
    from concurrent.futures import ThreadPoolExecutor



    # following two functions are used for better static workload distribution
    # among spawned threads when aggregating community grads
    def _flatten_nested_dict(d: Dict,
                            sep: str='*'):
        '''
        recursively flatten out dict keys
        {'a' : {'b':1 , 'c':2 }}
        after flattening:
        {'a*b':1, 'a*c':2}
        '''
        res = {}
        for k, v in d.items():
            if isinstance(v, dict):
                flattedn_group = _flatten_nested_dict(v, sep)
                for k_, v_ in flattedn_group.items():
                    res[f'{k}{sep}{k_}'] = v_
            else:
                res[k] = v
        return res

    def _stack_flatten_dict(d: Dict,
                            sep: str='*'):
        res = {}
        def __recursed_stack(nested_key, res_holder, value):
            if sep in nested_key:
                parent_k, children_key = nested_key.split(sep, 1)
                if parent_k not in res_holder:
                    res_holder[parent_k] = dict()  # stub non-existed key that can be passed as reference
                __recursed_stack(children_key, res_holder[parent_k], value)
            else:
                res_holder[nested_key] = value

        for k, v, in d.items():
            __recursed_stack(k, res, v)
        return res

    class TorchActor(BaseActor, GroupActor):
        '''
        MRO strucutre
        BaseActor   GroupActor [collabroative training support]
                \  /  
              TorchActor
        
        exposed api for subclass that supports model training
        - report_grads()
        - 
        
        '''
        def __init__(self):
            super().__init__()
            self._num_threads = 1

        def _get_torch_modules(self) -> Dict[str, nn.Module]:
            self_contained_torch_modules = {}
            for attr_name in self.__dir__():
                attr = getattr(self, attr_name)
                if issubclass(type(attr), nn.Module):
                    self_contained_torch_modules[attr_name] = attr
            return self_contained_torch_modules
        
        def _set_num_threads(self, num: int):
            self._num_threads = num
            return self

        def report_progress(self):
            res = self.report_grads()
            return res
        

        def recv_progress(self, comm_synced: List[ActorLocalProgress]):
            '''
            pipeline:
            - group grads of each parameter from `flattened actor progress`
            - computation, distribute to spawned thread if possible

            [important]
            Imposed by ray, those ActorLocalProgress as np.array
            are read-only. A local copy of grads should be made first
            '''

            # no multithread here, because of gil 
            comm_grads_state_dict: Dict[str, List[np.ndarray]] = {}
            per_worker_state: Dict[str, np.ndarray] = next(iter(comm_synced))
            for param_name in per_worker_state.keys():
                param_grad_comm_res = []
                for worker_res in comm_synced:
                        param_grad_comm_res.append(worker_res[param_name])
                comm_grads_state_dict[param_name] = param_grad_comm_res
    
            averged_grads_state_dict = {}

            if self._num_threads > 1:
                write_lock = threading.Lock()

                def _per_worker_job(workload, lock):
                    param_name, grads_list = workload
                    averged_param_grads = np.mean(grads_list, axis=-1)
                    lock.acquire()
                    averged_grads_state_dict[param_name] = averged_param_grads
                    lock.release()

                with ThreadPoolExecutor(max_workers=self._num_threads) as executor:
                    executor.map(partial(_per_worker_job, lock=write_lock), comm_grads_state_dict.items())
            else:
                for param_name, grad in comm_grads_state_dict.items():
                    averged_grads_state_dict[param_name] = np.mean(grad, axis=-1)


            return _stack_flatten_dict(averged_grads_state_dict)





        def report_grads(self):
            '''
            by default,
            when the actor is about to report its local gradients to be synced to the community
            actor will check all attributes of self instance that subclasses torch.nn.Module
            collect their gradients as local result 

            [tentative/experimental feature]
            report grads of model params as numpy array instead of torch tensor
            `considerations`:
            -  ray currently supports zero-copy for numpy array but not for torch tensor,
               so that numpy array is more memory friendly.
               There is only one `read_only` copy of each np array remote maintained
               in the object tables, which can be shared by multiple ray actors

            '''
            grads_group = {}
            for attr_name, module in self._get_torch_modules().items():
                module_grads = {}
                for param_name, param in module.named_parameters():
                    module_grads[param_name] = param.grad.cpu().numpy()
                grads_group[attr_name] = module_grads

            # flatten the output before reporting be default
            return _flatten_nested_dict(grads_group)
except ImportError:
    pass



