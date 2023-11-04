import ray

from typing import List

from .dispatcher import ExecEngine



class Comm:
    '''
    Community class for distributed training in explorer.
    It wraps multiple workers that share the same model and 
    performs the following operations:
    - periodically syncs worker's progress,
    - configures the barrier
    - re-lanuches workers
    '''
    def __init__(self, worker_procs: List[ExecEngine]):
        '''
        actor in the context attribute of input worker process should implement
        - report_grads
        - 
        '''
        self.worker_procs = worker_procs
        # assert all([issubclass(w, TorchActor) for w in worker_procs]), \
        # 'community should be wrapped over actor classes that subclass `TorchActor`'

    @classmethod
    def broadcast_futures(cls, worker_procs):
        futures = []
        for worker in worker_procs:
            futures.append(worker._report_actor_progress.remote())


        for worker in worker_procs:
            worker._config_dependency.remote(futures)
    
    @classmethod
    def sync(cls, worker_procs):
        futures = []
        for worker in worker_procs:
            futures.append(worker._sync_barrier.remote())
        ray.get(futures)
    
