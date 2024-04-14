from typing import List, Dict, Union

from .dispatcher import Context, BaseDispatcher, DebugDispatcher
from explorer.agent.actor import BaseActor
from explorer.environ.env import Env


Scalar = Union[int, float]
def run_simulation(
    actors: List[BaseActor],
    envs: List[Env],
    *,
    mode: str='debug',
    global_resources: Dict[str, Scalar]={'num_cpu':1, 'num_gpu':0}
):
    assert len(actors) == len(envs), 'actors and envs should be properly paired'
    contexts = []
    for actor, env in zip(actors, envs):
        contexts.append(Context(actor, env))

    dispatcher = _prepare_dispatcher(contexts, mode, global_resources)
    # dispatcher = DebugDispatcher(contexts)
    # dispatcher = BaseDispatcher(contexts, {'num_cpu':4, 'num_gpu':0.1})
    dispatcher.launch()



def _prepare_dispatcher(contexts, mode: str, global_resources: Dict[str, Scalar]) -> BaseDispatcher:
    if mode == 'debug':
        return DebugDispatcher(contexts)
    elif mode == 'basic':
        return BaseDispatcher(contexts, global_resources)
    else:
        raise NotImplementedError(
            f'''\
            {mode} type dispather has not been supported,
            Current Support: `debug`, `basic`
            '''
        )

