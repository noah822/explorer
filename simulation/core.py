from typing import List

from .dispatcher import Context, BaseDispatcher
from explorer.agent.actor import BaseActor
from explorer.environ.env import Env


def run_simulation(
    actors: List[BaseActor],
    envs: List[Env],
    config
):
    assert len(actors) == len(envs), 'actors and envs should be properly paired'
    contexts = []
    for actor, env in zip(actors, envs):
        contexts.append(Context(actor, env))

    dispatcher = BaseDispatcher(contexts, {'num_cpu':4, 'num_gpu':0.5})
    dispatcher.launch()



# def _prepare_dispatcher(contexts, config) -> BaseDispatcher:
#     return BaseDispatcher(conte)
