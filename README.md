# Introduce `explorer`
Easy to use Object Nav framework built on habitat and ray. `explorer` harnesses avaliable cpu/gpu resources
and provides user-friendly apis for batched Object Navigation simulation.
Empowered by ray, user can have good control over their hardware resources without the burden to write their python multiprocessing codes.
# Workflow
Habitat simulation backend is logically divided into actor and environment components. Environ provides current observations according to actor's
equipped sensor devices, and actor makes its decision of what the next action is in light of the received observations. Environment then updates observations
after taking the next action. The loop proceeds until simulation ends.
# Use Example
```python
from explorer.register import *
from explorer.agent import Actor
from explorer.environ import Env
from explorer.agent.sensor import RGBCamera, SensorPos
from explorer.simulation.core import run_simulation
camera = RGBCamera(name='camera1', resolution=[512, 512], position=SensorPos.LEFT)

# user config their actors with register apis
@register_moves('turn_left', 'turn_right')
@register_resources(num_cpu=2, num_gpu=1)
@register_sensors(camera)
class MyActor(Actor):
    def __init__(self):
        super().__init__()
    def step(self, observes):
        # your impl goes here
        return action

env_path = '<your_env_path>'
batch_size = 10
actors = [MyActor() for _ in range(batch_size)]
envs = [Env(env_path) for _ in range(batch_size)]
config = <your_configure_dict>
run_simulation(actors, envs, config)

```

# Code Structure
```
input:
    a list of actors
    a list of environs
[entry_point]
       run_simulation()
     prepare contexts
            |
            |
        Dispatcher
dispatch those contexts to [ray nodes]---collect--<<--
            |                            futures     |
            |                                        |                   
    Context Exec Engine + [Model Executor]        send back
            |                                        | 
            |________________________________________|  
```

