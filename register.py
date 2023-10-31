import re
import yaml

from typing import Callable, Dict
from functools import reduce
from explorer.agent.actor import BaseActor
from explorer.agent.sensor import RGBCamera, DepthCamera, SensorPos


__all__ = [
    'register_resources',
    'register_sensors',
    'register_moves',
    'register_enter_hook', 'register_exit_hook',
    'register_from_yaml'
]

class _WrappedBaseActor:
    def __init__(self, cls: BaseActor,
                 register_call: str,
                 *args,
                 **kwargs):
        self._hooked_actor_cls = cls
        self.register_call = register_call

        self.register_args = args
        self.register_kwargs = kwargs

    def __call__(self, *args, **kwargs):
        instance = self._hooked_actor_cls(*args, **kwargs)
        register_method = getattr(instance, self.register_call) 
        register_method(*self.register_args, **self.register_kwargs)
        return instance
        

class _YAMLRegister:
    def __init__(self, register_call: str,
                 yaml_unpack_fn: Callable=lambda x: x):
        '''
        internal register decorator that accepts two types of registeration
        1.
        @register_<key>(<value>)
        class foo
        2. 
        @register_<key>.from_yaml(path)
        class foo
        if use yaml file to do registration, key should be provided
        '''
        self.register_call = register_call
        self.yaml_key = register_call.split('_')[-1]

        self.yaml_unpack_fn = yaml_unpack_fn

    def __call__(self, *args, **kwargs):
        def _decorator(cls):
            return _WrappedBaseActor(cls, self.register_call, *args, **kwargs)
        return _decorator

    def from_yaml(self, yaml_path):
        with open(yaml_path, 'r') as handler:
            values = self.yaml_unpack_fn(
                yaml.safe_load(handler)[self.yaml_key]
            )
        def _decorator(cls):
            '''
            properly unpack yamled parameters,
            in our registeration implemenation,
            we pass a list of parameters as seperate args
            '''
            if isinstance(values, (list, tuple)):
                return _WrappedBaseActor(cls, self.register_call, *values)
            elif isinstance(values, dict):
                return _WrappedBaseActor(cls, self.register_call, **values)
            else:
                return _WrappedBaseActor(cls, self.register_call, values)
        return _decorator

def _parse_position(pos_config: Dict):
    literal2enmu = {
        'left' : SensorPos.LEFT,
        'right' : SensorPos.RIGHT,
        'up' : SensorPos.UP
    }

    def _aggregate(literal, value):
        return literal2enmu[literal] * float(value)

    return reduce(
        lambda x, y : _aggregate(*x) + _aggregate(*y),
        pos_config.items()
    )


def _unpack_yaml_sensor_config(config: Dict):
    sensors = []
    for uuid, setting in config.items():
        equip_type = setting['type'].split('_')[0]
        if equip_type == 'CAMERA':
            camera_type = setting['type'].split('_')[-1]
            if camera_type == 'COLOR':
                camera_setter = RGBCamera
            elif camera_type == 'DEPTH':
                camera_setter = DepthCamera
            else:
                raise ValueError(
                    ("camera domain has invalid subname, only `COLOR` and `DEPTH` are accepted, "
                      "your input is CAMERA_<{}>").format(camera_type)
                )
            sensors.append(camera_setter(
                uuid, setting['resolution'],
                _parse_position(setting['position'])
            ))
        else:
            raise NotImplementedError()
    return sensors


register_sensors = _YAMLRegister(
    'register_sensors',
    _unpack_yaml_sensor_config
)
register_resources = _YAMLRegister('register_resources')
register_moves = _YAMLRegister('register_moves')


def register_from_yaml(path):
    avaliable_registers = {
        'resources' : register_resources,
        'sensors' : register_sensors,
        'moves' : register_moves
    }
    def _decorator(cls):
        if issubclass(cls, BaseActor):
            with open(path, 'r') as handler:
                config = yaml.safe_load(handler)
            for reg_name, fn in avaliable_registers.items():
                if reg_name in config.keys():
                    cls = fn.from_yaml(path)(cls)
            return cls
        else:
            raise NotImplementedError()
    return _decorator

        

    return _decorator


def register_goal():
    pass

def register_enter_hook():
    pass

def register_exit_hook():
    pass


'''
@register_sensors.from_yaml()
@register_resources(num_cpus=0, num_gpus)
class MyActor(Actor):
    ...

actor.register_resources()


'''