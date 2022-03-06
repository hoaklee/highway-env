from highway_env.envs.highway_env import *
from highway_env.envs.merge_env import *
from highway_env.envs.parking_env import *
from highway_env.envs.summon_env import *
from highway_env.envs.roundabout_env import *
from highway_env.envs.two_way_env import *
from highway_env.envs.intersection_env import *
from highway_env.envs.lane_keeping_env import *
from highway_env.envs.u_turn_env import *
from highway_env.envs.exit_env import *

ENVS = {}

def register_env(name):
    """Registers a env by name for instantiation in rlkit."""

    def register_env_fn(fn):
        if name in ENVS:
            raise ValueError("Cannot register duplicate env {}".format(name))
        if not callable(fn):
            raise TypeError("env {} must be callable".format(name))
        ENVS[name] = fn
        return fn

    return register_env_fn