from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os
from .gather import GatherEnv
def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["gather"] = partial(env_fn, env=GatherEnv)


if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
