import numpy as np
import gymnasium as gym
from typing import Any


class FlattenSpaceEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        self.action_space = gym.spaces.Box(env.action_space.low.flatten(),
                                           env.action_space.high.flatten(),
                                           dtype=np.float64)

        self.observation_space = gym.spaces.Box(env.observation_space.low.flatten(),
                                                env.observation_space.high.flatten(),
                                                dtype=np.float64)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(
            np.expand_dims(action, axis=0))
        return (state.flatten(),
                reward.flatten().item(),
                terminated.flatten().item(),
                truncated.flatten().item(),
                info)

    def reset(self,
              *,
              seed=None,
              options=None
              ):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs.flatten(), info