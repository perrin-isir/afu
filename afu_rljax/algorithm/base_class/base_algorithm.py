from abc import ABC, abstractmethod
from functools import partial
import os
import jax
import numpy as np
from gymnasium.spaces import Box
from haiku import PRNGSequence

from afu_rljax.buffer import PrioritizedReplayBuffer, ReplayBuffer, RolloutBuffer
from afu_rljax.util import soft_update
from afu_rljax.plotting import single_episode_plot


class BaseAlgorithm(ABC):
    """
    Base class for algorithms.
    """

    name = None

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm,
        gamma,
    ):
        np.random.seed(seed)
        self.rng = PRNGSequence(seed)

        self.agent_step = 0
        self.episode_step = 0
        self.learning_step = 0
        self.num_agent_steps = num_agent_steps
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.discrete_action = False if type(action_space) == Box else True

    def get_mask(self, env, done):
        return done if self.episode_step != env._max_episode_steps or self.discrete_action else False

    def get_key_list(self, num_keys):
        return [next(self.rng) for _ in range(num_keys)]

    @abstractmethod
    def is_update(self):
        pass

    @abstractmethod
    def step(self, env, state):
        pass

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def explore(self, state):
        pass

    @abstractmethod
    def update(self, writer):
        pass

    @abstractmethod
    def save_params(self, save_dir):
        pass

    @abstractmethod
    def load_params(self, save_dir):
        pass

    def __str__(self):
        return self.name


class OffPolicyAlgorithm(BaseAlgorithm):
    """
    Base class for off-policy algorithms.
    """

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm,
        gamma,
        nstep,
        buffer_size,
        use_per,
        batch_size,
        start_steps,
        update_interval,
        update_interval_target=None,
        tau=None,
    ):
        assert update_interval_target or tau
        super(OffPolicyAlgorithm, self).__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
        )
        if not hasattr(self, "buffer"):
            if use_per:
                self.buffer = PrioritizedReplayBuffer(
                    buffer_size=buffer_size,
                    state_space=state_space,
                    action_space=action_space,
                    gamma=gamma,
                    nstep=nstep,
                    beta_steps=(num_agent_steps - start_steps) / update_interval,
                )
            else:
                self.buffer = ReplayBuffer(
                    buffer_size=buffer_size,
                    state_space=state_space,
                    action_space=action_space,
                    gamma=gamma,
                    nstep=nstep,
                )

        self.discount = gamma ** nstep
        self.use_per = use_per
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_interval = update_interval
        self.update_interval_target = update_interval_target
        self.cumulated_reward = 0.
        self.step_list = []

        if update_interval_target:
            self._update_target = jax.jit(partial(soft_update, tau=1.0))
        else:
            self._update_target = jax.jit(partial(soft_update, tau=tau))

    def is_update(self):
        return self.agent_step % self.update_interval == 0 and self.agent_step >= self.start_steps

    def step(self, env, state):
        self.agent_step += 1
        self.episode_step += 1

        if self.agent_step <= self.start_steps:
            action = env.action_space.sample()
        else:
            action = self.explore(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        # done = terminated or truncated
        # mask = self.get_mask(env, done)

        done = terminated or truncated
        mask = terminated

        self.buffer.append(state, action, reward, mask, next_state, done)

        if done:
            self.episode_step = 0
            next_state, _ = env.reset()

        return next_state

    def step_and_done(self, env, state):
        self.agent_step += 1
        self.episode_step += 1

        if self.agent_step <= self.start_steps:
            action = env.action_space.sample()
        else:
            action = self.explore(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        # done = terminated or truncated
        # mask = self.get_mask(env, done)

        done = terminated or truncated
        mask = terminated
        self.buffer.append(state, action, reward, mask, next_state, done)
        self.cumulated_reward += reward
        self.step_list.append(
            {"observation": state, "next_observation": next_state}
        )

        if done:
            os.makedirs("./plots", exist_ok=True)

            # def plot_projection(x):
            #     return x[0:2]
            #
            # single_episode_plot(
            #     os.path.join(
            #         os.path.expanduser("./plots"),
            #         f"{self.agent_step:12}.png".replace(" ", "0"),
            #     ),
            #     self.step_list,
            #     projection_function=plot_projection,
            #     plot_env_function=env.plot,
            # )
            # print(self.cumulated_reward)
            self.cumulated_reward = 0.
            self.step_list = []
            self.episode_step = 0
            next_state, _ = env.reset()

        return next_state, done


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    Base class for on-policy algorithms.
    """

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm,
        gamma,
        buffer_size,
        batch_size,
    ):
        super(OnPolicyAlgorithm, self).__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
        )
        self.buffer = RolloutBuffer(
            buffer_size=buffer_size,
            state_space=state_space,
            action_space=action_space,
        )
        self.discount = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def step(self, env, state):
        self.agent_step += 1
        self.episode_step += 1

        action, log_pi = self.explore(state)
        next_state, reward, done, _ = env.step(action)
        mask = self.get_mask(env, done)
        self.buffer.append(state, action, reward, mask, log_pi, next_state)

        if done:
            self.episode_step = 0
            next_state, _ = env.reset()

        return next_state
