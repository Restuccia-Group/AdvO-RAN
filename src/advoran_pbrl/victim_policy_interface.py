
import numpy as np
import tensorflow as tf
from tf_agents.trajectories import time_step as ts

from .victim_policy_interface import BaseVictimPolicy


class VictimPolicy(BaseVictimPolicy):

    def __init__(self, saved_model_dir: str):
        """
        :param saved_model_dir: directory passed to tf.saved_model.save(...)
        """
        # This returns a TFAgents policy object with .action(...)
        self._policy = tf.saved_model.load(saved_model_dir)

    def act(self, obs: np.ndarray) -> int:
        """
        :param obs: NumPy array with the environment observation.
                    For a Gym Box(obs_dim,) this should just be shape [obs_dim].
        :return: int action (discrete action index)
        """
        obs = np.asarray(obs, dtype=np.float32)

        # For simple Box obs: shape [obs_dim] -> [1, obs_dim]
        obs_tf = tf.convert_to_tensor(obs[None, ...], dtype=tf.float32)

        # Build a TimeStep for TF-Agents policy (no reward yet, first step)
        time_step = ts.restart(obs_tf)

        # Call the saved policy
        action_step = self._policy.action(time_step)

        # Assume discrete scalar action; take first element of batch
        action = action_step.action.numpy()[0]
        return int(action)
