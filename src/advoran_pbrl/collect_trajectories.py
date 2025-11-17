from typing import List
import numpy as np
import tensorflow as tf
from tf_agents.trajectories import time_step as ts

from .structures import StepKPM, Transition, Trajectory


def _victim_act(victim_policy, obs: np.ndarray) -> int:

    obs_tf = tf.convert_to_tensor(obs[None, ...], dtype=tf.float32)
    tstep = ts.restart(obs_tf)
    action_step = victim_policy.action(tstep)
    action = action_step.action.numpy()[0]
    return int(action)


def collect_trajectories(
    base_env,           
    victim_policy,      
    num_episodes: int,
) -> List[Trajectory]:

    env = base_env
    trajectories: List[Trajectory] = []

    for ep in range(num_episodes):
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            obs, info = reset_out
        else:
            obs, info = reset_out, {}
        done = False
        steps: List[Transition] = []

        while not done:
            a = _victim_act(victim_policy, obs.astype(float))

            step_out = env.step(a)
    
            if len(step_out) == 5:
                next_obs, _, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                next_obs, _, done, info = step_out

    
            kpm = StepKPM(
                throughput_bps=info.get("embb_throughput_bps", 0.0),
                latency_s=info.get("urllc_latency_s", 0.0),
            )
            steps.append(
                Transition(
                    s=obs.astype(float),
                    a=a,
                    s_next=next_obs.astype(float),
                    kpm=kpm,
                )
            )
            obs = next_obs

        trajectories.append(Trajectory(steps=steps))

    return trajectories