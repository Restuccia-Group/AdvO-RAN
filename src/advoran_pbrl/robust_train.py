import tensorflow as tf

from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.utils import common

from .robust_env import make_robust_tf_env
from .perturbation_net import PerturbationNet
from .reward_model import RewardModel


def build_ppo_agent(tf_env, learning_rate=3e-4):

    obs_spec = tf_env.observation_spec()
    act_spec = tf_env.action_spec()
    time_step_spec = tf_env.time_step_spec()

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        obs_spec,
        act_spec,
        fc_layer_params=(256, 256),
    )

    value_net = value_network.ValueNetwork(
        obs_spec,
        fc_layer_params=(256, 256),
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

    tf_agent = ppo_agent.PPOAgent(
        time_step_spec=time_step_spec,
        action_spec=act_spec,
        optimizer=optimizer,
        actor_net=actor_net,
        value_net=value_net,
        num_epochs=3,
        use_gae=True,
        lambda_value=0.95,
        discount_factor=0.99,
        entropy_regularization=0.0,
        importance_ratio_clipping=0.2,
        value_clipping=0.2,
        normalize_observations=False,
        normalize_rewards=False,
        gradient_clipping=None,
        train_step_counter=global_step,
    )
    tf_agent.initialize()
    return tf_agent, global_step


def train_robust_policy_ppo(
    base_env,
    perturbation_net: PerturbationNet,
    reward_model: RewardModel,
    epsilon: float = 0.1,
    num_iterations: int = 100_000,
    initial_collect_steps: int = 1000,
    collect_steps_per_iter: int = 1,
    replay_capacity: int = 100_000,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
):


    tf_env = make_robust_tf_env(
        base_env=base_env,
        perturbation_net=perturbation_net,
        reward_model=reward_model,
        epsilon=epsilon,
    )


    tf_agent, global_step = build_ppo_agent(tf_env, learning_rate=learning_rate)


    collect_policy = tf_agent.collect_policy
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_capacity,
    )

    driver = dynamic_step_driver.DynamicStepDriver(
        env=tf_env,
        policy=collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=collect_steps_per_iter,
    )


    tf_agent.train = common.function(tf_agent.train)
    driver.run = common.function(driver.run)

    time_step = tf_env.reset()
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)


    for _ in range(initial_collect_steps):
        time_step, policy_state = driver.run(time_step, policy_state)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2,   
    ).prefetch(3)
    iterator = iter(dataset)

    for _ in range(num_iterations):

        time_step, policy_state = driver.run(time_step, policy_state)

        experience, _ = next(iterator)
        train_loss = tf_agent.train(experience).loss

        step = global_step.numpy()
        if step % 1000 == 0:
            print(f"[Robust-PPO] step={step}, loss={train_loss.numpy():.4f}")

    return tf_agent.policy