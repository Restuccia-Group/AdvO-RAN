import tensorflow as tf

from tf_agents.agents.sac import sac_agent
from tf_agents.networks import actor_distribution_network, critic_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.utils import common

from .adv_env import make_adv_tf_env
from .reward_model import RewardModel


def build_sac_agent(tf_env, learning_rate=3e-4):
    observation_spec = tf_env.observation_spec()
    action_spec = tf_env.action_spec()
    time_step_spec = tf_env.time_step_spec()

    
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=(256, 256),
    )

   
    critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        joint_fc_layer_params=(256, 256),
    )
    critic_net_2 = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        joint_fc_layer_params=(256, 256),
    )

    actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate)
    alpha_optimizer = tf.keras.optimizers.Adam(learning_rate)

    global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

    tf_agent = sac_agent.SacAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        critic_network_2=critic_net_2,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        alpha_optimizer=alpha_optimizer,
        target_update_tau=0.005,
        target_update_period=1,
        gamma=0.99,
        reward_scale_factor=1.0,
        train_step_counter=global_step,
    )
    tf_agent.initialize()
    return tf_agent, global_step


def train_adversary_sac(
    reward_model: RewardModel,
    base_env,                   
    victim_policy,              
    num_iterations: int = 100_000,
    initial_collect_steps: int = 1000,
    collect_steps_per_iter: int = 1,
    replay_capacity: int = 100_000,
    batch_size: int = 256,
):

    tf_env = make_adv_tf_env(
        base_env=base_env,
        victim_policy=victim_policy,
        reward_model=reward_model,
        epsilon=0.05,
        lambda_pen=10.0,
    )

    tf_agent, global_step = build_sac_agent(tf_env)
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
            print(f"[AO-SAC] step={step}, loss={train_loss.numpy():.4f}")

    
    return tf_agent.policy