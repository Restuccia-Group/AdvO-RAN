import os
import tensorflow as tf

from RanEnv import RanEnv
from advoran_pbrl.reward_model import RewardModel
from advoran_pbrl.perturbation_net import PerturbationNet
from advoran_pbrl.robust_train import train_robust_policy_ppo


def main():

    base_env = RanEnv()  

    reward_model = tf.keras.models.load_model(
        "saved_models/advoran_reward_model",
        custom_objects={"RewardModel": RewardModel},
    )


    perturbation_net = tf.keras.models.load_model(
        "saved_models/perturbation_net",
        custom_objects={"PerturbationNet": PerturbationNet},
    )


    robust_policy = train_robust_policy_ppo(
        base_env=base_env,
        perturbation_net=perturbation_net,
        reward_model=reward_model,
        epsilon=0.1,
        num_iterations=50_000,
        initial_collect_steps=1000,
        collect_steps_per_iter=1,
        replay_capacity=100_000,
        batch_size=256,
        learning_rate=3e-4,
    )


    os.makedirs("saved_models", exist_ok=True)
    tf.saved_model.save(robust_policy, "saved_models/robust_policy_ppo")


if __name__ == "__main__":
    tf.config.run_functions_eagerly(False)
    main()