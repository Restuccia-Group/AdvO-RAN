import os
import tensorflow as tf

from advoran_pbrl.reward_model import RewardModel
from advoran_pbrl.sac_train import train_adversary_sac

from RanEnv import RanEnv   


def main():
   
    VICTIM_POLICY_DIR = "saved_policies/victim_policy"
    REWARD_MODEL_DIR = "saved_models/advoran_reward_model"
  

    
    base_env = RanEnv()  

    
    reward_model = tf.keras.models.load_model(
        REWARD_MODEL_DIR,
        custom_objects={"RewardModel": RewardModel},
    )

    
    victim_policy = tf.saved_model.load(VICTIM_POLICY_DIR)

    
    ao_policy = train_adversary_sac(
        reward_model=reward_model,
        base_env=base_env,
        victim_policy=victim_policy,
        num_iterations=50_000,        # tune
        initial_collect_steps=1000,
        collect_steps_per_iter=1,
        replay_capacity=100_000,
        batch_size=256,
    )

   
    os.makedirs("saved_models", exist_ok=True)
    tf.saved_model.save(ao_policy, "saved_models/advoran_ao_sac_policy")


if __name__ == "__main__":
    tf.config.run_functions_eagerly(False)
    main()