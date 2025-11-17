import os
import tensorflow as tf

from advoran_pbrl.sla import SLAEvaluator
from advoran_pbrl.collect_trajectories import collect_trajectories
from advoran_pbrl.reward_model import (
    build_preference_pairs,
    train_reward_model_with_preferences,
)

from RanEnv import RanEnv 


def main():
    
    NUM_ACTIONS = 15                           
    VICTIM_POLICY_DIR = "saved_policies/victim_policy"  

    base_env = RanEnv()   


    victim_policy = tf.saved_model.load(VICTIM_POLICY_DIR)

   
    trajectories = collect_trajectories(
        base_env=base_env,
        victim_policy=victim_policy,
        num_episodes=100,          
    )

    state_dim = trajectories[0].steps[0].s.shape[0]


    sla = SLAEvaluator(
        target_embb_bps=700_000,
        bound_urllc_s=0.010,
        sla_factor=0.7,
        demand_mu=0.7,
        demand_sigma=0.2,
    )


    pairs = build_preference_pairs(
        trajectories,
        sla=sla,
        slice_type="embb",
        dynamic=True,
        min_vsla_diff=0.05,
    )

   
    reward_model = train_reward_model_with_preferences(
        pairs=pairs,
        state_dim=state_dim,
        num_actions=NUM_ACTIONS,
        seq_len=10,        
        batch_size=16,
        hidden_dim=128,
        num_epochs=30,
        lr=1e-3,
    )

    os.makedirs("saved_models", exist_ok=True)
    reward_model.save(os.path.join("saved_models", "advoran_reward_model"))


if __name__ == "__main__":
    tf.config.run_functions_eagerly(True) 
    main()