import os
import datetime as dt
import itertools


feasible_prb_allocation_all = [
    [6, 39, 5], [6, 15, 29], [6, 27, 17], [12, 27, 11], [36, 9, 5], [42, 3, 5],
]

scheduling_combos = [
    [0, 0, 0], [0, 0, 1], [0, 0, 2],
    [1, 0, 0], [1, 0, 1], [1, 0, 2],
    [2, 0, 0], [2, 0, 1], [2, 0, 2],
]

feasible_prb_allocation_indexes = list(range(len(feasible_prb_allocation_all)))
actions = list(itertools.product(feasible_prb_allocation_indexes, scheduling_combos))
n_actions = len(actions)

metric_list_autoencoder = [
    "dl_buffer [bytes]",
    "tx_brate downlink [Mbps]",
    "tx_pkts downlink",
]

autoencoder_input_scale = 10.0

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(PROJECT_ROOT, "dataset", "embb_filtered.csv")
encoder_path = os.path.join(PROJECT_ROOT, "encoder.h5")

du_prb = 50
use_gpu_in_env = False
num_parallel_environments = 4

ppo_actor_fc_layers = (30, 30, 30, 30, 30)
ppo_value_fc_layers = (30, 30, 30, 30, 30)
ppo_learning_rate = 1e-4
ppo_num_epochs = 10
ppo_entropy_regularization = 0.05
ppo_importance_ratio_clipping = 0.2
ppo_normalize_observations = True
ppo_normalize_rewards = True
ppo_use_gae = True
greedy_eval = False

run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
drl_save_folder = os.path.join(PROJECT_ROOT, "saved_policy", "em-max", "em-agent-lp")
checkpoint_dir = os.path.join(drl_save_folder, "checkpoints")
policy_dir = drl_save_folder
log_dir = os.path.join(drl_save_folder, "logs", run_id)

max_train_steps = 2000
log_interval = 50
eval_interval = 50
checkpoint_interval = 100

num_steps_per_episode = 10
collect_episodes_per_iteration = 4
collect_steps_per_iteration = num_steps_per_episode * collect_episodes_per_iteration
train_steps_per_iteration = 1

num_eval_episodes = 1
num_eval_steps_per_episode = 10
num_test_steps_per_episode = 10

debug_summaries = False
summarize_grads_and_vars = False
summaries_flush_secs = 10
summary_interval = 2

use_reward_normalization = True
