# AdvO-RAN: Adversarial Deep Reinforcement Learning in AI-Driven Open Radio Access Networks

Adversarial and robust PPO workflows for AdvORAN.

Here we have codes for train an EM-MAX agent for traffic profile 1, train an agent robustly with the help of adversarial learned perturbator in the training loop. 


This repo `saved_policy/em-max/em-agent-lp` as the default victim policy.

## Folder Layout

| Path | Purpose |
| --- | --- |
| `dataset/` | Input CSV files used by the environment and reward-model training. Download and Save csv files under dataset folder [Download](https://drive.google.com/drive/folders/1xFQk5u293b_YNYp51WvM4NbRQCJFzA5O?usp=sharing)| | 
| `saved_policy/em-max/em-agent-lp/` | Victim PPO policy snapshots (`actor.npz`, `value.npz`, `optimizer.npz`) |
| `saved_policy/em-max/em-adversarial-agent/` | Reference adversarial policy used by the perturbator workflow |
| `saved_policy/em-max/em-agent-lp-robust/` | Default output folder for robust PPO training |
| `requirements/` | Dependency manifests (`current.txt` and `legacy-tf23.txt`) |
| `utils/` | Small helper scripts such as `run_actor_npz.py` |



## Active Scripts

| Script | Role |
| --- | --- |
| `train_modular.py` | Train or refresh the victim PPO policy |
| `reward_model.py` | Train `reward_model.h5` from a CSV file |
| `training_adversarial_policy.py` | Train the adversarial/reference PPO policy using `reward_model.h5` |
| `train_perturbator_policy.py` | Train the observation perturbator and save `pert.h5` |
| `train_robust_policy.py` | Train the robust PPO policy starting from `em-agent-lp` |
| `attack_wa.py` | Run a PGD-style attack against the victim actor |
| `evaluate_action_net.py` | Inspect the victim actor/action net behavior |
| `evaluate_perturbator_effect.py` | Measure how `pert.h5` changes the victim policy |
| `utils/run_actor_npz.py` | Step through a saved `actor.npz` snapshot in the environment |

## Dependencies

Two dependency tracks are kept on purpose:

- `requirements/current.txt`: current loose stack for the newer code path
- `requirements/legacy-tf23.txt`: reproducible legacy stack for the TensorFlow 2.3 experiments and saved checkpoints

### Important Compatibility Note

The exact package set below is **not** installable as written:

```text
cloudpickle==1.4.1
numpy<1.19.0
pandas
scipy==1.4.1
setuptools>=41.0.0
tensorflow==2.3
tensorflow-probability==0.7
tf_agents
```

Use `requirements/legacy-tf23.txt` instead. The corrected legacy stack is:

- Python `3.8.x`
- `tensorflow==2.3.0`
- `tensorflow-probability==0.11.0`
- `tf-agents==0.6.0`

The repo already includes this corrected stack in `requirements/legacy-tf23.txt`.

## Step-By-Step Run Procedure

Run every command below from the repo root.
No extra `PYTHONPATH` setup is required.

### 1. Create the environment

For the legacy TensorFlow 2.3 workflow:

```bash
python3.8 -m venv .venv-tf23
source .venv-tf23/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/legacy-tf23.txt
```

If you want the current loose stack instead:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Verify the required local assets

Make sure these files exist before running training or evaluation:

```bash
ls dataset/embb_filtered.csv
ls dataset/colosseum_oran_coloran_ue2_003.csv
ls encoder.h5
ls saved_policy/em-max/em-agent-lp/actor.npz
ls saved_policy/em-max/em-agent-lp/value.npz
ls reward_model.h5
ls pert.h5
```

Notes:

- `dataset/embb_filtered.csv` is the environment dataset used by `config_em_filtered.py`
- `dataset/colosseum_oran_coloran_ue2_003.csv` is the CSV used to rebuild the reward model
- `saved_policy/em-max/em-agent-lp/` is the default victim policy directory

### 3. Sanity-check the victim agent

Run the saved victim actor directly:

```bash
python utils/run_actor_npz.py \
  saved_policy/em-max/em-agent-lp/actor.npz \
  --steps 10 \
  --action_mode greedy
```

Inspect the action net in more detail:

```bash
python evaluate_action_net.py \
  saved_policy/em-max/em-agent-lp \
  --collect_steps 200 \
  --eval_episodes 5 \
  --eval_max_steps 10
```

### 4. Run attacks and perturbation evaluation

Worst action attack against the victim policy:

```bash
python attack_wa.py \
  --policy_dir saved_policy/em-max/em-agent-lp \
  --eps 0.3 \
  --alpha 0.015 \
  --iters 20 \
  --horizon 10
```

Evaluate the learned perturbator on the victim policy:

```bash
python evaluate_perturbator_effect.py \
  --policy_dir saved_policy/em-max/em-agent-lp \
  --perturbator_path pert.h5 \
  --collect_steps 200 \
  --eval_episodes 5 \
  --eval_max_steps 10
```

### 5. 

#### 5.1 Train or refresh the victim policy

This writes canonical snapshots into `saved_policy/em-max/em-agent-lp/`.

```bash
python train_modular.py --cpu_only
```

#### 5.2 Train the reward model

```bash
python reward_model.py \
  --csv dataset/colosseum_oran_coloran_ue2_003.csv \
  --out_model reward_model.h5
```

#### 5.3 Train the adversarial/reference policy

This writes to `saved_policy/em-max/em-adversarial-agent/`.

```bash
python training_adversarial_policy.py \
  --reward_model_path reward_model.h5 \
  --cpu_only
```

#### 5.4 Train the perturbator

Victim policy = `em-agent-lp`, reference policy = `em-adversarial-agent`.

```bash
python train_perturbator_policy.py \
  --policy_dir saved_policy/em-max/em-agent-lp \
  --ref_policy_dir saved_policy/em-max/em-adversarial-agent \
  --reward_model reward_model.h5 \
  --save_perturb_net pert.h5
```

#### 5.5 Train the robust policy

This starts from `saved_policy/em-max/em-agent-lp/` and writes to `saved_policy/em-max/em-agent-lp-robust/`.

```bash
python train_robust_policy.py \
  --policy_dir saved_policy/em-max/em-agent-lp \
  --perturbator_path pert.h5 \
  --reward_model_path reward_model.h5 \
  --cpu_only
```

### 6. Evaluate the final outputs

Evaluate the original victim again:

```bash
python evaluate_action_net.py saved_policy/em-max/em-agent-lp
```

Evaluate the robust policy:

```bash
python evaluate_action_net.py saved_policy/em-max/em-agent-lp-robust
```

Compare the perturbator effect on the victim:

```bash
python evaluate_perturbator_effect.py \
  --policy_dir saved_policy/em-max/em-agent-lp \
  --perturbator_path pert.h5
```

## Output Summary

| Output | Default location |
| --- | --- |
| Victim PPO snapshots | `saved_policy/em-max/em-agent-lp/` |
| Adversarial/reference PPO snapshots | `saved_policy/em-max/em-adversarial-agent/` |
| Robust PPO snapshots | `saved_policy/em-max/em-agent-lp-robust/` |
| Reward model | `reward_model.h5` |
| Perturbator | `pert.h5` |

## Short Workflow Guide

If you already trust the bundled victim policy and only want to run the attack pipeline, use this order:

1. Create the Python 3.8 environment with `requirements/legacy-tf23.txt`
2. Run `utils/run_actor_npz.py` or `evaluate_action_net.py` on `saved_policy/em-max/em-agent-lp`
3. Run `attack_wa.py`
4. Run `evaluate_perturbator_effect.py`

If you want to rebuild everything from scratch, use this order:

1. `train_modular.py`
2. `reward_model.py`
3. `training_adversarial_policy.py`
4. `train_perturbator_policy.py`
5. `train_robust_policy.py`
