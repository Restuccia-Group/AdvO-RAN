# Adv-ORAN

Adversarial and robust PPO workflows for ORAN scheduling.

## What is in this folder
- `train_modular.py`: train base PPO policy and export actor/value/optimizer snapshots.
- `attack_wa.py`: run PGD-based white-box attack evaluation.
- `train_perturbator_policy.py`: train learned perturbation model (`pert.h5`).
- `radial.py`: RADIAL-PPO robust training.
- `sa-ppo.py`: robust PPO training that uses a learned perturbator model.
- `saved_policy/em-max/em-agent-filtered`: current default policy snapshot directory.

## Prerequisites
- Python 3.10+
- Install deps:

```bash
pip install -r requirements.txt
```

### Important path note
`agent_builder.py` is in the parent directory (`../agent_builder.py`), so run commands with:

```bash
PYTHONPATH=..
```

## Quick sanity checks
```bash
PYTHONPATH=.. python train_modular.py --help
PYTHONPATH=.. python attack_wa.py --help
PYTHONPATH=.. python train_perturbator_policy.py --help
PYTHONPATH=.. python radial.py --help
PYTHONPATH=.. python sa-ppo.py --help
```

## Run commands

### 1) Train base policy
```bash
PYTHONPATH=.. python train_modular.py \
  --config_module config_em_filtered \
  --cpu_only
```

Optional fast smoke run:
```bash
PYTHONPATH=.. python train_modular.py \
  --config_module config_em_filtered \
  --max_train_steps 10 \
  --cpu_only
```

### 2) Evaluate / attack policy
Baseline only (no perturbation):
```bash
PYTHONPATH=.. python attack_wa.py \
  --horizon 10 \
  --no_attack
```

PGD attack enabled:
```bash
PYTHONPATH=.. python attack_wa.py \
  --eps 0.3 \
  --alpha 0.015 \
  --iters 20 \
  --horizon 10
```

### 3) Train perturbator (`pert.h5`)
```bash
PYTHONPATH=.. python train_perturbator_policy.py \
  --reward_model reward_model.h5 \
  --collect_steps 1000 \
  --epochs 100 \
  --save_perturb_net pert.h5
```

### 4) Run RADIAL-PPO robust training
```bash
PYTHONPATH=.. python radial.py \
  --policy_dir saved_policy/em-max/em-agent-filtered \
  --out_dir saved_policy/em-max/em-agent-radial
```

### 5) Run SA-PPO with learned perturbator
```bash
PYTHONPATH=.. python sa-ppo.py \
  --perturbator_path pert.h5 \
  --policy_dir saved_policy/em-max/em-agent-filtered \
  --out_dir saved_policy/em-max/em-agent-radial
```

## Outputs
- Base policy snapshots: `actor.npz`, `value.npz`, `optimizer.npz`, `metadata.json` under `saved_policy/em-max/em-agent-filtered`
- Robust outputs: `actor.npz`, `actor_best.npz`, `train_logs.npz`, `metadata.json` under `saved_policy/em-max/em-agent-radial`
- Perturbator model: `pert.h5` (or your custom `--save_perturb_net` path)


