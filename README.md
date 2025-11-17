# AdvO-RAN: Adversarial Deep Reinforcement Learning in AI-Driven Open Radio Access Networks

This repository contains the training code for **AdvO-RAN**, which learns robust xApp policies in three main stages:

1. **Train a reward model** from SLA violations (preference-based RL).
2. **Train an adversarial perturbation function** that attacks the victim policy.
3. **Train a robust policy** against the fixed adversary.

0. Install Dependencies

From the repository root:

pip install -r requirements.txt

1. Train the Reward Model

First, learn the PbRL reward model that maps state–action pairs to a scalar reward based on SLA violation patterns.

From the src directory:

cd src
python advoran_pbrl/train_reward_model.py

2. Train the Adversarial Perturbation Function 
Next, train the adversarial perturbation function that generates state-dependent perturbations

cd src 
python advoran_pbrl/train_ao_policy.py

3. Train the Robust Policy (Algorithm 2)

Finally, train a robust PPO policy against the fixed adversary
cd src 
python advoran_pbrl/train_robust_ppo_agent.py

For colosseum testing take the pretrained models and load following the instruction of the [Colosseum Near-Real-Time RIC](https://github.com/wineslab/colosseum-oran-commag-dataset.git) in the xApp codebase

