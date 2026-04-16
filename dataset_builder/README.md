# Dataset Builder

This folder documents the 2-step dataset build flow.

## Step 1: Build the raw dataset (`create_raw_dataset.py`)

First download dataset from https://github.com/wineslab/colosseum-oran-coloran-dataset.git

```bash
python create_raw_dataset.py \
  --root_dir colosseum-oran-coloran-dataset/rome_static_medium \
  --output_csv dataset/embb.csv \
  --reward_preset embb \
  --reward_norm zero_one \
  --normalize_metrics \
  --metric_scale 10 \
  --add_ratio_granted_req \
  --duplicate_norm_cols \
  --slices 0 1 2
```

Notes:
- Output of Step 1 in this example is `dataset/embb.csv`.

## Step 2: Filter by action combos (`filter_by_action_combos.py`)

Filter the raw dataset using allowed action combinations from `config_em_filtered.py`:

```bash
python dataset/filter.py \
  --input dataset/embb.csv \
  --output dataset/embb_filtered.csv
```

This keeps only rows where `(slice_id, slice_prb, scheduling_policy)` is allowed by:
- `feasible_prb_allocation_all`
- `scheduling_combos`

from `config_em_filtered.py`.
