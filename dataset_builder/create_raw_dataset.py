#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


METRIC_COLS = [
    "tx_brate downlink [Mbps]",
    "tx_pkts downlink",
    "dl_buffer [bytes]",
]

BASE_OUTPUT_COLS = [
    "slice_id",
    "slice_prb",
    "scheduling_policy",
    "tx_brate downlink [Mbps]",
    "tx_pkts downlink",
    "dl_buffer [bytes]",
    "ratio_granted_req",
    "reward",
]

REWARD_PRESETS: Dict[str, Dict[int, np.ndarray]] = {
    "embb": {
        0: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        1: np.array([0.0, 0.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 0.0, 0.0], dtype=np.float32),
    },
    "mtc": {
        0: np.array([0.0, 0.0, 0.0], dtype=np.float32),
        1: np.array([0.0, 1.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 0.0, 0.0], dtype=np.float32),
    },
    "urllc": {
        0: np.array([0.0, 0.0, 0.0], dtype=np.float32),
        1: np.array([0.0, 0.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 0.0, -1.0], dtype=np.float32),
    },
    "mixed": {
        0: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        1: np.array([0.0, 1.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 0.0, -1.0], dtype=np.float32),
    },
}


def find_metrics_csvs(root_dir: str, suffix: str = "_metrics.csv") -> List[str]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    paths = sorted(str(p) for p in root.rglob(f"*{suffix}") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No files ending with '{suffix}' found under: {root_dir}")
    return paths


def parse_vec3(text: str) -> np.ndarray:
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got: {text}")
    return np.asarray(parts, dtype=np.float32)


def get_weights(args) -> Dict[int, np.ndarray]:
    if args.reward_preset != "custom":
        return REWARD_PRESETS[args.reward_preset]

    if args.custom_weights_json:
        data = json.loads(args.custom_weights_json)
        out = {}
        for k, v in data.items():
            sid = int(k)
            if len(v) != 3:
                raise ValueError(f"Custom weight for slice {sid} must have 3 values")
            out[sid] = np.asarray(v, dtype=np.float32)
        for sid in [0, 1, 2]:
            out.setdefault(sid, np.zeros(3, dtype=np.float32))
        return out

    if not (args.w0 and args.w1 and args.w2):
        raise ValueError("For custom mode, provide --w0 --w1 --w2 or --custom_weights_json")

    return {
        0: parse_vec3(args.w0),
        1: parse_vec3(args.w1),
        2: parse_vec3(args.w2),
    }


def iter_source_chunks(source_files: Sequence[str], chunksize: int) -> Iterable[pd.DataFrame]:
    for src in source_files:
        for chunk in pd.read_csv(src, chunksize=chunksize):
            yield chunk


def compute_raw_reward(df: pd.DataFrame, weights: Dict[int, np.ndarray]) -> np.ndarray:
    x = df[METRIC_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    rewards = np.zeros(len(df), dtype=np.float32)
    slice_ids = pd.to_numeric(df["slice_id"], errors="coerce").fillna(-1).to_numpy(dtype=np.int32)

    for sid, w in weights.items():
        mask = slice_ids == int(sid)
        if np.any(mask):
            rewards[mask] = x[mask] @ w
    return rewards


def apply_reward_norm(values: np.ndarray, vmin: float, vmax: float, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if mode == "none":
        return values

    denom = float(vmax - vmin)
    if denom == 0.0:
        return np.zeros_like(values, dtype=np.float32)

    z = (values - float(vmin)) / denom
    z = np.clip(z, 0.0, 1.0)

    if mode == "zero_one":
        return z.astype(np.float32)
    if mode == "neg_one_one":
        return (z * 2.0 - 1.0).astype(np.float32)
    raise ValueError(f"Unknown reward norm mode: {mode}")


def add_ratio_granted_req(df: pd.DataFrame, replace_zero_with_one: bool) -> pd.DataFrame:
    req = "sum_requested_prbs"
    grd = "sum_granted_prbs"
    if req not in df.columns or grd not in df.columns:
        df["ratio_granted_req"] = 0.0
        return df

    denom = pd.to_numeric(df[req], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    numer = pd.to_numeric(df[grd], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    ratio = np.nan_to_num(numer / np.where(denom == 0.0, 1.0, denom), nan=0.0, posinf=0.0, neginf=0.0)
    ratio = np.clip(ratio, 0.0, 1.0)
    if replace_zero_with_one:
        ratio[denom <= 0.0] = 1.0
    df["ratio_granted_req"] = ratio.astype(np.float32)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified dataset generator")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_csv", type=str, help="Single source CSV")
    src.add_argument("--root_dir", type=str, help="Root directory to scan recursively for *_metrics.csv")

    parser.add_argument("--suffix", type=str, default="_metrics.csv", help="Suffix used with --root_dir")
    parser.add_argument("--output_csv", type=str, required=True, help="Output dataset CSV path")
    parser.add_argument("--chunksize", type=int, default=250000)

    parser.add_argument("--slices", nargs="*", type=int, default=[0, 1, 2], help="Slices to include")

    parser.add_argument("--reward_preset", choices=["embb", "mtc", "urllc", "mixed", "custom"], default="urllc")
    parser.add_argument("--custom_weights_json", type=str, default="", help='JSON like {"0":[1,0,0],"1":[0,1,0],"2":[0,0,-1]}')
    parser.add_argument("--w0", type=str, default="", help="Custom slice-0 weights, e.g. 1,0,0")
    parser.add_argument("--w1", type=str, default="", help="Custom slice-1 weights, e.g. 0,1,0")
    parser.add_argument("--w2", type=str, default="", help="Custom slice-2 weights, e.g. 0,0,1")
    parser.add_argument("--reward_norm", choices=["none", "zero_one", "neg_one_one"], default="zero_one")

    parser.add_argument("--normalize_metrics", action="store_true", help="Min-max normalize metric cols globally")
    parser.add_argument("--metric_scale", type=float, default=10.0, help="Scale factor after metric normalization")

    parser.add_argument("--add_ratio_granted_req", action="store_true")
    parser.add_argument("--replace_zero_req_with_one", action="store_true")

    parser.add_argument("--duplicate_norm_cols", action="store_true", help="Create slice_prb_norm/scheduling_policy_norm")

    args = parser.parse_args()

    output_csv = str(Path(args.output_csv).resolve())
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    source_files = [str(Path(args.input_csv).resolve())] if args.input_csv else find_metrics_csvs(args.root_dir, args.suffix)
    weights = get_weights(args)
    selected_slices = set(int(s) for s in args.slices)

    metric_bounds = {c: [float("inf"), float("-inf")] for c in METRIC_COLS}
    reward_bounds = {sid: [float("inf"), float("-inf")] for sid in [0, 1, 2]}
    prb_bounds = [float("inf"), float("-inf")]
    sched_bounds = [float("inf"), float("-inf")]

    # Pass 1: global bounds.
    for chunk in iter_source_chunks(source_files, args.chunksize):
        if "slice_id" not in chunk.columns:
            continue
        chunk = chunk[chunk["slice_id"].isin(selected_slices)].copy()
        if chunk.empty:
            continue

        for col in METRIC_COLS:
            if col not in chunk.columns:
                raise ValueError(f"Missing required metric column: {col}")
            vals = pd.to_numeric(chunk[col], errors="coerce").to_numpy(dtype=np.float32)
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            metric_bounds[col][0] = min(metric_bounds[col][0], float(vals.min()))
            metric_bounds[col][1] = max(metric_bounds[col][1], float(vals.max()))

        raw_reward = compute_raw_reward(chunk, weights)
        sid = pd.to_numeric(chunk["slice_id"], errors="coerce").fillna(-1).to_numpy(dtype=np.int32)
        for s in [0, 1, 2]:
            mask = sid == s
            if np.any(mask):
                reward_bounds[s][0] = min(reward_bounds[s][0], float(raw_reward[mask].min()))
                reward_bounds[s][1] = max(reward_bounds[s][1], float(raw_reward[mask].max()))

        if args.duplicate_norm_cols:
            for col, bounds in [("slice_prb", prb_bounds), ("scheduling_policy", sched_bounds)]:
                if col in chunk.columns:
                    vals = pd.to_numeric(chunk[col], errors="coerce").to_numpy(dtype=np.float32)
                    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                    bounds[0] = min(bounds[0], float(vals.min()))
                    bounds[1] = max(bounds[1], float(vals.max()))

    def norm_series(vals: np.ndarray, lo: float, hi: float) -> np.ndarray:
        den = hi - lo
        if den == 0.0:
            den = 1.0
        return np.clip((vals - lo) / den, 0.0, 1.0).astype(np.float32)

    # Pass 2: process and write.
    first_write = True
    out_cols = list(BASE_OUTPUT_COLS)
    if args.duplicate_norm_cols:
        out_cols = [
            "slice_id", "slice_prb", "slice_prb_norm", "scheduling_policy", "scheduling_policy_norm",
            "tx_brate downlink [Mbps]", "tx_pkts downlink", "dl_buffer [bytes]", "ratio_granted_req", "reward",
        ]

    for chunk in iter_source_chunks(source_files, args.chunksize):
        if "slice_id" not in chunk.columns:
            continue
        chunk = chunk[chunk["slice_id"].isin(selected_slices)].copy()
        if chunk.empty:
            continue

        if args.add_ratio_granted_req:
            chunk = add_ratio_granted_req(chunk, args.replace_zero_req_with_one)
        elif "ratio_granted_req" not in chunk.columns:
            chunk["ratio_granted_req"] = 0.0

        if args.normalize_metrics:
            for col in METRIC_COLS:
                vals = pd.to_numeric(chunk[col], errors="coerce").to_numpy(dtype=np.float32)
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                vals = norm_series(vals, metric_bounds[col][0], metric_bounds[col][1]) * float(args.metric_scale)
                chunk[col] = vals.astype(np.float32)

        raw_reward = compute_raw_reward(chunk, weights)
        sid = pd.to_numeric(chunk["slice_id"], errors="coerce").fillna(-1).to_numpy(dtype=np.int32)
        reward = np.zeros(len(chunk), dtype=np.float32)
        for s in [0, 1, 2]:
            mask = sid == s
            if np.any(mask):
                lo, hi = reward_bounds[s]
                reward[mask] = apply_reward_norm(raw_reward[mask], lo, hi, args.reward_norm)
        chunk["reward"] = reward

        if args.duplicate_norm_cols:
            if "slice_prb" in chunk.columns:
                v = pd.to_numeric(chunk["slice_prb"], errors="coerce").to_numpy(dtype=np.float32)
                chunk["slice_prb_norm"] = norm_series(v, prb_bounds[0], prb_bounds[1])
            else:
                chunk["slice_prb_norm"] = 0.0

            if "scheduling_policy" in chunk.columns:
                v = pd.to_numeric(chunk["scheduling_policy"], errors="coerce").to_numpy(dtype=np.float32)
                chunk["scheduling_policy_norm"] = norm_series(v, sched_bounds[0], sched_bounds[1])
            else:
                chunk["scheduling_policy_norm"] = 0.0

        missing = [c for c in out_cols if c not in chunk.columns]
        if missing:
            raise ValueError(f"Missing output columns after processing: {missing}")

        chunk[out_cols].to_csv(output_csv, mode="w" if first_write else "a", header=first_write, index=False)
        first_write = False

    # Tiny summary (fast) from head.
    with open(output_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        vals = []
        for i, row in enumerate(r):
            try:
                vals.append(float(row["reward"]))
            except Exception:
                pass
            if i >= 5000:
                break

    if vals:
        arr = np.asarray(vals, dtype=np.float32)
        print(f"[OK] wrote: {output_csv}")
        print(f"reward sample stats -> min={arr.min():.4f} mean={arr.mean():.4f} max={arr.max():.4f}")
    else:
        print(f"[OK] wrote: {output_csv}")


if __name__ == "__main__":
    main()
