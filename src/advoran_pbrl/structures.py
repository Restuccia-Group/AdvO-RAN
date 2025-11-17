# advoran_pbrl/structures.py
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class StepKPM:
    throughput_bps: float  # for eMBB slice
    latency_s: float       # for URLLC slice


@dataclass
class Transition:
    s: np.ndarray
    a: int
    s_next: np.ndarray
    kpm: StepKPM


@dataclass
class Trajectory:
    steps: List[Transition]
