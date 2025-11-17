# advoran_pbrl/sla.py
from dataclasses import dataclass
from typing import List
import numpy as np

from .structures import Trajectory


@dataclass
class SLAEvaluator:

    target_embb_bps: float
    bound_urllc_s: float
    sla_factor: float = 0.7
    demand_mu: float = 0.7
    demand_sigma: float = 0.2
    seed: int = 123

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def _sample_factor(self) -> float:
        f = self._rng.normal(self.demand_mu, self.demand_sigma)
        return float(np.clip(f, 0.0, 2.0))

    def violation_flag(
        self,
        slice_type: str,
        throughput_bps: float | None = None,
        latency_s: float | None = None,
        dynamic: bool = True,
    ) -> int:

        s = slice_type.lower()
        factor = self._sample_factor() if dynamic else 1.0

        if s == "embb":
    
            target = factor * self.target_embb_bps
            threshold = self.sla_factor * target
            return int(throughput_bps < threshold)

        if s == "urllc":
    
            bound = factor * self.bound_urllc_s
            threshold = self.sla_factor * bound
            return int(latency_s > threshold)

        raise ValueError(f"Unknown slice_type: {slice_type}")


def compute_vsla(
    traj: Trajectory,
    sla: SLAEvaluator,
    slice_type: str = "embb",
    dynamic: bool = True,
) -> float:

    flags: List[int] = []
    for step in traj.steps:
        if slice_type.lower() == "embb":
            v = sla.violation_flag(
                "embb",
                throughput_bps=step.kpm.throughput_bps,
                dynamic=dynamic,
            )
        else:
            v = sla.violation_flag(
                "urllc",
                latency_s=step.kpm.latency_s,
                dynamic=dynamic,
            )
        flags.append(v)

    return float(np.mean(flags)) if flags else 0.0
