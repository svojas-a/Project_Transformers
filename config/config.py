"""
Configuration objects for the dimensional-collapse characterization pipeline.

Kept as plain dataclasses so config is data, not behavior — anything that
needs to change per-experiment lives here, not scattered through the code
(Single Responsibility: this module's only job is to describe a run).
"""
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class DimensionSchedule:
    """
    Generates the sequence of hidden-dimension sizes to evaluate.

    Frozen/immutable so a schedule can't be mutated mid-experiment by
    accident once it's handed to the runner.
    """
    base_dim: int = 768            # DistilBERT default hidden size
    reduction_factor: float = 0.8  # keep 80% each step -> 20% reduction
    num_steps: int = 7
    round_to: int = 12             # DistilBERT n_heads; every dim must be
                                    # divisible by this for attention head
                                    # splitting to work (dim // n_heads)

    def generate(self) -> List[int]:
        dims = []
        d = self.base_dim
        for _ in range(self.num_steps):
            rounded = self._round_down(d)
            if rounded > 0:
                dims.append(rounded)
            d = d * self.reduction_factor
        # de-duplicate while preserving order, guard against rounding collisions
        seen = set()
        ordered = []
        for d in dims:
            if d not in seen:
                seen.add(d)
                ordered.append(d)
        return ordered

    def _round_down(self, d: float) -> int:
        if self.round_to <= 1:
            return int(round(d))
        return max(self.round_to, int(d // self.round_to) * self.round_to)


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str = "distilbert-base-uncased"
    seeds: List[int] = field(default_factory=lambda: [13, 42, 2024])
    tasks: List[str] = field(default_factory=lambda: ["sst2", "mnli", "conll2003"])
    dimension_schedule: DimensionSchedule = field(default_factory=DimensionSchedule)
    reduction_strategy_name: str = "truncation"   # matches ReductionStrategyFactory keys
    max_samples_for_metrics: int = 500          # must stay below dataset size for seed variance to work
    batch_size: int = 32
    output_dir: str = "./results"
    device: str = "cpu"
