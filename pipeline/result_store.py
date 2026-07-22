"""
ResultStore: owns reading/writing results to disk. Isolated so the runner
doesn't know or care whether results end up in JSON, a database, or cloud
storage (Single Responsibility + easy to swap later).
"""
import json
import os
from typing import Dict, List, Any


class ResultStore:
    def __init__(self, output_dir: str):
        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)
        self._records: List[Dict[str, Any]] = []

    def add_record(
        self,
        task: str,
        seed: int,
        hidden_dim: int,
        metrics: Dict[str, Any],
        task_performance: float = None,
    ) -> None:
        self._records.append({
            "task": task,
            "seed": seed,
            "hidden_dim": hidden_dim,
            "metrics": metrics,
            "task_performance": task_performance,
        })

    def save(self, filename: str = "phase1_results.json") -> str:
        path = os.path.join(self._output_dir, filename)
        with open(path, "w") as f:
            json.dump(self._records, f, indent=2, default=str)
        return path

    def as_records(self) -> List[Dict[str, Any]]:
        return self._records

    def to_dataframe(self):
        """Flattens nested metric dicts into a tidy DataFrame for analysis/plotting."""
        import pandas as pd
        rows = []
        for rec in self._records:
            row = {
                "task": rec["task"],
                "seed": rec["seed"],
                "hidden_dim": rec["hidden_dim"],
                "task_performance": rec["task_performance"],
            }
            for metric_name, value in rec["metrics"].items():
                if isinstance(value, dict):
                    if "error" in value:
                        continue
                    for sub_k, sub_v in value.items():
                        row[f"{metric_name}_{sub_k}"] = sub_v
                else:
                    row[metric_name] = value
            rows.append(row)
        return pd.DataFrame(rows)
