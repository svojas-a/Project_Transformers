"""
Dataset loading, isolated behind a narrow interface so swapping data
sources (HF `datasets`, local CSVs, cached tensors) never touches the
runner or metrics code.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class DatasetLoader(ABC):
    @abstractmethod
    def load(self, task_name: str) -> Tuple[List[str], Optional[List[int]]]:
        """Returns (texts, labels). labels is None for tasks without a
        simple classification label (e.g. token-level tasks needing custom
        handling upstream)."""
        raise NotImplementedError


class HuggingFaceDatasetLoader(DatasetLoader):
    """
    Thin adapter over the HF `datasets` library. Task -> (dataset name,
    config, split, text field, label field) mapping lives in one place so
    it's easy to audit/extend for new tasks.
    """

    _TASK_SPECS = {
        "sst2": dict(path="glue", name="sst2", split="validation",
                     text_field="sentence", label_field="label"),
        "mnli": dict(path="glue", name="mnli", split="validation_matched",
                     text_field="premise", label_field="label"),
        # conll2003 is token-level (NER); representation-level collapse
        # metrics here use sentence-pooled reps with NER tag of the first
        # entity as a coarse label proxy -- documented caveat, revisit if
        # NC1 needs true token-level granularity.
        "conll2003": dict(path="conll2003", name=None, split="validation",
                           text_field="tokens", label_field="ner_tags"),
    }

    def load(self, task_name: str) -> Tuple[List[str], Optional[List[int]]]:
        from datasets import load_dataset

        if task_name not in self._TASK_SPECS:
            raise ValueError(
                f"Unknown task '{task_name}'. Available: {list(self._TASK_SPECS)}"
            )
        spec = self._TASK_SPECS[task_name]
        ds = load_dataset(spec["path"], spec["name"], split=spec["split"])

        if task_name == "conll2003":
            texts = [" ".join(row["tokens"]) for row in ds]
            labels = [row["ner_tags"][0] if row["ner_tags"] else 0 for row in ds]
        else:
            texts = [row[spec["text_field"]] for row in ds]
            labels = [row[spec["label_field"]] for row in ds]

        return texts, labels
