"""
Tests for src/analysis/finetune_task.py. Offline: tiny random DistilBERT, a
local vocab file and hand-written rows in the real SST-2 / MNLI / CoNLL format.
The end-to-end test fine-tunes the tiny model and then runs the backward
restoration script on the checkpoint it produced.
"""

import json
import os

import numpy as np
import pytest
import torch
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
    DistilBertTokenizerFast,
)

from src.analysis import backward_restoration as br
from src.analysis import finetune_task as ft

WORDS = (
    "the movie was very good bad a an film plot actor not great terrible and . , "
    "eu rejects german call to boycott british lamb ##ing"
).split()

SST2 = [
    {"sentence": s, "label": i % 2}
    for i, s in enumerate(
        [
            "the movie was very good",
            "not a great film",
            "a bad bad film",
            "the plot and the actor were very good",
            "the film was not very great",
            "very good actor , good plot",
            "the movie was terrible and bad",
            "a very good film",
        ]
    )
]
MNLI = [
    {"premise": p, "hypothesis": h, "label": i % 3}
    for i, (p, h) in enumerate(
        [
            ("the movie was very good", "the film was great"),
            ("a bad film", "the plot was not good"),
            ("the actor was great", "the actor was terrible"),
            ("the plot and the film", "a good film"),
            ("not a great movie", "the movie was bad"),
            ("very good actor", "not a good actor"),
            ("the film was bad", "the film was very bad"),
            ("a good plot", "a bad plot"),
        ]
    )
]
CONLL = [
    {"tokens": ["-DOCSTART-"], "ner_tags": [0]},
    {
        "tokens": ["eu", "rejects", "german", "call", "to", "boycott"],
        "ner_tags": [3, 0, 7, 0, 0, 0],
    },
    {"tokens": ["british", "lamb", "was", "good"], "ner_tags": [7, 0, 0, 0]},
    {
        "tokens": ["the", "german", "film", "and", "the", "actor"],
        "ner_tags": [0, 7, 0, 0, 0, 0],
    },
    {
        "tokens": ["eu", "call", "to", "boycotting", "british", "lamb"],
        "ner_tags": [3, 0, 0, 0, 7, 0],
    },
    {
        "tokens": ["a", "bad", "german", "film", "not", "good"],
        "ner_tags": [0, 0, 7, 0, 0, 0],
    },
    {"tokens": ["the", "british", "actor", "was", "good"], "ner_tags": [0, 7, 0, 0, 0]},
    {"tokens": ["a", "good", "film", "and", "plot"], "ner_tags": [0, 0, 0, 0, 0]},
]
ROWS = {"sst2": SST2, "mnli": MNLI, "conll2003": CONLL}


@pytest.fixture()
def tokenizer(tmp_path):
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS))
    return DistilBertTokenizerFast(vocab_file=str(path), do_lower_case=True)


def _tiny(task):
    torch.manual_seed(0)
    cfg = DistilBertConfig(
        dim=64,
        hidden_dim=128,
        n_heads=4,
        n_layers=6,
        vocab_size=64,
        num_labels=ft.NUM_LABELS[task],
    )
    cls = (
        DistilBertForTokenClassification
        if task == "conll2003"
        else DistilBertForSequenceClassification
    )
    return cls(cfg)


def test_label_counts_match_the_datasets():
    assert ft.NUM_LABELS == {"sst2": 2, "mnli": 3, "conll2003": 9}


def test_token_labels_sit_on_first_subword_only(tokenizer):
    spec = br.TASKS["conll2003"]
    feats = ft.build_features(tokenizer, spec, [CONLL[4]], None, 0)
    labels = feats[0]["labels"]
    # [CLS] eu call to boycott ##ing british lamb [SEP]
    assert labels[0] == ft.IGNORE_INDEX and labels[-1] == ft.IGNORE_INDEX
    assert labels[1] == 3  # "eu"
    boycott = tokenizer.convert_ids_to_tokens(feats[0]["input_ids"]).index("##ing")
    assert labels[boycott] == ft.IGNORE_INDEX  # continuation subword
    assert labels[boycott - 1] == 0  # first subword carries the tag
    assert sum(x != ft.IGNORE_INDEX for x in labels) == len(CONLL[4]["tokens"])


def test_docstart_rows_are_dropped_and_max_samples_respected(tokenizer):
    spec = br.TASKS["conll2003"]
    assert len(ft.build_features(tokenizer, spec, CONLL, None, 0)) == len(CONLL) - 1
    assert len(ft.build_features(tokenizer, spec, CONLL, 3, 0)) == 3


def test_collate_pads_inputs_and_token_labels(tokenizer):
    spec = br.TASKS["conll2003"]
    feats = ft.build_features(tokenizer, spec, CONLL[1:3], None, 0)
    batch = ft.collate(tokenizer, feats, is_token=True)
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["labels"][1, -1] == ft.IGNORE_INDEX  # padded tail ignored


def test_mnli_features_encode_the_pair(tokenizer):
    feats = ft.build_features(tokenizer, br.TASKS["mnli"], MNLI, 2, 0)
    assert feats[0]["input_ids"].count(tokenizer.sep_token_id) == 2
    assert feats[0]["labels"] in (0, 1, 2)


@pytest.mark.parametrize("task", ["sst2", "mnli", "conll2003"])
def test_finetune_then_backward_restoration_end_to_end(
    tokenizer, tmp_path, monkeypatch, task
):
    monkeypatch.setattr(ft, "load_hf_split", lambda spec: ROWS[task])
    monkeypatch.setattr(br, "load_hf_split", lambda spec: ROWS[task])

    path = ft.finetune(
        task,
        out_root=str(tmp_path / "ckpt"),
        epochs=1,
        max_train_samples=6,
        max_eval_samples=4,
        batch_size=2,
        model=_tiny(task),
        tokenizer=tokenizer,
    )
    assert path == str(tmp_path / "ckpt" / f"{task}_finetuned")
    assert os.path.exists(os.path.join(path, "config.json"))

    out_dir = tmp_path / "m4"
    br.run_task(task, path, str(out_dir), str(out_dir), n_probe=3)
    with open(out_dir / f"M_bwd_{task}_effective_rank.json") as f:
        data = json.load(f)
    assert data["model_source"] == path and data["has_task_head"] is True
    assert data["sanity_check"]["passed"] is True
    assert np.array(data["matrix"]).shape == (6, 6)


def test_existing_checkpoint_is_never_silently_overwritten(
    tokenizer, tmp_path, monkeypatch
):
    monkeypatch.setattr(ft, "load_hf_split", lambda spec: ROWS["sst2"])
    existing = tmp_path / "sst2_finetuned"
    existing.mkdir()
    (existing / "marker.txt").write_text("keep me")
    assert (
        ft.finetune(
            "sst2", out_root=str(tmp_path), model=_tiny("sst2"), tokenizer=tokenizer
        )
        is None
    )
    assert (existing / "marker.txt").read_text() == "keep me"