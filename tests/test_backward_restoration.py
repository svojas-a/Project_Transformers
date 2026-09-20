"""
Tests for src/analysis/backward_restoration.py

Like the other tests in this repo, these avoid the network: the model is a
tiny randomly-initialised DistilBERT and the tokenizer is built from a small
local vocab file. Datasets are replaced by hand-written rows in the same
format as the real HF splits (SST-2, MNLI, CoNLL-2003).
"""

import json

import numpy as np
import pytest
import torch
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
    DistilBertModel,
    DistilBertTokenizerFast,
)

from src.analysis import backward_restoration as br

WORDS = (
    "the movie was very good bad a an film plot actor not great terrible and . , "
    "eu rejects german call to boycott british lamb"
).split()
N_LAYERS = br.NUM_LAYERS


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS
    path = tmp_path_factory.mktemp("vocab") / "vocab.txt"
    path.write_text("\n".join(vocab))
    return DistilBertTokenizerFast(vocab_file=str(path), do_lower_case=True)


def _config(**kw):
    return DistilBertConfig(
        dim=64, hidden_dim=128, n_heads=4, n_layers=N_LAYERS, vocab_size=64, **kw
    )


def _finish(model):
    model.eval()
    return model.to(br.DEVICE)


@pytest.fixture(scope="module")
def seq_model():
    torch.manual_seed(0)
    return _finish(DistilBertForSequenceClassification(_config(num_labels=3)))


@pytest.fixture(scope="module")
def token_model():
    torch.manual_seed(0)
    return _finish(DistilBertForTokenClassification(_config(num_labels=9)))


@pytest.fixture(scope="module")
def base_model():
    torch.manual_seed(0)
    return _finish(DistilBertModel(_config()))


SST2_ROWS = [
    {"sentence": " the movie was very good ", "label": 1},
    {"sentence": "not a great film", "label": 0},
    {"sentence": "terrible", "label": 0},  # too short -> filtered
    {"sentence": "the plot and the actor were very good", "label": 1},
    {"sentence": "a bad bad film", "label": 0},
    {"sentence": "the film was not very great", "label": 0},
    {"sentence": "very good actor , good plot", "label": 1},
    {"sentence": "the movie was terrible and bad", "label": 0},
]
MNLI_ROWS = [
    {
        "premise": "the movie was very good",
        "hypothesis": "the film was great",
        "label": 0,
    },
    {"premise": "a bad film", "hypothesis": "the plot was not good", "label": 1},
    {
        "premise": "the actor was great",
        "hypothesis": "the actor was terrible",
        "label": 2,
    },
    {"premise": "the plot and the film", "hypothesis": "a good film", "label": 0},
    {"premise": "not a great movie", "hypothesis": "the movie was bad", "label": 0},
    {"premise": "very good actor", "hypothesis": "not a good actor", "label": 2},
    {"premise": "the film was bad", "hypothesis": "the film was very bad", "label": 1},
]
CONLL_ROWS = [
    {"tokens": ["-DOCSTART-"], "ner_tags": [0]},  # separator -> skipped
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
        "tokens": ["eu", "call", "to", "boycott", "british", "lamb"],
        "ner_tags": [3, 0, 0, 0, 7, 0],
    },
    {
        "tokens": ["a", "bad", "german", "film", "not", "good"],
        "ner_tags": [0, 0, 7, 0, 0, 0],
    },
    {"tokens": ["the", "british", "actor", "was", "good"], "ner_tags": [0, 7, 0, 0, 0]},
]
ROWS = {"sst2": SST2_ROWS, "mnli": MNLI_ROWS, "conll2003": CONLL_ROWS}


def _model_for(task, seq_model, token_model):
    return token_model if task == "conll2003" else seq_model


def _examples(tokenizer, task, n=4):
    return br.load_probe_examples(
        tokenizer, br.TASKS[task], n, dataset=ROWS[task], min_seq_len=5
    )


ALL_TASKS = ["sst2", "mnli", "conll2003"]


# ---------------------------------------------------------------------------
# task registry / data handling
# ---------------------------------------------------------------------------
def test_registry_covers_the_three_doc_tasks():
    assert set(br.TASKS) == {"sst2", "mnli", "conll2003"}
    assert br.TASKS["sst2"].input_kind == "single"
    assert br.TASKS["mnli"].input_kind == "pair"
    assert br.TASKS["conll2003"].input_kind == "tokens"


def test_row_to_example_for_each_kind():
    assert br.row_to_example(br.TASKS["sst2"], SST2_ROWS[0]) == (
        "the movie was very good",
        1,
    )
    text, label = br.row_to_example(br.TASKS["mnli"], MNLI_ROWS[0])
    assert text == ("the movie was very good", "the film was great") and label == 0
    words, tags = br.row_to_example(br.TASKS["conll2003"], CONLL_ROWS[1])
    assert words[0] == "eu" and tags[0] == 3 and len(words) == len(tags)


def test_conll_docstart_rows_are_skipped():
    assert br.row_to_example(br.TASKS["conll2003"], CONLL_ROWS[0]) is None


@pytest.mark.parametrize("task", ALL_TASKS)
def test_probe_selection_is_reproducible(tokenizer, task):
    a = _examples(tokenizer, task)
    b = _examples(tokenizer, task)
    assert [e.text for e in a] == [e.text for e in b]
    assert all(e.seq_len >= 5 for e in a)


def test_short_examples_are_filtered(tokenizer):
    texts = [e.text for e in _examples(tokenizer, "sst2", n=10)]
    assert "terrible" not in texts


def test_pair_examples_encode_both_sentences(tokenizer):
    ex = _examples(tokenizer, "mnli", n=1)[0]
    ids = br.encode(tokenizer, br.TASKS["mnli"], ex)["input_ids"][0].tolist()
    sep = tokenizer.sep_token_id
    assert ids.count(sep) == 2  # one [SEP] after each sentence
    assert "token_type_ids" not in br.encode(tokenizer, br.TASKS["mnli"], ex)


def test_token_examples_keep_word_alignment(tokenizer):
    ex = _examples(tokenizer, "conll2003", n=1)[0]
    assert ex.word_ids is not None and len(ex.word_ids) == ex.seq_len
    assert ex.word_ids[0] is None and ex.word_ids[-1] is None  # [CLS] / [SEP]
    assert len(ex.label) == len(ex.text)


# ---------------------------------------------------------------------------
# hooks / passes
# ---------------------------------------------------------------------------
def test_hooks_are_removed_after_every_pass(tokenizer, seq_model):
    spec, ex = br.TASKS["sst2"], _examples(tokenizer, "sst2", 2)
    clean = br.run_pass(seq_model, tokenizer, spec, ex)
    br.run_pass(
        seq_model,
        tokenizer,
        spec,
        ex,
        clean_acts=clean.acts,
        restore_layers={1},
        collapse_layers={0, 2, 3, 4, 5},
    )
    for layer in br.get_transformer_layers(seq_model):
        assert len(layer._forward_hooks) == 0


def test_pass_with_no_mutators_equals_clean_pass(tokenizer, seq_model):
    spec, ex = br.TASKS["sst2"], _examples(tokenizer, "sst2", 3)
    a = br.run_pass(seq_model, tokenizer, spec, ex)
    b = br.run_pass(seq_model, tokenizer, spec, ex)
    for i in range(N_LAYERS):
        for x, y in zip(a.acts[i], b.acts[i]):
            assert torch.equal(x, y)


def test_run_pass_rejects_bad_arguments(tokenizer, seq_model):
    spec, ex = br.TASKS["sst2"], _examples(tokenizer, "sst2", 1)
    with pytest.raises(ValueError):
        br.run_pass(seq_model, tokenizer, spec, ex, restore_layers={0})  # no clean_acts
    clean = br.run_pass(seq_model, tokenizer, spec, ex)
    with pytest.raises(ValueError):
        br.run_pass(
            seq_model,
            tokenizer,
            spec,
            ex,
            clean_acts=clean.acts,
            restore_layers={0},
            collapse_layers={0},
        )


@pytest.mark.parametrize("task", ALL_TASKS)
def test_full_restoration_reproduces_clean_pass(
    tokenizer, seq_model, token_model, task
):
    model = _model_for(task, seq_model, token_model)
    spec, ex = br.TASKS[task], _examples(tokenizer, task)
    clean = br.run_pass(model, tokenizer, spec, ex)
    full = br.run_pass(
        model,
        tokenizer,
        spec,
        ex,
        clean_acts=clean.acts,
        restore_layers=set(range(N_LAYERS)),
    )
    sanity = br.check_full_restoration(clean, full)
    assert sanity["passed"], sanity
    assert sanity["max_activation_diff"] <= br.SANITY_TOL
    assert sanity["predictions_identical"]


def test_sanity_check_detects_a_broken_restoration(tokenizer, seq_model):
    spec, ex = br.TASKS["sst2"], _examples(tokenizer, "sst2")
    clean = br.run_pass(seq_model, tokenizer, spec, ex)
    collapsed = br.run_pass(
        seq_model, tokenizer, spec, ex, collapse_layers=set(range(N_LAYERS))
    )
    assert not br.check_full_restoration(clean, collapsed)["passed"]


def test_collapse_actually_changes_activations(tokenizer, seq_model):
    spec, ex = br.TASKS["sst2"], _examples(tokenizer, "sst2")
    clean = br.run_pass(seq_model, tokenizer, spec, ex)
    collapsed = br.run_pass(
        seq_model, tokenizer, spec, ex, collapse_layers=set(range(N_LAYERS))
    )
    diff = (clean.acts[0][0] - collapsed.acts[0][0]).abs().max().item()
    assert diff > 1e-4


def test_single_layer_restoration_is_causal(tokenizer, seq_model):
    """Restoring layer 2: layer 2 is clean, layers 0-1 are untouched (still
    the collapsed baseline), and layer 3+ changes."""
    spec, ex = br.TASKS["sst2"], _examples(tokenizer, "sst2", 3)
    all_layers = set(range(N_LAYERS))
    clean = br.run_pass(seq_model, tokenizer, spec, ex)
    collapsed = br.run_pass(seq_model, tokenizer, spec, ex, collapse_layers=all_layers)
    restored = br.run_pass(
        seq_model,
        tokenizer,
        spec,
        ex,
        clean_acts=clean.acts,
        restore_layers={2},
        collapse_layers=all_layers - {2},
    )
    for k in range(len(ex)):
        assert torch.allclose(restored.acts[2][k], clean.acts[2][k], atol=1e-6)
        for upstream in (0, 1):
            assert torch.equal(restored.acts[upstream][k], collapsed.acts[upstream][k])
    changed = (restored.acts[3][0] - collapsed.acts[3][0]).abs().max().item()
    assert changed > 1e-6


# ---------------------------------------------------------------------------
# matrix + summaries
# ---------------------------------------------------------------------------
def _matrices(tokenizer, model, task, metric_name="effective_rank"):
    spec, ex = br.TASKS[task], _examples(tokenizer, task)
    all_layers = set(range(N_LAYERS))
    fn = br.METRIC_FUNCS[metric_name]
    clean = br.run_pass(model, tokenizer, spec, ex)
    collapsed = br.run_pass(model, tokenizer, spec, ex, collapse_layers=all_layers)
    restored = {
        i: br.run_pass(
            model,
            tokenizer,
            spec,
            ex,
            clean_acts=clean.acts,
            restore_layers={i},
            collapse_layers=all_layers - {i},
        )
        for i in range(N_LAYERS)
    }
    clean_m = br.compute_metric_per_example(clean.acts, fn)
    collapsed_m = br.compute_metric_per_example(collapsed.acts, fn)
    restored_m = {
        i: br.compute_metric_per_example(restored[i].acts, fn) for i in restored
    }
    R = br.compute_recovery_tensor(clean_m, collapsed_m, restored_m, metric_name)
    baseline = [
        br.compute_damage(clean_m[j], collapsed_m[j], metric_name).mean()
        for j in range(N_LAYERS)
    ]
    return R, baseline, (clean, collapsed, restored, ex)


@pytest.mark.parametrize("task", ALL_TASKS)
def test_matrix_shape_diagonal_and_lower_triangle(
    tokenizer, seq_model, token_model, task
):
    model = _model_for(task, seq_model, token_model)
    R, baseline, _ = _matrices(tokenizer, model, task)
    M = br.matrix_from_recovery(R)
    assert M.shape == (N_LAYERS, N_LAYERS)
    assert np.allclose(np.tril(M, k=-1), 0.0)  # unreachable cells are 0.0
    # Diagonal = full baseline damage at i (layer i is patched to its clean value).
    assert np.allclose(np.diag(M), baseline, atol=1e-5)


def test_summarize_rows_excludes_diagonal_and_nans_last_row():
    n = 5
    R = np.full((N_LAYERS, N_LAYERS, n), np.nan)
    for i in range(N_LAYERS):
        for j in range(i, N_LAYERS):
            R[i, j] = 100.0 if i == j else float(j - i)  # huge diagonal, ramp after
    s = br.summarize_rows(R)
    for i in range(N_LAYERS - 1):
        assert s["row_avg_recovery"][i] == pytest.approx((N_LAYERS - i) / 2)  # no 100
        assert s["adjacent_recovery"][i] == pytest.approx(1.0)
        assert s["final_layer_recovery"][i] == pytest.approx(N_LAYERS - 1 - i)
    assert np.isnan(s["row_avg_recovery"][-1])
    assert s["per_sentence_row_avg_recovery"][-1] is None
    assert s["propagation_slope"][0] == pytest.approx(1.0)
    assert np.isnan(s["propagation_slope"][N_LAYERS - 2])  # only one downstream cell
    assert br.top_layer(s["row_avg_recovery"]) == 0
    assert br.top_layer(s["adjacent_recovery"]) in range(
        N_LAYERS - 1
    )  # never the NaN row


def test_top_layer_ignores_nan_and_handles_all_nan():
    assert br.top_layer([0.1, 0.9, 0.3, float("nan")]) == 1
    assert br.top_layer([float("nan")] * 3) is None


def test_compute_damage_sign_convention():
    assert br.compute_damage(10.0, 4.0, "effective_rank") == 6.0
    assert br.compute_damage(0.2, 0.9, "mean_cosine_sim") == pytest.approx(0.7)


def test_bootstrap_ci_brackets_the_mean():
    values = np.random.default_rng(0).normal(2.0, 1.0, size=40)
    lo, hi = br.bootstrap_ci(values, n_bootstrap=500)
    assert lo < values.mean() < hi


# ---------------------------------------------------------------------------
# task performance
# ---------------------------------------------------------------------------
def test_score_task_sequence():
    spec = br.TASKS["sst2"]
    ex = [br.ProbeExample("a", lab, 5) for lab in (0, 1, 2)]
    logits = [
        torch.tensor([9.0, 0, 0]),
        torch.tensor([0, 9.0, 0]),
        torch.tensor([9.0, 0, 0]),
    ]
    assert br.score_task(spec, ex, logits) == pytest.approx(2 / 3)


def test_score_task_tokens_uses_first_subword_of_each_word():
    spec = br.TASKS["conll2003"]
    # words: 0,1,2 ; word 1 is split into two subwords (positions 2 and 3)
    ex = [br.ProbeExample(["a", "b", "c"], [1, 2, 0], 6, [None, 0, 1, 1, 2, None])]
    good = torch.zeros(6, 3)
    good[1, 1] = good[2, 2] = good[4, 0] = 1.0
    good[3, 1] = 5.0  # second subword of word 1 is wrong but must be ignored
    assert br.score_task(spec, ex, [good]) == pytest.approx(1.0)
    bad = good.clone()
    bad[4] = torch.tensor([0.0, 0.0, 9.0])
    assert br.score_task(spec, ex, [bad]) == pytest.approx(2 / 3)


def test_headless_model_gives_no_logits_or_performance(tokenizer, base_model):
    spec, ex = br.TASKS["sst2"], _examples(tokenizer, "sst2", 2)
    clean = br.run_pass(base_model, tokenizer, spec, ex)
    assert clean.logits is None and not br.model_has_head(base_model)
    assert br.task_performance_summary(spec, ex, clean, clean, {}, clean) is None
    assert br.check_full_restoration(clean, clean)["passed"]


# ---------------------------------------------------------------------------
# cross-check / loading
# ---------------------------------------------------------------------------
def _fwd_json(top_row):
    mat = np.zeros((N_LAYERS, N_LAYERS))
    for i in range(N_LAYERS):
        for j in range(i, N_LAYERS):
            mat[i, j] = 5.0 if i == top_row else 1.0
    return {"matrix": mat.tolist(), "row_avg_damage": mat.mean(axis=1).tolist()}


def test_cross_check_match_mismatch_and_missing(tmp_path):
    assert br.cross_check_with_forward(str(tmp_path), "sst2", "stable_rank", 0) is None
    (tmp_path / "M_fwd_sst2_stable_rank.json").write_text(json.dumps(_fwd_json(2)))
    hit = br.cross_check_with_forward(str(tmp_path), "sst2", "stable_rank", 2)
    miss = br.cross_check_with_forward(str(tmp_path), "sst2", "stable_rank", 4)
    assert hit["match"] is True and miss["match"] is False


def test_missing_checkpoint_fails_loudly_instead_of_falling_back(tmp_path):
    with pytest.raises(FileNotFoundError):
        br.load_model_and_tokenizer(br.TASKS["mnli"], str(tmp_path / "nope"), False)


def test_failed_hf_sources_are_all_reported(monkeypatch):
    datasets = pytest.importorskip("datasets")

    def boom(*args, **kwargs):
        raise FileNotFoundError("offline")

    monkeypatch.setattr(datasets, "load_dataset", boom)
    with pytest.raises(RuntimeError) as err:
        br.load_hf_split(br.TASKS["conll2003"])
    assert "eriktks/conll2003" in str(err.value) and "conll2003:" in str(err.value)


# ---------------------------------------------------------------------------
# end to end (all three tasks, writes the JSON files)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("task", ALL_TASKS)
def test_run_task_writes_valid_schema_files(
    tokenizer, seq_model, token_model, tmp_path, monkeypatch, task
):
    model = _model_for(task, seq_model, token_model)
    monkeypatch.setattr(
        br, "load_model_and_tokenizer", lambda *a: (model, tokenizer, "tiny-random")
    )
    monkeypatch.setattr(br, "load_hf_split", lambda spec: ROWS[task])
    monkeypatch.setattr(br, "MIN_SEQ_LEN", 5)
    monkeypatch.setattr(br, "N_BOOTSTRAP", 50)

    results = br.run_task(task, "ignored", str(tmp_path), str(tmp_path), n_probe=4)
    assert len(results) == len(br.METRIC_FUNCS)

    for metric_name in br.METRIC_FUNCS:
        path = tmp_path / f"M_bwd_{task}_{metric_name}.json"
        with open(path) as f:
            text = f.read()
        data = json.loads(text, parse_constant=lambda c: pytest.fail(f"non-JSON {c}"))
        for key in (
            "task",
            "direction",
            "granularity",
            "matrix",
            "metric",
            "target_rank_used",
        ):
            assert key in data
        assert data["task"] == task and data["direction"] == "backward"
        assert data["granularity"] == "layer" and data["metric"] == metric_name
        assert (
            isinstance(data["target_rank_used"], int) and data["target_rank_used"] >= 1
        )
        assert np.array(data["matrix"]).shape == (N_LAYERS, N_LAYERS)
        assert (
            data["row_avg_recovery"][-1] is None
        )  # last layer has no downstream cells
        assert data["sanity_check"]["passed"] is True
        assert (
            data["task_performance"]["fully_restored"]
            == data["task_performance"]["clean"]
        )
        assert data["model_source"] == "tiny-random"