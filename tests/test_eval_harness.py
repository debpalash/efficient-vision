"""Tests for the evaluation harness (task #15).

Solve rate is the number this whole project is judged on, so the harness itself
has to be trustworthy. The ordering rule gets asserted directly: predictions and
ground truth are both sorted left-to-right, so an ordering bug would cancel out
end-to-end and produce a plausible -- but wrong -- solve rate.
"""

from __future__ import annotations

import pytest
import torch

from efficientvision.eval.harness import (
    EvalResult,
    decode_to_string,
    evaluate_solve_rate,
    levenshtein,
    nms,
    sweep_confidence,
)

NAMES = [str(i) for i in range(10)] + list("abcdef")


def test_levenshtein_basics():
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("abc", "abd") == 1
    assert levenshtein("abc", "ab") == 1
    assert levenshtein("", "abc") == 3


def test_decode_orders_left_to_right_not_by_score():
    """The right-most box has the highest score; the string must still read l-to-r."""
    boxes = torch.tensor([
        [60.0, 0.0, 80.0, 20.0],
        [0.0, 0.0, 20.0, 20.0],
        [30.0, 0.0, 50.0, 20.0],
    ])
    scores = torch.zeros(3, len(NAMES))
    scores[0, 2] = 0.99      # right-most, highest confidence
    scores[1, 0] = 0.90      # left-most
    scores[2, 1] = 0.95      # middle
    assert decode_to_string(boxes, scores, NAMES, conf=0.5, iou=0.7) == "012"


def test_decode_respects_confidence_threshold():
    boxes = torch.tensor([[0.0, 0.0, 20.0, 20.0], [30.0, 0.0, 50.0, 20.0]])
    scores = torch.zeros(2, len(NAMES))
    scores[0, 0] = 0.90
    scores[1, 1] = 0.30
    assert decode_to_string(boxes, scores, NAMES, conf=0.5, iou=0.7) == "0"
    assert decode_to_string(boxes, scores, NAMES, conf=0.2, iou=0.7) == "01"


def test_nms_is_class_agnostic():
    """Two overlapping boxes on one glyph with different classes are duplicates.

    Per-class NMS would keep both and insert a spurious character, breaking the
    string. Class-agnostic NMS must collapse them to one.
    """
    boxes = torch.tensor([[0.0, 0.0, 20.0, 20.0], [1.0, 1.0, 21.0, 21.0]])
    scores = torch.zeros(2, len(NAMES))
    scores[0, 0] = 0.90      # class '0'
    scores[1, 5] = 0.85      # class '5', same location
    assert decode_to_string(boxes, scores, NAMES, conf=0.5, iou=0.5) == "0"


def test_nms_keeps_distinct_characters():
    boxes = torch.tensor([[0.0, 0.0, 20.0, 20.0], [40.0, 0.0, 60.0, 20.0]])
    scores = torch.zeros(2, len(NAMES))
    scores[0, 0] = 0.90
    scores[1, 5] = 0.85
    assert decode_to_string(boxes, scores, NAMES, conf=0.5, iou=0.5) == "05"


def test_nms_on_empty_input():
    assert nms(torch.zeros(0, 4), torch.zeros(0), 0.5).numel() == 0
    assert decode_to_string(
        torch.zeros(0, 4), torch.zeros(0, len(NAMES)), NAMES, 0.5, 0.7
    ) == ""


class _PerfectModel:
    """Returns the ground truth verbatim -- the harness must score it 1.0."""

    def __init__(self, boxes, labels, mask, nc):
        self.boxes, self.labels, self.mask, self.nc = boxes, labels, mask, nc

    def to(self, device):
        return self

    def eval(self):
        return self

    def predict(self, imgs):
        b, a = imgs.shape[0], self.boxes.shape[1]
        scores = torch.zeros(b, a, self.nc)
        for i in range(b):
            for j in range(a):
                if self.mask[i, j]:
                    scores[i, j, self.labels[i, j]] = 0.99
        return self.boxes, scores


@pytest.fixture
def perfect_batch():
    boxes = torch.tensor([[
        [0.0, 0.0, 20.0, 40.0],
        [30.0, 0.0, 50.0, 40.0],
        [60.0, 0.0, 80.0, 40.0],
    ]])
    labels = torch.tensor([[3, 1, 4]])
    mask = torch.tensor([[True, True, True]])
    imgs = torch.zeros(1, 3, 64, 64)
    return imgs, boxes, labels, mask


def test_perfect_model_scores_one(perfect_batch):
    imgs, boxes, labels, mask = perfect_batch
    model = _PerfectModel(boxes, labels, mask, len(NAMES))
    result = evaluate_solve_rate(model, [(imgs, boxes, labels, mask)], NAMES, conf=0.5)

    assert result.solve_rate == 1.0
    assert result.char_accuracy == 1.0
    assert result.count_match == 1.0
    assert result.too_many_boxes == 0.0 and result.too_few_boxes == 0.0
    assert result.n_images == 1


def test_one_wrong_character_fails_the_whole_string(perfect_batch):
    """The contract is all-or-nothing per image: 2/3 characters right scores 0."""
    imgs, boxes, labels, mask = perfect_batch
    wrong = labels.clone()
    wrong[0, 1] = 9
    model = _PerfectModel(boxes, wrong, mask, len(NAMES))
    result = evaluate_solve_rate(model, [(imgs, boxes, labels, mask)], NAMES, conf=0.5)

    assert result.solve_rate == 0.0
    assert result.count_match == 1.0                    # detection was fine
    assert result.char_accuracy == pytest.approx(2 / 3)


def test_sweep_returns_the_best_operating_point(perfect_batch):
    imgs, boxes, labels, mask = perfect_batch
    model = _PerfectModel(boxes, labels, mask, len(NAMES))
    best, all_results = sweep_confidence(
        model, [(imgs, boxes, labels, mask)], NAMES, confs=(0.1, 0.5, 0.999)
    )
    assert len(all_results) == 3
    assert best.solve_rate == 1.0
    # A threshold above every score must drop all detections.
    assert min(r.solve_rate for r in all_results) == 0.0


def test_eval_result_is_serialisable(perfect_batch):
    imgs, boxes, labels, mask = perfect_batch
    model = _PerfectModel(boxes, labels, mask, len(NAMES))
    d = evaluate_solve_rate(model, [(imgs, boxes, labels, mask)], NAMES).as_dict()
    assert isinstance(d, dict) and "solve_rate" in d
    assert isinstance(EvalResult(**d), EvalResult)
