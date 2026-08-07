"""Tests for the count-prior decoder (task #24).

The decoder's whole job is to fix the two failure modes that cost 35% of images:
dropping a real character, and admitting a spurious one. Both get a direct test
with hand-built detections where the correct answer is unambiguous.
"""

from __future__ import annotations

import math

import pytest
import torch

from efficientvision.eval.decode import (
    CountPrior,
    decode_with_count_prior,
    score_hypothesis,
    _pairwise_overlap_penalty,
    _size_consistency_penalty,
)

NAMES = [str(i) for i in range(10)] + list("abcdef")


def _boxes(xs, w=20, h=40, y=5):
    return torch.tensor([[float(x), float(y), float(x + w), float(y + h)] for x in xs])


def _scores(classes, confs):
    s = torch.zeros(len(classes), len(NAMES))
    for i, (c, p) in enumerate(zip(classes, confs)):
        s[i, c] = p
    return s


# --------------------------------------------------------------------------
# CountPrior
# --------------------------------------------------------------------------

def test_prior_prefers_the_common_count():
    prior = CountPrior.from_counts({4: 10, 5: 100, 6: 10})
    assert prior.logp(5) > prior.logp(4)
    assert prior.logp(5) > prior.logp(6)


def test_prior_smoothing_keeps_unseen_counts_possible():
    """A count absent from training must be unlikely, not impossible."""
    prior = CountPrior.from_counts({5: 100, 7: 5})
    assert prior.logp(6) > -math.inf
    assert prior.logp(6) < prior.logp(5)


def test_prior_probabilities_sum_to_one():
    prior = CountPrior.from_counts({3: 5, 4: 20, 5: 60, 6: 15})
    total = sum(math.exp(prior.logp(k)) for k in prior.support)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_prior_from_label_dir(tmp_path):
    for i, n in enumerate([5, 5, 4]):
        (tmp_path / f"{i}.txt").write_text(
            "\n".join(f"{j} 0.5 0.5 0.1 0.4" for j in range(n))
        )
    prior = CountPrior.from_label_dir(tmp_path)
    assert prior.logp(5) > prior.logp(4)


# --------------------------------------------------------------------------
# geometric penalties
# --------------------------------------------------------------------------

def test_overlap_penalty_zero_for_separated_boxes():
    assert _pairwise_overlap_penalty(_boxes([0, 40, 80])) == pytest.approx(0.0)


def test_overlap_penalty_high_for_duplicate_boxes():
    """Two boxes on the same glyph -- the duplicate-detection case."""
    assert _pairwise_overlap_penalty(_boxes([0, 1])) > 0.9


def test_overlap_penalty_tolerates_real_neighbour_overlap():
    """Measured p5 gap is -24.7px, so partial overlap must not be punished hard."""
    assert _pairwise_overlap_penalty(_boxes([0, 15])) < 0.4


def test_size_penalty_zero_for_uniform_boxes():
    assert _size_consistency_penalty(_boxes([0, 40, 80])) == pytest.approx(0.0)


def test_size_penalty_positive_for_mismatched_boxes():
    b = torch.tensor([[0., 0., 20., 40.], [40., 0., 60., 8.]])
    assert _size_consistency_penalty(b) > 0.3


# --------------------------------------------------------------------------
# hypothesis scoring
# --------------------------------------------------------------------------

def test_scoring_rejections_removes_the_length_bias():
    """Truncating a set of equally-confident boxes must not score better.

    Scoring only accepted boxes with a sum would make the 2-box hypothesis win
    here purely by having fewer negative terms.
    """
    prior = CountPrior.from_counts({2: 1, 3: 1, 4: 1, 5: 1})   # flat over 2..5
    b, s = _boxes([0, 40, 80, 120, 160]), torch.full((5,), 0.9)
    short = score_hypothesis(b[:2], s[:2], prior, rejected=s[2:])
    full = score_hypothesis(b, s, prior, rejected=s[5:])
    assert full > short


def test_rejecting_a_confident_box_is_expensive():
    """This is what lets strong evidence overrule the count prior."""
    prior = CountPrior.from_counts({2: 1, 3: 1})
    b, s = _boxes([0, 40, 80]), torch.full((3,), 0.95)
    drop_confident = score_hypothesis(b[:2], s[:2], prior, rejected=s[2:])
    keep_all = score_hypothesis(b, s, prior, rejected=s[3:])
    assert keep_all > drop_confident


# --------------------------------------------------------------------------
# end-to-end decoding
# --------------------------------------------------------------------------

def test_recovers_a_clean_five_character_string():
    prior = CountPrior.from_counts({5: 100})
    boxes = _boxes([0, 40, 80, 120, 160])
    scores = _scores([1, 2, 3, 4, 5], [0.9] * 5)
    assert decode_with_count_prior(boxes, scores, NAMES, prior) == "12345"


def test_rejects_a_spurious_low_confidence_sixth_box():
    """The standard decoder admits this box for clearing the threshold."""
    prior = CountPrior.from_counts({5: 100})
    boxes = _boxes([0, 40, 80, 120, 160, 200])
    scores = _scores([1, 2, 3, 4, 5, 9], [0.9, 0.9, 0.9, 0.9, 0.9, 0.30])
    assert decode_with_count_prior(boxes, scores, NAMES, prior) == "12345"


def test_keeps_a_weak_but_real_fifth_box():
    """The mirror failure: a faint glyph a high threshold would have dropped.

    too_few_boxes was the larger error at 0.255, so this direction matters most.
    """
    prior = CountPrior.from_counts({5: 100})
    boxes = _boxes([0, 40, 80, 120, 160])
    scores = _scores([1, 2, 3, 4, 5], [0.9, 0.9, 0.9, 0.9, 0.12])
    assert decode_with_count_prior(boxes, scores, NAMES, prior) == "12345"


def test_prior_can_be_overruled_by_strong_evidence():
    """A prior favouring 5 must still yield 6 when six boxes are all confident."""
    prior = CountPrior.from_counts({5: 60, 6: 40})
    boxes = _boxes([0, 40, 80, 120, 160, 200])
    scores = _scores([1, 2, 3, 4, 5, 6], [0.95] * 6)
    assert decode_with_count_prior(boxes, scores, NAMES, prior) == "123456"


def test_output_is_ordered_left_to_right_not_by_confidence():
    prior = CountPrior.from_counts({3: 100})
    boxes = _boxes([80, 0, 40])
    scores = _scores([3, 1, 2], [0.99, 0.80, 0.90])
    assert decode_with_count_prior(boxes, scores, NAMES, prior) == "123"


def test_duplicate_detections_collapse_to_one_character():
    prior = CountPrior.from_counts({3: 100})
    boxes = torch.tensor([
        [0., 5., 20., 45.], [1., 6., 21., 46.],   # same glyph twice
        [40., 5., 60., 45.], [80., 5., 100., 45.],
    ])
    scores = _scores([1, 7, 2, 3], [0.9, 0.85, 0.9, 0.9])
    assert decode_with_count_prior(boxes, scores, NAMES, prior) == "123"


def test_empty_detections_give_empty_string():
    prior = CountPrior.from_counts({5: 100})
    assert decode_with_count_prior(
        torch.zeros(0, 4), torch.zeros(0, len(NAMES)), NAMES, prior
    ) == ""


def test_all_detections_below_floor_give_empty_string():
    prior = CountPrior.from_counts({5: 100})
    boxes = _boxes([0, 40])
    scores = _scores([1, 2], [0.01, 0.02])
    assert decode_with_count_prior(
        boxes, scores, NAMES, prior, base_conf=0.05
    ) == ""


def test_fewer_candidates_than_prior_support_does_not_crash():
    prior = CountPrior.from_counts({5: 100})
    boxes = _boxes([0, 40])
    scores = _scores([1, 2], [0.9, 0.9])
    assert decode_with_count_prior(boxes, scores, NAMES, prior) == "12"
