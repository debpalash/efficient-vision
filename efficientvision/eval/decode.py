"""Sequence decoding with a character-count prior (task #24).

At the best operating point the pooled teacher loses **35% of images to count
errors alone** -- 25.5% predict too few characters, 9.5% too many -- against only
~14% that get the count right and a character wrong. So the single largest
failure mode is not recognition. It is deciding *how many* characters there are.

The standard decoder makes that decision with one global confidence threshold:
every box above it survives, and the count falls out as a side effect. That
throws away everything known about the domain. Measured from the training labels:

  * images contain 3-7 characters, p50 = 5 (the 230x60 source is *always* 5)
  * characters read left to right and barely overlap
  * within an image they are of broadly similar size

This decoder instead *searches over counts*. For each plausible k it takes the k
best-scoring non-suppressed boxes and scores that whole hypothesis -- detector
confidence, the empirical count prior, and geometric plausibility together -- then
returns the best-scoring string. A spurious 8th box no longer silently joins the
output just for clearing a threshold; it has to make the whole hypothesis better.

The threshold does not disappear, it just stops being load-bearing: it only has to
be permissive enough to keep the true boxes as candidates.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import torch


@dataclass
class CountPrior:
    """Empirical distribution over characters-per-image, measured from labels."""

    log_probs: dict[int, float] = field(default_factory=dict)
    floor: float = -12.0

    @classmethod
    def from_counts(cls, counts: dict[int, int], smoothing: float = 0.5) -> "CountPrior":
        """Build from a {count: n_images} histogram with add-alpha smoothing.

        Smoothing matters: a count never seen in training gets probability zero,
        and a single zero would make that hypothesis impossible rather than merely
        unlikely. The val set will contain counts the train set missed.
        """
        if not counts:
            return cls({})
        lo, hi = min(counts), max(counts)
        total = sum(counts.values()) + smoothing * (hi - lo + 1)
        return cls({
            k: math.log((counts.get(k, 0) + smoothing) / total)
            for k in range(lo, hi + 1)
        })

    @classmethod
    def from_label_dir(cls, label_dir) -> "CountPrior":
        from pathlib import Path

        hist: Counter = Counter()
        for p in Path(label_dir).iterdir():
            if p.suffix != ".txt":
                continue
            hist[sum(1 for ln in p.read_text().splitlines() if ln.strip())] += 1
        return cls.from_counts(dict(hist))

    def logp(self, k: int) -> float:
        return self.log_probs.get(k, self.floor)

    @property
    def support(self) -> list[int]:
        return sorted(self.log_probs)


def _pairwise_overlap_penalty(boxes: torch.Tensor) -> float:
    """Total horizontal overlap between neighbouring boxes, normalized.

    Characters in these CAPTCHAs do overlap somewhat (measured p5 gap is -24.7 px),
    so overlap is penalised rather than forbidden. What this actually suppresses is
    the duplicate-detection case, where two boxes sit almost exactly on top of each
    other -- near-total overlap, which real neighbours never have.
    """
    if len(boxes) < 2:
        return 0.0
    order = ((boxes[:, 0] + boxes[:, 2]) / 2).argsort()
    b = boxes[order]
    widths = (b[:, 2] - b[:, 0]).clamp(min=1e-6)
    overlaps = (b[:-1, 2] - b[1:, 0]).clamp(min=0)
    frac = overlaps / torch.minimum(widths[:-1], widths[1:])
    return float(frac.clamp(0, 1).sum())


def _size_consistency_penalty(boxes: torch.Tensor) -> float:
    """Coefficient of variation of glyph height.

    Real per-character size does vary within an image, so this is a weak signal
    and is weighted accordingly -- but a box that is wildly out of scale with its
    neighbours is usually a merged pair or a fragment of a distractor line.
    """
    if len(boxes) < 2:
        return 0.0
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
    return float((h.std(unbiased=False) / h.mean()).clamp(0, 2))


def score_hypothesis(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    prior: CountPrior,
    rejected: torch.Tensor | None = None,
    w_overlap: float = 1.5,
    w_size: float = 0.5,
) -> float:
    """Log-likelihood of accepting `boxes` and rejecting `rejected`.

    Every candidate contributes exactly one term: `log p` if accepted, `log(1-p)`
    if rejected. That is what makes counts comparable.

    Two simpler scorings were tried first and both are broken:

      * **sum of accepted log-confidences** biases every image toward the shortest
        hypothesis, since each extra term is negative regardless of evidence.
      * **mean of accepted log-confidences** removes that bias but makes the score
        invariant to count when candidates are equally confident, so the prior can
        never be overruled -- six clearly-present characters still decode as five.
        `test_prior_can_be_overruled_by_strong_evidence` caught exactly this.

    Scoring rejections fixes both: the term count is constant across hypotheses
    (so no length bias), and discarding a confident box costs `log(1-0.95)` which
    strong evidence can use to overrule the prior.
    """
    total = prior.logp(len(boxes))
    if len(boxes):
        total += float(torch.log(scores.clamp(min=1e-9)).sum())
        total -= w_overlap * _pairwise_overlap_penalty(boxes)
        total -= w_size * _size_consistency_penalty(boxes)
    if rejected is not None and len(rejected):
        total += float(torch.log((1.0 - rejected).clamp(min=1e-9)).sum())
    return total


def calibrate(p: torch.Tensor, gamma: float) -> torch.Tensor:
    """Map raw detector confidence onto something usable as P(is a character).

    The accept/reject likelihood in `score_hypothesis` assumes `p` is that
    probability. It is not: a detector box at confidence 0.3 is a real glyph far
    more than 30% of the time, because confidence is trained against an IoU-
    weighted target, not against "does a character exist here". Taking the score
    at face value made rejection look cheap and cost 10 points of solve rate --
    the decoder under-predicted the count on 46% of images.

    `p ** gamma` with gamma < 1 lifts low confidences while preserving order, so
    the ranking the detector produces is untouched and only the accept/reject
    trade-off moves. gamma = 1 recovers the uncalibrated behaviour.
    """
    return p.clamp(1e-6, 1.0) ** gamma


def decode_with_count_prior(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    names: list[str],
    prior: CountPrior,
    base_conf: float = 0.05,
    nms_iou: float = 0.5,
    max_candidates: int = 24,
    w_overlap: float = 1.5,
    w_size: float = 0.5,
    gamma: float = 1.0,
) -> str:
    """Decode to a string by searching over plausible character counts.

    Args:
        boxes: (A, 4) xyxy. scores: (A, nc) per-class probabilities.
        base_conf: permissive floor. It only has to retain the true boxes as
            candidates; the count decision is made by the search, not by this.

    Returns the best-scoring string, ordered left to right.
    """
    from efficientvision.eval.harness import nms

    best_score, best_cls = scores.max(-1)
    keep = best_score >= base_conf
    if not keep.any():
        return ""

    b, s, c = boxes[keep], best_score[keep], best_cls[keep]
    idx = nms(b, s, nms_iou)
    b, s, c = b[idx], s[idx], c[idx]

    # NMS returns highest-score-first, so a prefix of length k is the top-k.
    if len(b) > max_candidates:
        b, s, c = b[:max_candidates], s[:max_candidates], c[:max_candidates]

    s_cal = calibrate(s, gamma)

    support = prior.support or [len(b)]
    best, best_val = None, -math.inf
    for k in support:
        if k == 0 or k > len(b):
            continue
        val = score_hypothesis(b[:k], s_cal[:k], prior, s_cal[k:], w_overlap, w_size)
        if val > best_val:
            best_val, best = val, k

    if best is None:
        # Every plausible count exceeds the candidates available; fall back to
        # taking everything rather than returning nothing.
        best = len(b)

    sel_b, sel_c = b[:best], c[:best]
    order = ((sel_b[:, 0] + sel_b[:, 2]) / 2).argsort()
    return "".join(names[int(k)] for k in sel_c[order])
