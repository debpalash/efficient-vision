"""Signal-to-noise preprocessing for the low-contrast CAPTCHA sources (task #21).

The per-source breakdown split cleanly into two clusters: sources with bold,
high-contrast glyphs (char accuracy 0.89-0.95) and sources with pale, thin glyphs
buried under saturated dashed lines and dense speckle (0.67-0.77). The failure is
signal-to-noise, not capacity -- a 22M-parameter teacher does not fix it.

Zooming into the real images showed exactly what the noise is made of:

  * distractor lines are **1 px wide** and near-horizontal
  * speckle is **isolated single pixels**
  * glyph strokes are **thicker than both**

That thickness difference is the entire opening. A small median filter removes
structures thinner than its kernel while leaving thicker strokes intact, so it
deletes the lines and speckle almost selectively. CLAHE then restores the contrast
the pale glyphs lack.

Anything applied here also runs at inference, so it is part of the deployed
latency budget and is benchmarked with the model rather than treated as free.
"""

from __future__ import annotations

import numpy as np


def median_denoise(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Remove 1px lines and speckle; keep thicker glyph strokes.

    A median filter replaces each pixel with the median of its neighbourhood, so
    a structure thinner than half the kernel is outvoted by the background around
    it and vanishes. Gaussian blur would instead smear the noise into the glyph,
    lowering contrast exactly where it is already the problem.
    """
    import cv2

    return cv2.medianBlur(img, ksize)


def clahe_contrast(img: np.ndarray, clip: float = 2.5, grid: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on the L channel.

    Adaptive rather than global: the background tint varies per image and
    sometimes across one image, so a single global curve leaves parts of the
    frame flat. Only L is touched -- equalising the colour channels would destroy
    the hue separation between glyph and distractor lines, which is real signal.
    """
    import cv2

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def saturation_boost(img: np.ndarray, gain: float = 1.4) -> np.ndarray:
    """Push saturation up so pale glyphs separate from the near-white paper."""
    import cv2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * gain, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def unsharp(img: np.ndarray, amount: float = 1.0, sigma: float = 1.5) -> np.ndarray:
    """Sharpen edges after denoising, to recover stroke definition."""
    import cv2

    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


# Named pipelines, so an experiment refers to a recipe rather than a pile of flags.
PIPELINES: dict[str, list] = {
    "none": [],
    "median": [median_denoise],
    "clahe": [clahe_contrast],
    "median_clahe": [median_denoise, clahe_contrast],
    "median_clahe_sharp": [median_denoise, clahe_contrast, unsharp],
    "median_sat_clahe": [median_denoise, saturation_boost, clahe_contrast],
}


def apply_pipeline(img: np.ndarray, name: str) -> np.ndarray:
    if name not in PIPELINES:
        raise ValueError(f"unknown pipeline {name!r}; have {sorted(PIPELINES)}")
    for fn in PIPELINES[name]:
        img = fn(img)
    return img
