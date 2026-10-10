"""CPU-friendly change-vector baseline for co-registered raster pairs."""
from __future__ import annotations

import numpy as np
from skimage.exposure import match_histograms
from skimage.filters import threshold_otsu
from skimage.morphology import opening, disk
from skimage.measure import label


def change_vector_magnitude(
    before: np.ndarray,
    after: np.ndarray,
    valid_mask: np.ndarray | None = None,
    histogram_match: bool = True,
) -> np.ndarray:
    """Return normalized change magnitude for arrays shaped (bands, rows, cols).

    This is a simple baseline, not a trained model. Histogram matching can
    suppress real global changes, so evaluate it against histogram_match=False.
    """
    b = _as_chw(before, "before")
    a = _as_chw(after, "after")
    if b.shape != a.shape:
        raise ValueError(f"Before/after shapes differ: {b.shape} != {a.shape}")
    valid = _valid_mask(b, a, valid_mask)
    if not np.any(valid):
        raise ValueError("No valid pixels remain after nodata/finite-value masking")

    b = b.astype(np.float32, copy=False)
    # Copy to avoid mutating a caller-owned float32 array during matching.
    a = a.astype(np.float32, copy=True)
    if histogram_match:
        # Match only valid pixels; invalid/nodata borders must not influence
        # the reference distribution. Apply the resulting matched valid values
        # back only at valid locations, leaving excluded pixels out of the fit.
        for idx in range(b.shape[0]):
            ref_values = b[idx][valid]
            src_values = a[idx][valid]
            if ref_values.size and src_values.size:
                a[idx][valid] = match_histograms(src_values, ref_values, channel_axis=None)

    # Normalize each band by a robust pre-event spread so bands with larger
    # numeric ranges do not dominate the magnitude. Constant bands use scale 1.
    scales = np.ones((b.shape[0], 1, 1), dtype=np.float32)
    for idx in range(b.shape[0]):
        values = b[idx][valid]
        spread = float(np.percentile(values, 95) - np.percentile(values, 5))
        scales[idx, 0, 0] = spread if spread > 1e-6 else 1.0

    difference = (a - b) / scales
    magnitude = np.sqrt(np.mean(np.square(difference), axis=0))
    magnitude[~valid] = 0.0
    return magnitude.astype(np.float32)


def threshold_change(
    magnitude: np.ndarray,
    valid_mask: np.ndarray | None = None,
    min_object_size: int = 8,
    opening_radius: int = 1,
) -> tuple[np.ndarray, float]:
    """Otsu-threshold the change magnitude and remove small/noisy regions."""
    mag = np.asarray(magnitude, dtype=np.float32)
    if mag.ndim != 2:
        raise ValueError(f"magnitude must be 2-D, got shape {mag.shape}")
    valid = np.isfinite(mag)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != mag.shape:
            raise ValueError("valid_mask and magnitude must have the same shape")
        valid &= supplied
    if not np.any(valid):
        raise ValueError("No valid pixels available for thresholding")

    values = mag[valid]
    if values.size == 0 or float(values.max()) <= float(values.min()):
        return np.zeros(mag.shape, dtype=bool), float(values.max()) if values.size else 0.0

    threshold = float(threshold_otsu(values))
    prediction = (mag > threshold) & valid
    if opening_radius > 0:
        prediction = opening(prediction, footprint=disk(opening_radius))
    if min_object_size > 1 and np.any(prediction):
        # Explicit connected-component filtering avoids version-sensitive
        # remove_small_objects parameter changes across scikit-image releases.
        component_labels = label(prediction, connectivity=2)
        sizes = np.bincount(component_labels.ravel())
        keep_component = sizes >= min_object_size
        keep_component[0] = False
        prediction = keep_component[component_labels]
    prediction &= valid
    return prediction.astype(bool), threshold


def run_baseline(
    before: np.ndarray,
    after: np.ndarray,
    valid_mask: np.ndarray | None = None,
    histogram_match: bool = True,
    min_object_size: int = 8,
    opening_radius: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (change_mask, magnitude, threshold)."""
    magnitude = change_vector_magnitude(before, after, valid_mask, histogram_match)
    prediction, threshold = threshold_change(
        magnitude, valid_mask, min_object_size=min_object_size, opening_radius=opening_radius
    )
    return prediction, magnitude, threshold


def _as_chw(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"{name} must be (bands, rows, cols) or (rows, cols); got {arr.shape}")
    return arr


def _valid_mask(before: np.ndarray, after: np.ndarray, supplied: np.ndarray | None) -> np.ndarray:
    valid = np.isfinite(before).all(axis=0) & np.isfinite(after).all(axis=0)
    if supplied is not None:
        mask = np.asarray(supplied, dtype=bool)
        if mask.shape != valid.shape:
            raise ValueError("valid_mask must match image height and width")
        valid &= mask
    return valid
