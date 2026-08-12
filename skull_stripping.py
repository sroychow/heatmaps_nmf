"""ThresContCT_TBI skull stripping and medical-image input utilities.

This module implements the rule-based preprocessing pipeline described by Rahman
et al. (2025): intensity thresholding, morphological closing, contour filtering,
and distance-transform refinement.  It intentionally operates slice-by-slice,
matching the 2-D algorithm evaluated in the paper.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pydicom


@dataclass(frozen=True)
class ThresContConfig:
    """Parameters reported for the ThresContCT_TBI algorithm."""

    skull_threshold: int = 220
    closing_kernel: int = 20
    minimum_contour_area: float = 500.0
    distance_threshold: float = 10.0
    artifact_threshold: int = 240
    window_min: float = -100.0
    window_max: float = 1000.0


def normalize_ct_slice(image, window_min=-100.0, window_max=1000.0):
    """Window a CT slice and return the 8-bit representation used by the paper."""
    if window_min >= window_max:
        raise ValueError("window_min must be less than window_max")
    values = np.nan_to_num(
        np.asarray(image, dtype=np.float32), nan=window_min, posinf=window_max,
        neginf=window_min,
    )
    values = np.clip(values, window_min, window_max)
    return np.rint((values - window_min) * 255.0 / (window_max - window_min)).astype(
        np.uint8
    )


def threscont_ct_tbi(image, config=ThresContConfig(), already_normalized=False):
    """Extract one brain slice with the paper's ThresContCT_TBI method.

    Returns ``(stripped, brain_mask, normalized)``.  Raw DICOM/NIfTI intensities
    are CT-windowed first; callers supplying an 8-bit image may set
    ``already_normalized=True``.
    """
    if config.closing_kernel <= 0:
        raise ValueError("closing_kernel must be positive")
    normalized = (
        np.asarray(image, dtype=np.uint8)
        if already_normalized
        else normalize_ct_slice(image, config.window_min, config.window_max)
    )
    if normalized.ndim != 2:
        raise ValueError("ThresContCT_TBI expects a two-dimensional CT slice")

    _, high_intensity = cv2.threshold(
        normalized, config.skull_threshold, 255, cv2.THRESH_BINARY
    )
    kernel = np.ones(
        (config.closing_kernel, config.closing_kernel), dtype=np.uint8
    )
    closed = cv2.morphologyEx(high_intensity, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    retained = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= config.minimum_contour_area
    ]

    enclosed = np.zeros_like(normalized)
    if retained:
        cv2.drawContours(enclosed, retained, -1, 255, thickness=cv2.FILLED)
    distances = cv2.distanceTransform(enclosed, cv2.DIST_L2, 5)
    brain_mask = distances > config.distance_threshold

    stripped = np.where(brain_mask, normalized, 0).astype(np.uint8)
    stripped[stripped > config.artifact_threshold] = 0
    return stripped, brain_mask, normalized


def _dicom_slices(path):
    dataset = pydicom.dcmread(path)
    modality = str(getattr(dataset, "Modality", "")).upper()
    if modality and modality != "CT":
        raise ValueError(f"{path} is modality {modality!r}, not CT")
    stored = np.asarray(dataset.pixel_array)
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    volume = stored.astype(np.float32) * slope + intercept
    if volume.ndim == 2:
        volume = volume[np.newaxis, ...]
    elif volume.ndim != 3:
        raise ValueError(f"Unsupported DICOM pixel shape: {volume.shape}")
    for index, image in enumerate(volume):
        yield index, image


def iter_ct_slices(path):
    """Yield slices from DICOM, NIfTI, or common 8-bit raster inputs recursively."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    candidates = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    found = False
    for candidate in candidates:
        suffixes = "".join(candidate.suffixes).lower()
        if suffixes.endswith((".nii", ".nii.gz")):
            data = np.asarray(nib.load(candidate).dataobj, dtype=np.float32)
            if data.ndim == 2:
                data = data[..., np.newaxis]
            if data.ndim != 3:
                raise ValueError(f"Unsupported NIfTI image shape: {data.shape}")
            for index in range(data.shape[2]):
                found = True
                yield candidate, index, data[:, :, index]
        elif candidate.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        }:
            image = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not read raster image: {candidate}")
            found = True
            yield candidate, 0, image
        elif pydicom.misc.is_dicom(candidate):
            for index, image in _dicom_slices(candidate):
                found = True
                yield candidate, index, image
    if not found:
        raise ValueError(f"No DICOM, NIfTI, JPEG, or PNG CT images found in {path}")


def dump_threscont_previews(input_path, output_dir, config=ThresContConfig()):
    """Write before, after, and side-by-side PNGs for every input slice."""
    output_dir = Path(output_dir)
    before_dir, after_dir = output_dir / "before", output_dir / "after"
    comparison_dir = output_dir / "comparison"
    before_dir.mkdir(parents=True, exist_ok=True)
    after_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for number, (source, index, image) in enumerate(iter_ct_slices(input_path)):
        # Raster images already use the 8-bit intensity scale assumed by the
        # paper. Medical formats contain raw/calibrated values and require the
        # configurable CT window before applying the reported thresholds.
        is_raster = source.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        }
        stripped, _, normalized = threscont_ct_tbi(
            image, config, already_normalized=is_raster
        )
        stem = source.name.replace(".nii.gz", "").replace(".nii", "")
        filename = f"{number:05d}_{stem}_slice-{index:04d}.png"
        before, after = before_dir / filename, after_dir / filename
        comparison = comparison_dir / filename
        side_by_side = np.hstack((normalized, stripped))
        if not cv2.imwrite(str(before), normalized) or not cv2.imwrite(
            str(after), stripped
        ) or not cv2.imwrite(str(comparison), side_by_side):
            raise OSError(f"Could not write preview {filename}")
        written.append((before, after, comparison))
    return written
