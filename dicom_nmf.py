"""DICOM CT preprocessing and NMF training utilities.

The pipeline converts stored DICOM pixels to Hounsfield units (HU), masks values
outside a configurable brain-tissue range, and only then constructs the feature
matrix used to fit NMF.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom

from nmf import build_feature_matrix, train_nmf_detector


DEFAULT_BRAIN_HU_RANGE = (0.0, 100.0)


def pixels_to_hounsfield(dataset, pixels=None):
    """Return a DICOM pixel array converted to Hounsfield units.

    Rescale Slope and Rescale Intercept are part of the CT modality transform.
    Pixel-padding values are represented as NaN so they cannot become tissue.
    """
    stored = np.asarray(dataset.pixel_array if pixels is None else pixels)
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    hu = stored.astype(np.float32) * slope + intercept

    if hasattr(dataset, "PixelPaddingValue"):
        padding = stored == float(dataset.PixelPaddingValue)
        if hasattr(dataset, "PixelPaddingRangeLimit"):
            low, high = sorted(
                (float(dataset.PixelPaddingValue), float(dataset.PixelPaddingRangeLimit))
            )
            padding = (stored >= low) & (stored <= high)
        hu[padding] = np.nan
    return hu


def apply_brain_hu_mask(hu_image, hu_min=0.0, hu_max=100.0, fill_value=0.0):
    """Keep likely brain soft tissue and remove air, bone/skull, and padding.

    The defaults are intentionally configurable because acquisition protocols and
    clinical tasks differ. The returned data are non-negative and suitable for NMF.
    """
    if hu_min > hu_max:
        raise ValueError("hu_min must be less than or equal to hu_max")
    hu = np.asarray(hu_image, dtype=np.float32)
    keep = np.isfinite(hu) & (hu >= hu_min) & (hu <= hu_max)
    return np.where(keep, hu, fill_value).astype(np.float32), keep


def read_dicom(path, hu_min=0.0, hu_max=100.0):
    """Read one CT DICOM file and return its dataset, HU pixels, mask, and result."""
    path = Path(path)
    dataset = pydicom.dcmread(path)
    modality = str(getattr(dataset, "Modality", "")).upper()
    if modality and modality != "CT":
        raise ValueError(f"{path} is modality {modality!r}; Hounsfield conversion requires CT")
    hu = pixels_to_hounsfield(dataset)
    if hu.ndim != 2:
        raise ValueError(
            f"{path} has {hu.ndim} dimensions; "
            "only single-frame CT slices are supported"
        )
    screened, mask = apply_brain_hu_mask(hu, hu_min=hu_min, hu_max=hu_max)
    return dataset, hu, mask, screened


def find_dicom_files(input_path):
    """Find readable DICOM files below a file or directory, including extensionless files."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    candidates = (
        [input_path]
        if input_path.is_file()
        else sorted(path for path in input_path.rglob("*") if path.is_file())
    )
    files = []
    for candidate in candidates:
        try:
            if pydicom.misc.is_dicom(candidate):
                files.append(candidate)
        except (OSError, ValueError):
            continue
    return files


def load_dicom_dataframe(input_path, hu_min=0.0, hu_max=100.0):
    """Load and HU-screen CT slices into the DataFrame shape expected by NMF."""
    records = []
    errors = []
    for path in find_dicom_files(input_path):
        try:
            dataset, _, mask, screened = read_dicom(path, hu_min, hu_max)
            records.append(
                {
                    "path": str(path),
                    "patient_id": str(getattr(dataset, "PatientID", "")),
                    "study_uid": str(getattr(dataset, "StudyInstanceUID", "")),
                    "series_uid": str(getattr(dataset, "SeriesInstanceUID", "")),
                    "instance_number": int(getattr(dataset, "InstanceNumber", len(records))),
                    "brain_pixel_fraction": float(mask.mean()),
                    "data": screened,
                }
            )
        except (ValueError, AttributeError, NotImplementedError, OSError) as exc:
            errors.append(f"{path}: {exc}")
    if not records:
        detail = f" ({'; '.join(errors)})" if errors else ""
        raise ValueError(f"No usable single-frame CT DICOM images found in {input_path}{detail}")
    return pd.DataFrame(records).sort_values(
        ["study_uid", "series_uid", "instance_number"], ignore_index=True
    )


def plot_dicom(path, output_path=None, hu_min=0.0, hu_max=100.0, show=False):
    """Plot stored pixels, calibrated HU, and the brain-only screened image."""
    dataset, hu, _, screened = read_dicom(path, hu_min, hu_max)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    panels = (
        (np.asarray(dataset.pixel_array), "Stored pixels", "gray"),
        (hu, "Hounsfield units", "gray"),
        (screened, f"Brain HU mask [{hu_min:g}, {hu_max:g}]", "gray"),
    )
    for axis, (image, title, cmap) in zip(axes, panels):
        rendered = axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(Path(path).name)
    fig.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def train_dicom_nmf(input_path, output_model, n_components=10, hu_min=0.0,
                    hu_max=100.0, target_shape=None, max_iter=500):
    """HU-screen every input CT slice, build features, train NMF, and save it."""
    frame = load_dicom_dataframe(input_path, hu_min=hu_min, hu_max=hu_max)
    if n_components > len(frame):
        raise ValueError(
            f"n_components ({n_components}) cannot exceed CT slice count ({len(frame)})"
        )
    features, metadata, shape = build_feature_matrix(
        frame, target_shape=target_shape, force_nonnegative=True, feature_transform="flatten"
    )
    model, errors = train_nmf_detector(
        features, n_components=n_components, max_iter=max_iter
    )
    artifact = {
        "nmf": model,
        "shape": shape,
        "hu_min": float(hu_min),
        "hu_max": float(hu_max),
        "feature_transform": "flatten",
    }
    output_model = Path(output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_model)
    result = metadata.drop(columns=["data"]).copy()
    result["reconstruction_error"] = errors
    return artifact, result
