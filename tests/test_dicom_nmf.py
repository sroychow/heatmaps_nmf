from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from dicom_nmf import apply_brain_hu_mask, calibrate_threshold, pixels_to_hounsfield
from run_dicom_nmf import parse_shape


def test_pixels_are_converted_to_hounsfield_and_padding_is_removed():
    dataset = SimpleNamespace(
        RescaleSlope=2, RescaleIntercept=-1024, PixelPaddingValue=-2000
    )
    stored = np.array([[-2000, 512], [562, 612]], dtype=np.int16)

    hu = pixels_to_hounsfield(dataset, pixels=stored)

    assert np.isnan(hu[0, 0])
    np.testing.assert_array_equal(hu[~np.isnan(hu)], [0, 100, 200])


def test_brain_mask_removes_air_and_skull_before_nmf():
    hu = np.array([[-1000, -1, 0], [20, 100, 101], [np.nan, 50, 1000]])

    screened, mask = apply_brain_hu_mask(hu, hu_min=0, hu_max=100)

    np.testing.assert_array_equal(screened, [[0, 0, 0], [20, 100, 0], [0, 50, 0]])
    assert mask.sum() == 4
    assert screened.min() >= 0


def test_invalid_hu_range_is_rejected():
    with pytest.raises(ValueError, match="hu_min"):
        apply_brain_hu_mask(np.ones((2, 2)), hu_min=100, hu_max=0)


def test_target_shape_parser():
    assert parse_shape("256x512") == (256, 512)
    with pytest.raises(Exception):
        parse_shape("bad")


def test_threshold_uses_held_out_normal_score_percentile():
    assert calibrate_threshold([1, 2, 3, 4, 5], percentile=80) == pytest.approx(4.2)
    with pytest.raises(ValueError, match="between 0 and 100"):
        calibrate_threshold([1, 2], percentile=100)
