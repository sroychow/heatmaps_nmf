import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from skull_stripping import (
    ThresContConfig,
    dump_threscont_previews,
    normalize_ct_slice,
    threscont_ct_tbi,
)


def test_ct_window_is_mapped_to_eight_bit():
    image = np.array([[-100, 450, 1000, np.nan]], dtype=float)
    result = normalize_ct_slice(image, -100, 1000)
    np.testing.assert_array_equal(result, [[0, 128, 255, 0]])


def test_threscont_keeps_enclosed_brain_and_removes_skull():
    image = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (89, 89), 230, thickness=5)
    image[20:80, 20:80] = 100
    config = ThresContConfig(
        closing_kernel=3, minimum_contour_area=100, distance_threshold=3
    )

    stripped, mask, _ = threscont_ct_tbi(image, config, already_normalized=True)

    assert stripped[50, 50] == 100
    assert stripped[10, 50] == 0
    assert mask[50, 50]
    assert not mask[0, 0]


def test_threscont_rejects_non_slice_input():
    with pytest.raises(ValueError, match="two-dimensional"):
        threscont_ct_tbi(np.zeros((2, 2, 2)), already_normalized=True)


@pytest.mark.parametrize("extension", ["png", "jpg"])
def test_raster_input_is_dumped_without_ct_windowing(tmp_path, extension):
    source = tmp_path / f"scan.{extension}"
    image = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (89, 89), 230, thickness=5)
    image[20:80, 20:80] = 100
    assert cv2.imwrite(str(source), image)
    config = ThresContConfig(
        closing_kernel=3, minimum_contour_area=100, distance_threshold=3
    )

    written = dump_threscont_previews(source, tmp_path / "dump", config)

    assert len(written) == 1
    before = cv2.imread(str(written[0][0]), cv2.IMREAD_GRAYSCALE)
    after = cv2.imread(str(written[0][1]), cv2.IMREAD_GRAYSCALE)
    assert before[50, 50] == pytest.approx(100, abs=5)
    assert after[50, 50] == pytest.approx(100, abs=5)
    assert after[10, 50] == 0
