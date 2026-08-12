# heatmaps_nmf
Generation of heatmaps for visualising anomaly patterns
This project explores the application of Non-Negative Matrix Factorization (NMF) on the CERN CMS DIALS dataset for anomaly detection and pattern discovery. The primary objective is to decompose high-dimensional detector data into meaningful latent components and visualize the extracted patterns through heatmaps.

Dataset
The project utilizes the CMS DIALS Dataset from the CERN Compact Muon Solenoid (CMS) experiment.

## ThresContCT_TBI skull-stripping preview

The repository implements the paper's **ThresContCT_TBI** preprocessing method:
8-bit CT intensity thresholding (220), 20x20 morphological closing, removal of
contours smaller than 500 pixels, and a Euclidean distance-transform refinement
(10 pixels), followed by removal of residual intensities above 240. These are the
reported defaults and every parameter is available on the command line.

To inspect preprocessing without training NMF, pass a DICOM file/series, a NIfTI
(`.nii` or `.nii.gz`) volume, a JPEG/PNG image, or a directory containing any of
these formats:

```bash
python run_skull_stripping.py /path/to/scan-or-directory \
  --output-dir outputs/threscont-preview
```

The command processes 3-D volumes slice-by-slice and writes matched PNGs beneath
`outputs/threscont-preview/before/` and `outputs/threscont-preview/after/`. It
also writes each before/after pair in one side-by-side frame beneath
`outputs/threscont-preview/comparison/`, with the input on the left and the
skull-stripped result on the right. Raw
medical-image values are first mapped to 8-bit using a configurable CT window
(default -100 to 1000):

```bash
python run_skull_stripping.py scan.nii.gz \
  --window-min -100 --window-max 1000 \
  --skull-threshold 220 --distance-threshold 10
```

JPEG, PNG, BMP, and TIFF images are converted to grayscale and used directly on
their existing 0–255 intensity scale, matching the JPEG inputs evaluated in the
paper. The CT window options therefore apply only to DICOM and NIfTI values, not
to raster images. Lossless PNG or TIFF is preferable when exporting new scans;
JPEG compression may introduce artifacts around high-contrast skull boundaries.

This implementation is a research preprocessing utility, not a clinical or
diagnostic tool. Visually validate the dumps on images from each acquisition
protocol before using the resulting brain-only pixels for NMF.

## CT DICOM brain anomaly pipeline

The repository can also train NMF directly from single-frame CT DICOM slices. Each
slice follows this order of operations:

1. Read the stored pixel data with `pydicom`.
2. Apply DICOM `RescaleSlope` and `RescaleIntercept` to obtain Hounsfield units
   (HU), while excluding pixel-padding values.
3. Keep only a configurable brain soft-tissue interval (default **0 to 100 HU**),
   setting air, padding, and high-density skull/bone pixels to zero.
4. Resize the screened slices to one common shape, flatten them, and train NMF.

Install dependencies and run the pipeline on a DICOM file or a directory:

```bash
python -m pip install -r requirements.txt
python run_dicom_nmf.py /path/to/ct-series \
  --components 10 \
  --hu-min 0 --hu-max 100 \
  --target-shape 256x256 \
  --preview outputs/first-slice.png \
  --output-model outputs/dicom_nmf_detector.joblib \
  --scores-csv outputs/reconstruction_scores.csv
```

The input directory is searched recursively and extensionless DICOM files are
supported. Non-CT objects and multi-frame objects are skipped. The saved model
artifact records the image shape and HU bounds so the same preprocessing can be
applied during inference. The scores CSV contains a reconstruction error for each
slice; larger errors indicate slices represented less well by the trained basis.

The 0–100 HU defaults are a starting point for non-contrast head CT rather than a
clinical segmentation guarantee. Adjust `--hu-min` and `--hu-max` for the scanner,
protocol, and intended analysis, and validate the resulting mask using `--preview`
before training. This research pipeline is not a medical device or diagnostic tool.

### Complete training and testing workflow

Use independent patient groups to avoid data leakage. A recommended directory
layout is:

```text
data/
  train_normal/       # normal patients used to learn the NMF basis
  validation_normal/  # different normal patients used only for the threshold
  test/               # unseen normal and/or abnormal patients
```

Do not put slices from one patient in more than one directory. Install the
dependencies, then train, calibrate, and test in one command:

```bash
python -m pip install -r requirements.txt
python run_dicom_nmf.py data/train_normal \
  --validation-input data/validation_normal \
  --test-input data/test \
  --components 10 \
  --hu-min 0 --hu-max 100 \
  --target-shape 256x256 \
  --threshold-percentile 95 \
  --preview outputs/training-preview.png \
  --output-model outputs/dicom_nmf_detector.joblib \
  --scores-csv outputs/train_scores.csv \
  --validation-scores-csv outputs/validation_scores.csv \
  --test-scores-csv outputs/test_scores.csv
```

The command performs these operations in order:

1. It recursively reads each training CT DICOM, converts stored pixels to HU,
   applies the brain HU mask, resizes each screened slice, and trains NMF.
2. It applies the identical saved HU bounds and image shape to held-out normal
   validation slices. The selected percentile of their reconstruction errors
   becomes the anomaly threshold.
3. It saves that threshold in the model artifact and applies the same pipeline to
   every test slice.
4. It writes `reconstruction_error` and `predicted_anomaly` to the test CSV. A
   prediction is true when its reconstruction error is at or above the calibrated
   threshold.

Review `outputs/training-preview.png` before accepting a run: the third panel
should retain the intended brain tissue and suppress the skull. Inspect the score
distributions as well; the percentile is a configurable operating point, not a
guarantee that five percent of future scans will be abnormal. If ground-truth test
labels are available, join them to `test_scores.csv` by `path` and calculate
patient-level sensitivity, specificity, ROC AUC, and precision-recall AUC. Slice
scores should not be treated as independent patient diagnoses.
