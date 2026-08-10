"""Command-line entry point for CT DICOM HU screening and NMF training."""

import argparse

from dicom_nmf import DEFAULT_BRAIN_HU_RANGE, find_dicom_files, plot_dicom, train_dicom_nmf


def parse_shape(value):
    try:
        rows, columns = (int(item) for item in value.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("shape must use ROWSxCOLUMNS, for example 512x512") from exc
    if rows <= 0 or columns <= 0:
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return rows, columns


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert CT DICOM pixels to HU, remove non-brain intensities, then train NMF."
    )
    parser.add_argument("input", help="DICOM file or directory (searched recursively)")
    parser.add_argument("--output-model", default="dicom_nmf_detector.joblib")
    parser.add_argument("--scores-csv", default="dicom_nmf_scores.csv")
    parser.add_argument("--preview", help="Save a three-panel preview of the first CT slice")
    parser.add_argument("--components", type=int, default=10)
    parser.add_argument("--hu-min", type=float, default=DEFAULT_BRAIN_HU_RANGE[0])
    parser.add_argument("--hu-max", type=float, default=DEFAULT_BRAIN_HU_RANGE[1])
    parser.add_argument("--target-shape", type=parse_shape, metavar="ROWSxCOLUMNS")
    parser.add_argument("--max-iter", type=int, default=500)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.components <= 0:
        raise SystemExit("--components must be positive")
    if args.preview:
        files = find_dicom_files(args.input)
        if not files:
            raise SystemExit("No DICOM files found")
        plot_dicom(files[0], args.preview, args.hu_min, args.hu_max)
    artifact, scores = train_dicom_nmf(
        args.input, args.output_model, args.components, args.hu_min, args.hu_max,
        args.target_shape, args.max_iter
    )
    scores.to_csv(args.scores_csv, index=False)
    print(f"Trained NMF on {len(scores)} HU-screened CT slices with shape {artifact['shape']}")
    print(f"Saved model to {args.output_model} and scores to {args.scores_csv}")


if __name__ == "__main__":
    main()
