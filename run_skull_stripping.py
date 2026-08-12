"""Dump ThresContCT_TBI preprocessing previews without training NMF."""

import argparse

from skull_stripping import ThresContConfig, dump_threscont_previews


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Apply ThresContCT_TBI skull stripping and dump before, after, and "
            "side-by-side PNGs."
        )
    )
    parser.add_argument(
        "input", help="DICOM, NIfTI, JPEG/PNG image, or directory"
    )
    parser.add_argument("--output-dir", default="threscont_previews")
    parser.add_argument("--skull-threshold", type=int, default=220)
    parser.add_argument("--closing-kernel", type=int, default=20)
    parser.add_argument("--minimum-contour-area", type=float, default=500.0)
    parser.add_argument("--distance-threshold", type=float, default=10.0)
    parser.add_argument("--artifact-threshold", type=int, default=240)
    parser.add_argument("--window-min", type=float, default=-100.0)
    parser.add_argument("--window-max", type=float, default=1000.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = ThresContConfig(
        skull_threshold=args.skull_threshold,
        closing_kernel=args.closing_kernel,
        minimum_contour_area=args.minimum_contour_area,
        distance_threshold=args.distance_threshold,
        artifact_threshold=args.artifact_threshold,
        window_min=args.window_min,
        window_max=args.window_max,
    )
    written = dump_threscont_previews(args.input, args.output_dir, config)
    print(f"Wrote {len(written)} before/after comparisons to {args.output_dir}")


if __name__ == "__main__":
    main()
