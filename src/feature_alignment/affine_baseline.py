import argparse
from pathlib import Path

import cv2
import numpy as np


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def load_matched_points(path):
    data = np.load(path)
    input_points = data["input_points"].astype(np.float32)
    reference_points = data["reference_points"].astype(np.float32)
    if len(input_points) < 3 or len(reference_points) < 3:
        raise ValueError("Affine transform requires at least 3 matched point pairs.")
    return input_points, reference_points


def estimate_affine(input_points, reference_points):
    matrix, inliers = cv2.estimateAffinePartial2D(
        input_points,
        reference_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
    )
    if matrix is None:
        raise RuntimeError("Could not estimate affine transform from matched points.")

    linear_part = matrix[:, :2]
    scale = float(np.sqrt((linear_part ** 2).sum(axis=0).mean()))
    if scale < 1e-3:
        raise RuntimeError("Estimated affine transform is degenerate; matched points are not reliable enough.")

    return matrix, inliers


def warp_to_reference(input_image, reference_image, matrix):
    ref_height, ref_width = reference_image.shape[:2]
    return cv2.warpAffine(input_image, matrix, (ref_width, ref_height), flags=cv2.INTER_LINEAR)


def save_side_by_side(reference_image, warped_image, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if reference_image.shape[:2] != warped_image.shape[:2]:
        warped_image = cv2.resize(warped_image, (reference_image.shape[1], reference_image.shape[0]))
    comparison = np.hstack([reference_image, warped_image])
    cv2.imwrite(str(output_path), comparison)


def main():
    parser = argparse.ArgumentParser(description="Warp an input image to a reference image using matched points.")
    parser.add_argument("--reference", required=True, type=Path, help="Path to the reference image.")
    parser.add_argument("--input", required=True, type=Path, help="Path to the input image to be warped.")
    parser.add_argument("--points", required=True, type=Path, help="Path to .npz points from feature_matching.py.")
    parser.add_argument("--output", default=Path("outputs/affine_result.jpg"), type=Path, help="Path for warped image.")
    parser.add_argument(
        "--comparison-output",
        default=Path("outputs/affine_comparison.jpg"),
        type=Path,
        help="Path for reference/warped side-by-side view.",
    )
    args = parser.parse_args()

    reference_image = read_image(args.reference)
    input_image = read_image(args.input)
    input_points, reference_points = load_matched_points(args.points)

    matrix, inliers = estimate_affine(input_points, reference_points)
    warped_image = warp_to_reference(input_image, reference_image, matrix)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), warped_image)
    save_side_by_side(reference_image, warped_image, args.comparison_output)

    inlier_count = int(inliers.sum()) if inliers is not None else 0
    print("affine matrix:")
    print(matrix)
    print(f"matched pairs: {len(input_points)}")
    print(f"ransac inliers: {inlier_count}")
    print(f"saved warped image: {args.output}")
    print(f"saved comparison image: {args.comparison_output}")


if __name__ == "__main__":
    main()
