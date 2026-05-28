import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from affine_baseline import estimate_affine
from feature_matching import draw_and_save_matches, extract_feature_matches, matches_to_points, read_image


def create_synthetic_warp(image):
    height, width = image.shape[:2]
    angle = 8.0
    scale = 0.95
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[0, 2] += width * 0.04
    matrix[1, 2] -= height * 0.03
    warped = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped, matrix


def affine_error(estimated_matrix, true_matrix):
    return float(np.linalg.norm(estimated_matrix - true_matrix))


def save_summary(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)


def main():
    parser = argparse.ArgumentParser(description="Evaluate feature matching on a synthetic affine warp.")
    parser.add_argument("--image", required=True, type=Path, help="Source image used as the reference.")
    parser.add_argument("--output-dir", default=Path("outputs/synthetic_feature_test"), type=Path)
    parser.add_argument("--method", default="orb", choices=["orb", "sift"])
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--max-features", default=2000, type=int)
    parser.add_argument("--ratio", default=0.75, type=float)
    args = parser.parse_args()

    reference_image = read_image(args.image)
    input_image, reference_to_input_matrix = create_synthetic_warp(reference_image)
    input_to_reference_matrix = cv2.invertAffineTransform(reference_to_input_matrix)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_input_path = args.output_dir / "synthetic_input.jpg"
    cv2.imwrite(str(synthetic_input_path), input_image)

    ref_keypoints, inp_keypoints, matches = extract_feature_matches(
        reference_image=reference_image,
        input_image=input_image,
        method=args.method,
        max_features=args.max_features,
        ratio=args.ratio,
        use_clahe=args.clahe,
    )

    if len(matches) < 3:
        raise RuntimeError("Need at least 3 matched points for affine evaluation.")

    input_points, reference_points = matches_to_points(ref_keypoints, inp_keypoints, matches)
    estimated_matrix, inliers = estimate_affine(input_points, reference_points)
    inlier_count = int(inliers.sum()) if inliers is not None else 0

    points_path = args.output_dir / "synthetic_points.npz"
    np.savez(
        points_path,
        input_points=input_points,
        reference_points=reference_points,
        true_matrix=input_to_reference_matrix,
        estimated_matrix=estimated_matrix,
    )

    draw_and_save_matches(
        reference_image=reference_image,
        input_image=input_image,
        ref_keypoints=ref_keypoints,
        inp_keypoints=inp_keypoints,
        matches=matches,
        output_path=args.output_dir / "synthetic_matches.jpg",
        max_draw=80,
    )

    summary = {
        "method": args.method,
        "clahe": args.clahe,
        "reference_keypoints": len(ref_keypoints),
        "input_keypoints": len(inp_keypoints),
        "good_matches": len(matches),
        "ransac_inliers": inlier_count,
        "inlier_ratio": inlier_count / len(matches),
        "affine_matrix_error": affine_error(estimated_matrix, input_to_reference_matrix),
    }
    save_summary(args.output_dir / "summary.csv", summary)

    print(f"method: {args.method}")
    print(f"clahe: {args.clahe}")
    print(f"reference keypoints: {len(ref_keypoints)}")
    print(f"input keypoints: {len(inp_keypoints)}")
    print(f"good matches: {len(matches)}")
    print(f"ransac inliers: {inlier_count}")
    print(f"inlier ratio: {summary['inlier_ratio']:.4f}")
    print(f"affine matrix error: {summary['affine_matrix_error']:.4f}")
    print(f"saved synthetic input: {synthetic_input_path}")
    print(f"saved match visualization: {args.output_dir / 'synthetic_matches.jpg'}")
    print(f"saved summary: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
