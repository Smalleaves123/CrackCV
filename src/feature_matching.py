import argparse
from pathlib import Path

import cv2
import numpy as np


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def extract_orb_matches(reference_image, input_image, max_features, ratio):
    ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    inp_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    ref_keypoints, ref_descriptors = orb.detectAndCompute(ref_gray, None)
    inp_keypoints, inp_descriptors = orb.detectAndCompute(inp_gray, None)

    if ref_descriptors is None or inp_descriptors is None:
        return ref_keypoints or [], inp_keypoints or [], []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(inp_descriptors, ref_descriptors, k=2)

    good_matches = []
    for first, second in knn_matches:
        if first.distance < ratio * second.distance:
            good_matches.append(first)

    good_matches.sort(key=lambda match: match.distance)
    return ref_keypoints, inp_keypoints, good_matches


def matches_to_points(ref_keypoints, inp_keypoints, matches):
    input_points = np.float32([inp_keypoints[m.queryIdx].pt for m in matches])
    reference_points = np.float32([ref_keypoints[m.trainIdx].pt for m in matches])
    return input_points, reference_points


def draw_and_save_matches(reference_image, input_image, ref_keypoints, inp_keypoints, matches, output_path, max_draw):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    drawn = cv2.drawMatches(
        input_image,
        inp_keypoints,
        reference_image,
        ref_keypoints,
        matches[:max_draw],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(output_path), drawn)


def main():
    parser = argparse.ArgumentParser(description="Extract matched ORB keypoints between an input image and a reference image.")
    parser.add_argument("--reference", required=True, type=Path, help="Path to the reference image.")
    parser.add_argument("--input", required=True, type=Path, help="Path to the input image to be warped.")
    parser.add_argument("--output", default=Path("outputs/orb_matches.jpg"), type=Path, help="Path for the match visualization.")
    parser.add_argument("--points-out", default=Path("outputs/orb_points.npz"), type=Path, help="Path for matched point arrays.")
    parser.add_argument("--max-features", default=2000, type=int, help="Maximum number of ORB features.")
    parser.add_argument("--ratio", default=0.75, type=float, help="Lowe ratio threshold for match filtering.")
    parser.add_argument("--max-draw", default=80, type=int, help="Maximum number of matches to draw.")
    args = parser.parse_args()

    reference_image = read_image(args.reference)
    input_image = read_image(args.input)

    ref_keypoints, inp_keypoints, matches = extract_orb_matches(
        reference_image=reference_image,
        input_image=input_image,
        max_features=args.max_features,
        ratio=args.ratio,
    )

    print(f"reference keypoints: {len(ref_keypoints)}")
    print(f"input keypoints: {len(inp_keypoints)}")
    print(f"good matches: {len(matches)}")

    if len(matches) < 3:
        raise RuntimeError("Need at least 3 matched points for affine transform.")

    input_points, reference_points = matches_to_points(ref_keypoints, inp_keypoints, matches)

    args.points_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.points_out, input_points=input_points, reference_points=reference_points)

    draw_and_save_matches(
        reference_image=reference_image,
        input_image=input_image,
        ref_keypoints=ref_keypoints,
        inp_keypoints=inp_keypoints,
        matches=matches,
        output_path=args.output,
        max_draw=args.max_draw,
    )

    print(f"saved match visualization: {args.output}")
    print(f"saved matched points: {args.points_out}")


if __name__ == "__main__":
    main()
