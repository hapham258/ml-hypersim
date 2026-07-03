import argparse
import glob
import os

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm
import h5py


def load_hdf5(filename):
    """Load the first dataset from a Hypersim HDF5 file."""
    with h5py.File(filename, "r") as f:
        return f[list(f.keys())[0]][()]


def compute_weight(color, diffuse_reflectance, diffuse_illumination, gamma=2.0):
    """
    Compute a non-diffuse confidence map.

    diffuse pixel -> weight ≈ 0
    mirror        -> weight ≈ 1
    shiny metal   -> intermediate
    """
    eps = 1e-6
    diffuse = diffuse_reflectance * diffuse_illumination
    residual = color - diffuse
    residual_intens = np.mean(np.abs(residual), axis=-1)
    color_intens = np.mean(color, axis=-1)
    weight = residual_intens / np.maximum(color_intens, eps)
    weight = weight ** gamma
    return np.clip(weight, 0.0, 1.0)


if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser(description="Compute Hypersim weighting maps.")
    parser.add_argument(
        "--scene_dir",
        required=True,
        help="Path to a Hypersim scene (e.g. ai_001_002)",
    )
    parser.add_argument(
        "--camera",
        required=True,
        help="Camera ID: cam_00, cam_01, cam_02, or cam_03",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Root output directory",
    )
    args = parser.parse_args()

    #
    scene_name = os.path.basename(os.path.abspath(args.scene_dir))
    input_dir = os.path.join(
        args.scene_dir,
        "images",
        f"scene_{args.camera}_final_hdf5",
    )
    output_dir = os.path.join(
        args.output_dir,
        scene_name,
        "images",
        f"scene_{args.camera}_final_weights",
    )
    os.makedirs(output_dir, exist_ok=True)
    color_files = sorted(glob.glob(os.path.join(input_dir, "*.color.hdf5")))
    for color_file in tqdm(color_files):
        base = os.path.basename(color_file).replace(".color.hdf5", "")
        color = load_hdf5(color_file)
        diffuse_reflectance = load_hdf5(
            os.path.join(input_dir, base + ".diffuse_reflectance.hdf5")
        )
        diffuse_illumination = load_hdf5(
            os.path.join(input_dir, base + ".diffuse_illumination.hdf5")
        )
        weight = compute_weight(
            color,
            diffuse_reflectance,
            diffuse_illumination,
        )
        imageio.imwrite(
            os.path.join(output_dir, base + ".weight.png"),
            (weight * 255).astype(np.uint8),
            compress_level=0,
        )
