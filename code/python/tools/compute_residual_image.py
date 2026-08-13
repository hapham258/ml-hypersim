import os
import glob
import argparse

import h5py
import numpy as np
from tqdm import tqdm
import imageio
import OpenEXR
import Imath


def load_hdf5(filename):
    """Load the first dataset from a Hypersim HDF5 file."""
    with h5py.File(filename, "r") as f:
        return f[list(f.keys())[0]][()]


def save_hdf5(filename, array):
    """Save an array to an HDF5 file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with h5py.File(filename, "w") as f:
        f.create_dataset("dataset", data=array, compression="gzip")


def tone_map(img):
    """
    Compress HDR values into the displayable range [0, 1]. Use Reinhard tone mapping:
        L_out = L_in / (1 + L_in)
    """
    img = np.clip(img, 0.0, None)
    return img / (1.0 + img)


def gamma_correct(img):
    """
    Do gamma correction. Assumes the input is already in the range [0, 1].
    """
    img = np.clip(img, 0.0, 1.0)
    mask = img <= 0.0031308
    srgb = np.empty_like(img)
    srgb[mask] = 12.92 * img[mask]
    srgb[~mask] = 1.055 * np.power(img[~mask], 1.0 / 2.4) - 0.055
    return np.clip(srgb, 0.0, 1.0)


def save_jpg(filename, img, tonemap="hypersim", valid_mask=None, quality=95):
    """
    Save a linear RGB image as JPEG.

    tonemap:
        "hypersim" : automatic exposure (Hypersim official preview)
        "reinhard" : Reinhard tone mapping + sRGB conversion
    """
    img = np.maximum(img, 0.0)
    if tonemap == "hypersim":
        gamma = 1.0 / 2.2
        percentile = 90
        target = 0.8
        if valid_mask is None:
            valid_mask = np.ones(img.shape[:2], dtype=bool)
        luminance = (
            0.30 * img[..., 0] + 0.59 * img[..., 1] + 0.11 * img[..., 2]
        )  # CCIR601 coefficients
        luminance = luminance[valid_mask]
        if luminance.size == 0:
            scale = 1.0
        else:
            p = np.percentile(luminance, percentile)
            scale = 0.0 if p < 1e-4 else target ** (1 / gamma) / p
        img = np.power(scale * img, gamma)
    elif tonemap == "reinhard":
        img = gamma_correct(tone_map(img))
    else:
        raise ValueError(f"Unknown tonemap: {tonemap}")
    img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    imageio.imwrite(filename, img, quality=quality)


def save_exr(filename, img):
    """Save a float16 RGB image as an EXR file."""
    img = np.asarray(img, dtype=np.float16)
    height, width, channels = img.shape
    assert channels == 3
    header = OpenEXR.Header(width, height)
    pixel_type = Imath.PixelType(Imath.PixelType.HALF)
    header["channels"] = {
        "R": Imath.Channel(pixel_type),
        "G": Imath.Channel(pixel_type),
        "B": Imath.Channel(pixel_type),
    }
    out = OpenEXR.OutputFile(filename, header)
    out.writePixels(
        {
            "R": img[:, :, 0].tobytes(),
            "G": img[:, :, 1].tobytes(),
            "B": img[:, :, 2].tobytes(),
        }
    )
    out.close()


def print_statistics(color, diffuse_reflectance, diffuse_illumination, residual):
    """Print statistics for a color/decomposition tuple."""

    channel_names = ["R", "G", "B"]

    def stats(name, x):
        print(f"{name}:")
        print(f"  shape = {x.shape}")
        print(f"  dtype = {x.dtype}")
        for c, cname in enumerate(channel_names):
            xc = x[..., c]
            print(f"  {cname}:")
            print(
                f"    min={xc.min():.6f}  max={xc.max():.6f}  mean={xc.mean():.6f}  std={xc.std():.6f}"
            )
            print(
                "    percentiles = "
                f"[0% {np.percentile(xc,0):.6f}, "
                f"1% {np.percentile(xc,1):.6f}, "
                f"50% {np.percentile(xc,50):.6f}, "
                f"99% {np.percentile(xc,99):.6f}, "
                f"100% {np.percentile(xc,100):.6f}]"
            )
        print()

    print("=" * 80)
    print("Statistics of frame")
    print("=" * 80)
    stats("Color", color)
    stats("Diffuse Reflectance", diffuse_reflectance)
    stats("Diffuse Illumination", diffuse_illumination)
    stats("Residual", residual)
    for c, name in enumerate(channel_names):
        print(
            f"{name}: +energy={np.mean(np.maximum(residual[..., c], 0)):.6f}  "
            f"-energy={np.mean(np.maximum(-residual[..., c], 0)):.6f}"
        )


def process_camera(scene_dir, camera, output_dir, num_imgs=None, seed=None):
    """
    Compute residual = color - diffuse_reflectance * diffuse_illumination.
    """
    #
    camera_name = f"scene_{camera}_final_hdf5"
    input_dir = os.path.join(scene_dir, "images", camera_name)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Cannot find camera directory:\n{input_dir}")
    scene_name = os.path.basename(os.path.normpath(scene_dir))
    output_cam_dir = os.path.join(
        output_dir,
        scene_name,
        "images",
        camera_name,
    )
    os.makedirs(output_cam_dir, exist_ok=True)
    color_files = sorted(glob.glob(os.path.join(input_dir, "*.color.hdf5")))
    print(f"Processing {scene_name} ({camera})")
    print(f"Found {len(color_files)} frames.")
    if num_imgs is not None and num_imgs < len(color_files):
        rng = np.random.default_rng(seed)
        color_files = list(rng.choice(color_files, num_imgs, replace=False))
        print(f"Randomly selected {len(color_files)} frames.")

    #
    for idx, color_file in enumerate(tqdm(color_files)):
        #
        base = os.path.basename(color_file).replace(".color.hdf5", "")
        refl_file = os.path.join(
            input_dir,
            base + ".diffuse_reflectance.hdf5",
        )
        if not os.path.exists(refl_file):
            print(f"Missing: {refl_file}")
            continue
        illum_file = os.path.join(
            input_dir,
            base + ".diffuse_illumination.hdf5",
        )
        if not os.path.exists(illum_file):
            print(f"Missing: {illum_file}")
            continue

        #
        color = load_hdf5(color_file).astype(np.float32)
        diffuse_reflectance = load_hdf5(refl_file).astype(np.float32)
        diffuse_illumination = load_hdf5(illum_file).astype(np.float32)
        residual = color - diffuse_reflectance * diffuse_illumination
        positive_residual = np.maximum(residual, 0.0)
        negative_residual = np.maximum(-residual, 0.0)
        if idx == 0:
            print_statistics(color, diffuse_reflectance, diffuse_illumination, residual)

        #
        save_hdf5(os.path.join(output_cam_dir, base + ".residual.hdf5"), residual)
        save_jpg(
            os.path.join(output_cam_dir, base + ".color.jpg"),
            color,
        )
        save_jpg(
            os.path.join(output_cam_dir, base + ".diffuse_reflectance.jpg"),
            diffuse_reflectance,
        )
        save_jpg(
            os.path.join(output_cam_dir, base + ".diffuse_illumination.jpg"),
            diffuse_illumination,
        )
        save_jpg(
            os.path.join(output_cam_dir, base + ".positive_residual.jpg"),
            positive_residual,
        )
        save_jpg(
            os.path.join(output_cam_dir, base + ".negative_residual.jpg"),
            negative_residual,
        )
        save_exr(
            os.path.join(output_cam_dir, base + ".color.exr"),
            color,
        )
        save_exr(
            os.path.join(output_cam_dir, base + ".diffuse_reflectance.exr"),
            diffuse_reflectance,
        )
        save_exr(
            os.path.join(output_cam_dir, base + ".diffuse_illumination.exr"),
            diffuse_illumination,
        )
        save_exr(
            os.path.join(output_cam_dir, base + ".residual.exr"),
            residual,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Hypersim residual images.")
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
    parser.add_argument(
        "--num_imgs",
        type=int,
        default=10,
        help="Randomly process only N images. If omitted, process all images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for image selection. Default: None.",
    )
    args = parser.parse_args()
    process_camera(
        scene_dir=args.scene_dir,
        camera=args.camera,
        output_dir=args.output_dir,
        num_imgs=args.num_imgs,
        seed=args.seed,
    )
