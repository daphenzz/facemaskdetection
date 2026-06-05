#!/usr/bin/env python3
"""Run full pipeline: train -> evaluate -> visualize -> grad-cam samples.

Usage example:
  python run_pipeline.py --data data/images --out models/best.h5 --epochs 5
"""
import argparse
from datetime import datetime
import os
import glob
import logging
import subprocess
import sys

from models.trainer import train
from data.dataset import get_splits
from models.evaluate import evaluate_model_on_data
from models.visualize import visualize_predictions
from models.gradcam import grad_cam


def collect_sample_images(root: str, n: int):
    files = []
    for root_dir, dirs, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                files.append(os.path.join(root_dir, fn))
    files = sorted(files)
    return files[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset root (class subfolders)")
    parser.add_argument("--out", default="models/best.h5", help="Output model path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_generator", action="store_true")
    parser.add_argument("--logdir", help="TensorBoard logdir")
    parser.add_argument("--visualize_num", type=int, default=6, help="Number of images to show in montage")
    parser.add_argument("--gradcam_n", type=int, default=3, help="Number of grad-cam images to create")
    parser.add_argument("--out_dir", default="outputs", help="Directory to save outputs")
    parser.add_argument("--timestamp", action="store_true", help="Create a timestamped results/<TIMESTAMP> output directory")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing out_dir")
    args = parser.parse_args()

    if args.timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = os.path.join("results", ts)

    if os.path.exists(args.out_dir) and not args.overwrite:
        raise SystemExit(f"Out dir {args.out_dir} already exists (use --overwrite to replace)")

    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("pipeline")

    logger.info("Starting training...")
    metrics = train(data_path=args.data, out=args.out, epochs=args.epochs, batch_size=args.batch_size, seed=42, use_generator=args.use_generator, logdir=args.logdir)
    logger.info(f"Training finished: {metrics}")

    logger.info("Loading test split for evaluation...")
    _, _, X_test, _, _, y_test, classes = get_splits(args.data)

    logger.info("Evaluating model on test set...")
    eval_res = evaluate_model_on_data(args.out, X_test, y_test, classes, out_prefix=os.path.join(args.out_dir, "eval"))
    logger.info(f"Eval outputs: {eval_res}")

    logger.info("Creating prediction montage...")
    vis_out = visualize_predictions(args.out, args.data, out_path=os.path.join(args.out_dir, "maskeli_dogru.png"), num=args.visualize_num)
    logger.info(f"Saved montage to {vis_out}")

    # Also generate grouped montages (mask/correct/wrong) using the script
    try:
        script_path = os.path.join(os.path.dirname(__file__), "scripts", "generate_montages.py")
        cmd = [sys.executable, script_path, "--model", args.out, "--data", args.data, "--out_dir", args.out_dir]
        logger.info(f"Running montage script: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except Exception as e:
        logger.warning(f"Failed to run montage script: {e}")

    if args.gradcam_n > 0:
        logger.info("Creating grad-cam overlays for sample images...")
        samples = collect_sample_images(args.data, args.gradcam_n)
        gc_paths = []
        for i, s in enumerate(samples):
            outp = os.path.join(args.out_dir, f"gradcam_{i}.png")
            try:
                grad_cam(args.out, s, out_path=outp)
                gc_paths.append(outp)
                logger.info(f"Saved grad-cam: {outp}")
            except Exception as e:
                logger.warning(f"Grad-cam failed for {s}: {e}")

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
