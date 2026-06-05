#!/usr/bin/env python3
"""Project main entrypoint: run training, prediction or tests from one CLI.

Usage examples:
  python main.py train --data data/images --epochs 5
  python main.py predict --model models/best.h5 --image path/to.jpg
  python main.py test
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(__file__)
TRAIN_SCRIPT = os.path.join(ROOT, "models", "train.py")
PREDICT_SCRIPT = os.path.join(ROOT, "models", "prediction.py")
PIPELINE_SCRIPT = os.path.join(ROOT, "run_pipeline.py")


def run_python_script(script_path, args_list):
    cmd = [sys.executable, script_path] + args_list
    return subprocess.run(cmd, check=False)


def cmd_train(args):
    cmd_args = []
    if args.data:
        cmd_args += ["--data", args.data]
    if args.epochs:
        cmd_args += ["--epochs", str(args.epochs)]
    if args.batch_size:
        cmd_args += ["--batch_size", str(args.batch_size)]
    if args.out:
        cmd_args += ["--out", args.out]
    return run_python_script(TRAIN_SCRIPT, cmd_args)


def cmd_predict(args):
    cmd_args = ["--model", args.model]
    if args.image:
        cmd_args += ["--image", args.image]
    if args.dir:
        cmd_args += ["--dir", args.dir]
    return run_python_script(PREDICT_SCRIPT, cmd_args)


def cmd_test(_args):
    return subprocess.run([sys.executable, "-m", "pytest", "-q"], check=False)


def cmd_pipeline(args):
    cmd_args = []
    if args.data:
        cmd_args += ["--data", args.data]
    if args.out:
        cmd_args += ["--out", args.out]
    if args.epochs:
        cmd_args += ["--epochs", str(args.epochs)]
    if args.batch_size:
        cmd_args += ["--batch_size", str(args.batch_size)]
    if args.logdir:
        cmd_args += ["--logdir", args.logdir]
    if args.visualize_num:
        cmd_args += ["--visualize_num", str(args.visualize_num)]
    if args.gradcam_n:
        cmd_args += ["--gradcam_n", str(args.gradcam_n)]
    if args.out_dir:
        cmd_args += ["--out_dir", args.out_dir]
    if args.timestamp:
        cmd_args.append("--timestamp")
    if args.overwrite:
        cmd_args.append("--overwrite")
    if args.use_generator:
        cmd_args.append("--use_generator")
    return run_python_script(PIPELINE_SCRIPT, cmd_args)


def main():
    parser = argparse.ArgumentParser(prog="main.py")
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch_size", type=int, default=32)
    p_train.add_argument("--out", default="models/best.h5")

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--image")
    p_pred.add_argument("--dir")

    p_test = sub.add_parser("test")

    p_pipe = sub.add_parser("pipeline")
    p_pipe.add_argument("--data")
    p_pipe.add_argument("--out")
    p_pipe.add_argument("--epochs", type=int)
    p_pipe.add_argument("--batch_size", type=int)
    p_pipe.add_argument("--use_generator", action="store_true")
    p_pipe.add_argument("--logdir")
    p_pipe.add_argument("--visualize_num", type=int)
    p_pipe.add_argument("--gradcam_n", type=int)
    p_pipe.add_argument("--out_dir")
    p_pipe.add_argument("--timestamp", action="store_true")
    p_pipe.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.cmd == "train":
        return cmd_train(args)
    if args.cmd == "predict":
        return cmd_predict(args)
    if args.cmd == "test":
        return cmd_test(args)
    if args.cmd == "pipeline":
        return cmd_pipeline(args)

    parser.print_help()


if __name__ == "__main__":
    sys.exit(main())
