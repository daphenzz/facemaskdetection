#!/usr/bin/env python3
import argparse
from .trainer import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out", default="models/best.h5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_generator", action="store_true")
    parser.add_argument("--logdir")
    args = parser.parse_args()

    res = train(data_path=args.data, out=args.out, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, use_generator=args.use_generator, logdir=args.logdir)
    print(res)


if __name__ == "__main__":
    main()
