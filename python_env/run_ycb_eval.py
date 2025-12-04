# python_env/run_ycb_eval.py
# -*- coding: utf-8 -*-
"""
Convenience CLI for evaluating the YCBSight dataset with the DNF black box.

Examples
--------
# Sequence mode (stream frames; one row per sequence)
python -m python_env.run_ycb_eval \
  --config configs/config.yaml \
  --cert configs/probe.yaml \
  --datasets configs/datasets.yaml \
  --split test

# Per-sample mode (full sample packs; one pack per image)
python -m python_env.run_ycb_eval \
  --config configs/config.yaml \
  --cert configs/probe.yaml \
  --datasets configs/datasets.yaml \
  --split test \
  --per-sample --sample-steps 600 --out artifacts/runs
"""

from __future__ import annotations
import argparse
from pathlib import Path
from python_env.bb_run import run_black_box


def main():
    ap = argparse.ArgumentParser(description="YCBSight evaluation wrapper.")
    ap.add_argument("--config", default="configs/config.yaml",
                    help="DNF params YAML (default: configs/config.yaml)")
    ap.add_argument("--cert", default="configs/probe.yaml",
                    help="Certification gates YAML (default: configs/probe.yaml)")
    ap.add_argument("--datasets", default="configs/datasets.yaml",
                    help="Datasets registry YAML (default: configs/datasets.yaml)")
    ap.add_argument("--split", default="test", help="Dataset split (default: test)")
    ap.add_argument("--out", default="artifacts/runs",
                    help="Output directory root (default: artifacts/runs)")
    ap.add_argument("--dataset", default="ycbsight",
                    help="Dataset key from datasets.yaml (default: ycbsight)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Limit frames per sequence (debugging).")
    ap.add_argument("--ema-alpha", type=float, default=None,
                    help="EMA smoothing for sequence mode only.")
    ap.add_argument("--per-sample", action="store_true",
                    help="Evaluate each image independently and emit sample packs.")
    ap.add_argument("--sample-steps", type=int, default=600,
                    help="DNF steps per image in per-sample mode (default: 600).")
    ap.add_argument("--print-every", type=int, default=10,
                    help="Log every N samples in per-sample mode.")
    ap.add_argument("--save-traces", action="store_true",
                    help="Also write legacy CSV traces per image.")
    ap.add_argument("--quiet", action="store_true", help="Reduce console logs.")
    args = ap.parse_args()

    run_black_box(
        config_file=Path(args.config),
        cert_file=Path(args.cert),
        datasets_cfg=Path(args.datasets),
        dataset_name=args.dataset,
        split=args.split,
        out_dir=Path(args.out),
        max_frames=args.max_frames,
        ema_alpha=args.ema_alpha,
        verbose=not args.quiet,
        per_sample=args.per_sample,
        sample_steps=args.sample_steps,
        print_every=args.print_every,
        save_traces=args.save_traces,
    )


if __name__ == "__main__":
    main()
