# python_env/bb_report.py
# -*- coding: utf-8 -*-
"""
Build a lightweight report for a black-box run:
- Scans samples/*/*_metrics.json
- Writes artifacts:
    run_report.md
    aggregate_plots/{k90_hist.png, ripple_hist.png, peak_hist.png, fwhm_hist.png, pass_counts.png}
    summary.csv (if missing or requested)

Usage:
  python -m python_env.bb_report --run artifacts/runs/ycbsight_test_2025-10-18T12-00-00Z
  python -m python_env.bb_report --run <RUN_DIR> --rebuild-csv

Notes:
- No pandas dependency (pure stdlib + numpy + matplotlib).
- Safe to run on sequence-mode runs (will just produce a small stub).
"""

from __future__ import annotations
import argparse, json, csv, math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


KEYS = [
    "sample_id", "seq_id", "sample_idx", "pass",
    "k90_ms", "k99_ms", "ripple_pct", "peak",
    "islands", "fwhm_bins", "steps", "amp_used"
]


def _find_sample_metrics(run_dir: Path) -> List[Path]:
    root = run_dir / "samples"
    if not root.exists():
        return []
    return sorted(root.glob("*/*_metrics.json"))


def _read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_csv(rows: List[Dict[str, Any]], out_csv: Path):
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=KEYS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in KEYS})


def _hist(xs: np.ndarray, title: str, xlabel: str, out: Path):
    if xs.size == 0 or np.all(np.isnan(xs)):
        return
    data = xs[~np.isnan(xs)]
    if data.size == 0:
        return
    fig = plt.figure(figsize=(4.8, 3.2))
    plt.hist(data, bins=min(40, max(8, int(len(data) ** 0.5))))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _bar(labels: List[str], counts: List[int], title: str, out: Path):
    fig = plt.figure(figsize=(4.8, 3.2))
    idx = np.arange(len(labels))
    plt.bar(idx, counts)
    plt.xticks(idx, labels, rotation=0)
    plt.title(title)
    plt.ylabel("count")
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def build_report(run_dir: Path, rebuild_csv: bool = False, gallery_n: int = 6):
    run_dir = Path(run_dir)
    metrics_files = _find_sample_metrics(run_dir)
    agg_dir = _ensure_dir(run_dir / "aggregate_plots")

    rows = []
    for jf in metrics_files:
        m = _read_json(jf)
        rows.append({
            "sample_id": m.get("sample_id", jf.stem.replace("_metrics", "")),
            "seq_id": m.get("seq_id", ""),
            "sample_idx": m.get("sample_idx"),
            "pass": bool(m.get("pass", False)),
            "k90_ms": m.get("k90_ms"),
            "k99_ms": m.get("k99_ms"),
            "ripple_pct": m.get("ripple_pct"),
            "peak": m.get("peak", m.get("a_max_ss")),
            "islands": m.get("islands"),
            "fwhm_bins": m.get("fwhm_bins") or m.get("fwhm_px"),
            "steps": m.get("steps"),
            "amp_used": m.get("amp_used"),
            "_pack_dir": jf.parent,
        })

    # write/update summary.csv
    csv_path = run_dir / "summary.csv"
    if rebuild_csv or (not csv_path.exists() and rows):
        _write_csv(rows, csv_path)

    # quick stats
    total = len(rows)
    n_pass = sum(1 for r in rows if r["pass"])
    pass_rate = (100.0 * n_pass / total) if total else 0.0

    # histograms
    k90 = np.array([_to_float(r["k90_ms"]) for r in rows])
    ripple = np.array([_to_float(r["ripple_pct"]) for r in rows])
    peak = np.array([_to_float(r["peak"]) for r in rows])
    fwhm = np.array([_to_float(r["fwhm_bins"]) for r in rows])

    _hist(k90, "k90 rise time", "ms", agg_dir / "k90_hist.png")
    _hist(ripple, "Ripple (steady window)", "%", agg_dir / "ripple_hist.png")
    _hist(peak, "Steady-state peak (a_max_ss)", "a.u.", agg_dir / "peak_hist.png")
    _hist(fwhm, "FWHM (bins)", "bins", agg_dir / "fwhm_hist.png")

    # pass/fail bar
    _bar(["PASS", "FAIL"], [n_pass, total - n_pass], "Certification outcome", agg_dir / "pass_counts.png")

    # pick a small gallery (prefer failures first)
    rows_sorted = sorted(rows, key=lambda r: (r["pass"], _to_float(r["k99_ms"]) or 1e9))
    gallery = rows_sorted[:gallery_n]

    # Markdown report
    md = []
    md.append(f"# Run Report\n")
    md.append(f"- **Run dir:** `{run_dir}`")
    md.append(f"- **Samples:** {total}  |  **PASS:** {n_pass}  |  **Pass rate:** {pass_rate:.1f}%\n")
    if total:
        def _safe_mean(x): 
            x = np.array([_to_float(v) for v in x]); x = x[~np.isnan(x)]
            return float(x.mean()) if x.size else float("nan")
        md.append("## Averages\n")
        md.append(f"- k90_ms: { _safe_mean([r['k90_ms'] for r in rows]):.3f}")
        md.append(f"- k99_ms: { _safe_mean([r['k99_ms'] for r in rows]):.3f}")
        md.append(f"- ripple_pct: { _safe_mean([r['ripple_pct'] for r in rows]):.3f}")
        md.append(f"- peak (a_max_ss): { _safe_mean([r['peak'] for r in rows]):.3f}")
        md.append(f"- FWHM (bins): { _safe_mean([r['fwhm_bins'] for r in rows]):.3f}\n")

    # embed aggregate figures (relative paths)
    md.append("## Aggregate Plots\n")
    for name in ["k90_hist.png", "ripple_hist.png", "peak_hist.png", "fwhm_hist.png", "pass_counts.png"]:
        p = agg_dir / name
        if p.exists():
            md.append(f"![{name}](aggregate_plots/{name})")

    # tiny gallery of samples
    if gallery:
        md.append("\n## Sample Gallery\n")
        for r in gallery:
            pack = Path(r["_pack_dir"])
            base = r["sample_id"]
            rise = pack / f"{base}_rise_curve.png"
            aheat = pack / f"{base}_a_heatmap.png"
            stim = pack / f"{base}_stimulus.png"
            status = "PASS" if r["pass"] else "FAIL"
            md.append(f"**{base}** — {status} | k90={r['k90_ms']} ms | ripple={r['ripple_pct']}% | peak={r['peak']}")
            if rise.exists():
                md.append(f"![]({rise.relative_to(run_dir).as_posix()})")
            if aheat.exists():
                md.append(f"![]({aheat.relative_to(run_dir).as_posix()})")
            elif stim.exists():
                md.append(f"![]({stim.relative_to(run_dir).as_posix()})")

    (run_dir / "run_report.md").write_text("\n\n".join(md) + "\n", encoding="utf-8")
    print(f"Report written: {run_dir/'run_report.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to a black-box run directory.")
    ap.add_argument("--rebuild-csv", action="store_true", help="Force (re)writing summary.csv from sample metrics.")
    ap.add_argument("--gallery", type=int, default=6, help="How many samples to preview in the gallery.")
    args = ap.parse_args()
    build_report(Path(args.run), rebuild_csv=args.rebuild_csv, gallery_n=args.gallery)


if __name__ == "__main__":
    main()
