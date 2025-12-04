# python_env/bb_run.py
# -*- coding: utf-8 -*-
"""
Black-box runner for the DNF inner loop.

Modes
-----
1) Sequence mode (default): stream varying I_ext over time per folder/sequence,
   produce one metrics row per sequence.
2) Per-sample mode (--per-sample): treat each image as an independent run:
   reset DNF, feed constant I_ext for --sample-steps, produce one row + plot
   (and optional trace) per image, including 'amp_used' (actual input amplitude).

Outputs (under artifacts/runs/<dataset>_<split>_<timestamp>/)
----------------------------------------------------------------
copies/{config.yaml, certification.yaml, datasets.yaml}
summary.json
# Sequence mode:
<dataset>_<split>_metrics.csv, registry.jsonl
# Per-sample mode:
<dataset>_<split>_samples.csv, plots/<seq_id>/sample_XXXX.pdf
(optional) plots/<seq_id>/sample_XXXX_trace.csv with --save-traces
Additionally, in per-sample mode this script now emits a full "sample pack" in:
samples/<seq_id>/sample_XXXX/
  ├─ *_fields.npz (t_ms, u, a, stim, meta)
  ├─ *_stimulus.png
  ├─ *_rise_curve.png
  ├─ *_a_heatmap.png
  ├─ *_u_heatmap.png
  ├─ *_bump_profile.png
  ├─ *_metrics.json
  └─ manifest.json
"""

from __future__ import annotations
import argparse, json, csv, sys, hashlib
from pathlib import Path
from datetime import datetime
import importlib

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from python_env.bb_config import load_config, load_cert, config_hash
from python_env.bb_core import step_dnf_stream
from python_env.bb_metrics import certify_sequence
import python_env.adapters.registry as reg


# ----------------------- small utilities -----------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")

def _iter_sequences(adapter, split: str):
    for seq in adapter.iter_sequences(split):
        yield seq

def _iter_Iext_for_seq(adapter, seq, cfg: dict, max_frames: int | None):
    n = 0
    for frame in adapter.iter_frames(seq):
        yield adapter.frame_to_Iext(frame, cfg)
        n += 1
        if max_frames is not None and n >= max_frames:
            break

def _repeat_I(I: np.ndarray, steps: int):
    for _ in range(int(steps)):
        yield I

def _reasons_from_metrics(m: dict, gates: dict) -> list[str]:
    """Human-readable failure reasons."""
    reasons = []
    # Prefer boolean flags if present
    for key, msg in [
        ("peak_pass", "peak < min_peak"),
        ("ripple_pass", "ripple > max_ripple"),
        ("k90_pass", "k90 > gate"),
        ("k99_pass", "k99 > gate"),
        ("islands_pass", "islands > gate"),
        ("fwhm_pass", "FWHM > gate"),
    ]:
        if key in m and not m[key]:
            reasons.append(msg)

    # If no booleans, infer from values/gates
    if not reasons:
        if "peak" in m and "min_peak" in gates and m["peak"] < gates["min_peak"]:
            reasons.append(f"peak={m['peak']:.3f} < {gates['min_peak']:.3f}")
        if "ripple_pct" in m and "max_ripple" in gates:
            gate_pct = gates["max_ripple"]*100.0 if gates["max_ripple"] <= 1.0 else gates["max_ripple"]
            if m["ripple_pct"] > gate_pct:
                reasons.append(f"ripple={m['ripple_pct']:.2f}% > {gate_pct:.2f}%")
        if "k90_ms" in m and "k90_ms" in gates and m["k90_ms"] > gates["k90_ms"]:
            reasons.append(f"k90={m['k90_ms']:.1f}ms > {gates['k90_ms']:.1f}ms")
        if "k99_ms" in m and "k99_ms" in gates and m["k99_ms"] > gates["k99_ms"]:
            reasons.append(f"k99={m['k99_ms']:.1f}ms > {gates['k99_ms']:.1f}ms")
    return reasons


# ---------------------- per-sample artifact helpers ----------------------

def _compute_steady(a_max, t_ms, frac=0.2, min_ms=1.0):
    tail = max(int(len(t_ms)*frac), int(min_ms/(t_ms[1]-t_ms[0])))
    aa = a_max[-tail:]
    return float(np.median(aa)), a_max.size - tail

def _first_time(t_ms, cond):
    idx = np.argmax(cond)  # returns 0 if all False
    return None if not cond.any() else float(t_ms[idx])

def _kx_time(a_max, t_ms, a_ss, x):
    return _first_time(t_ms, a_max >= a_ss*(x/100.0))

def _settle_time(a_max, t_ms, a_ss, eps=0.01, dwell_ms=1.0):
    dt = t_ms[1]-t_ms[0]
    win = max(1, int(dwell_ms/dt))
    band = np.abs(a_max - a_ss) <= eps*a_ss
    for i in range(len(t_ms)-win):
        if band[i:i+win].all():
            return float(t_ms[i])
    return None

def _fwhm(a_profile):
    peak = float(a_profile.max())
    if peak <= 0: return None
    half = 0.5 * peak
    idx = np.where(a_profile >= half)[0]
    return None if len(idx)==0 else float(idx[-1]-idx[0]+1)

def _emit_sample_pack(root: Path, sample_id: str, sim: dict, stim_1d: np.ndarray, cfg_obj, seq_id: str, idx: int):
    """Write the per-sample pack to disk and return a rich metrics dict."""
    outdir = _ensure_dir(root / seq_id / f"sample_{idx:04d}")
    # ----- arrays
    t_ms = sim["t"] * sim["dt"] * 1000.0
    u = sim.get("u")
    a = sim.get("a")
    stim = stim_1d.astype("float32") if stim_1d.ndim == 1 else stim_1d

    meta = {
        "dt_ms": float(sim["dt"]*1000.0),
        "N": int(a.shape[1] if a is not None and a.size else stim_1d.shape[0]),
        "config_hash": config_hash(cfg_obj),
        "sample_id": sample_id,
        "seq_id": seq_id,
        "sample_idx": int(idx),
    }
    np.savez_compressed(outdir / f"{sample_id}_fields.npz",
                        t_ms=t_ms, u=u, a=a, stim=stim, meta=meta)

    # ----- curves
    a_max = sim["a_max"]; u_max = sim["u_max"]
    a_ss, _ = _compute_steady(a_max, t_ms)
    k10 = _kx_time(a_max, t_ms, a_ss, 10)
    k50 = _kx_time(a_max, t_ms, a_ss, 50)
    k90 = _kx_time(a_max, t_ms, a_ss, 90)
    t_settle = _settle_time(a_max, t_ms, a_ss)

    # steady window (last 20% or last 1ms)
    tail = max(int(len(t_ms)*0.2), int(1.0/(t_ms[1]-t_ms[0])))
    a_tail = a_max[-tail:]
    overshoot = 100.0*(np.max(a_max)-a_ss)/max(a_ss, 1e-8)
    ripple = 100.0*(np.max(a_tail)-np.min(a_tail))/max(a_ss, 1e-8)

    # steady-state spatial profile and shape stats
    a_ss_profile = a[-tail:,:].mean(axis=0) if a is not None and a.size else None
    peak_idx = int(a_ss_profile.argmax()) if a_ss_profile is not None else int(sim["peak_idx"][-1])
    width = _fwhm(a_ss_profile) if a_ss_profile is not None else None

    # ----- centroid & jitter (fixed)
    if a is not None and a.size:
        x = np.arange(a.shape[1], dtype=np.float32)
        # steady-state centroid
        denom_ss = float(a_ss_profile.sum() + 1e-12)
        cent = float((a_ss_profile * x).sum() / denom_ss)
        # centroid trajectory over last 'tail' steps
        W = a[-tail:,:]
        num = (W * x[None, :]).sum(axis=1)         # shape: (tail,)
        den = (W.sum(axis=1) + 1e-12)              # shape: (tail,)
        traj_cent = num / den                       # shape: (tail,)
        jitter = float(np.std(traj_cent))
    else:
        x = np.arange(stim_1d.shape[0], dtype=np.float32)
        cent = float(peak_idx)
        traj_cent = np.full(tail, cent, dtype=np.float32)
        jitter = 0.0

    # ----- figures
    fig = plt.figure(figsize=(6, 3))
    plt.plot(stim_1d); plt.title("Stimulus"); plt.xlabel("Position"); plt.ylabel("Intensity")
    plt.tight_layout(); fig.savefig(outdir / f"{sample_id}_stimulus.png", dpi=150); plt.close(fig)

    fig = plt.figure(figsize=(6, 3))
    plt.plot(t_ms, a_max, label="a_max"); plt.plot(t_ms, u_max, label="u_max")
    for val,lab in [(k10,"k10"),(k50,"k50"),(k90,"k90")]:
        if val is not None: plt.axvline(val, ls="--", alpha=0.6, label=lab)
    plt.axhline(a_ss, ls=":", alpha=0.6, label="steady")
    plt.legend(); plt.xlabel("ms"); plt.ylabel("peak"); plt.title("Activation Rise")
    plt.tight_layout(); fig.savefig(outdir / f"{sample_id}_rise_curve.png", dpi=150); plt.close(fig)

    if a is not None and a.size:
        fig = plt.figure(figsize=(9, 3))
        plt.imshow(a, aspect="auto", origin="lower",
                   extent=[0, a.shape[1], float(t_ms[0]), float(t_ms[-1])])
        plt.colorbar(label="a"); plt.title("Activation a(x,t)")
        plt.xlabel("Position"); plt.ylabel("Time (ms)")
        plt.tight_layout(); fig.savefig(outdir / f"{sample_id}_a_heatmap.png", dpi=150); plt.close(fig)

    if u is not None and u.size:
        fig = plt.figure(figsize=(9, 3))
        plt.imshow(u, aspect="auto", origin="lower",
                   extent=[0, u.shape[1], float(t_ms[0]), float(t_ms[-1])])
        plt.colorbar(label="u"); plt.title("DNF field u(x,t)")
        plt.xlabel("Position"); plt.ylabel("Time (ms)")
        plt.tight_layout(); fig.savefig(outdir / f"{sample_id}_u_heatmap.png", dpi=150); plt.close(fig)

    if a_ss_profile is not None:
        fig = plt.figure(figsize=(6, 3))
        plt.plot(a_ss_profile); plt.title("Steady bump (a)")
        plt.xlabel("Position"); plt.ylabel("a")
        plt.tight_layout(); fig.savefig(outdir / f"{sample_id}_bump_profile.png", dpi=150); plt.close(fig)

    # ----- metrics (rich)
    metrics = dict(
        sample_id=sample_id,
        N=int(meta["N"]),
        dt_ms=float(meta["dt_ms"]),
        T_ms=float(t_ms[-1]-t_ms[0]) if len(t_ms) > 1 else float(t_ms[-1]),
        peak_index=int(peak_idx),
        a_max_ss=float(a_ss),
        k10_ms=k10, k50_ms=k50, k90_ms=k90,
        t_settle_ms=t_settle, overshoot_pct=float(overshoot), ripple_pct=float(ripple),
        fwhm_px=(None if width is None else float(width)),
        centroid_px=float(cent), centroid_jitter_px=float(jitter),
        energy_L2_u=(float(np.linalg.norm(u[-tail:,:])) if u is not None and u.size else None),
        energy_pos_u=(float(np.maximum(u[-tail:,:],0).sum()) if u is not None and u.size else None),
        status="UNKNOWN", fail_reasons=[],
        config_hash=meta["config_hash"],
        seq_id=seq_id, sample_idx=int(idx),
    )
    (outdir / f"{sample_id}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    artifacts = [
        f"{sample_id}_fields.npz",
        f"{sample_id}_stimulus.png",
        f"{sample_id}_rise_curve.png",
    ]
    if a is not None and a.size: artifacts.append(f"{sample_id}_a_heatmap.png")
    if u is not None and u.size: artifacts.append(f"{sample_id}_u_heatmap.png")
    artifacts.append(f"{sample_id}_bump_profile.png")
    artifacts.append(f"{sample_id}_metrics.json")
    (outdir / "manifest.json").write_text(json.dumps({
        "sample_id": sample_id, "artifacts": artifacts
    }, indent=2), encoding="utf-8")

    return metrics, outdir



def _save_trace_plot(plots_dir: Path, seq_id: str, sample_idx: int, sim: dict, m: dict):
    sub = _ensure_dir(plots_dir / seq_id)
    fig = plt.figure(figsize=(6.0, 3.0))
    plt.plot(sim["t"], sim["u_max"])
    title = (f"{seq_id} | sample {sample_idx:04d} | "
             f"k90={m['k90_ms']:.1f}ms k99={m['k99_ms']:.1f}ms "
             f"ripple={m['ripple_pct']:.2f}% peak={m['peak']:.2f} "
             f"{'PASS' if m.get('pass', False) else 'FAIL'}")
    plt.title(title)
    plt.xlabel("step"); plt.ylabel("u_max (a.u.)")
    plt.tight_layout()
    fig.savefig(sub / f"sample_{sample_idx:04d}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

def _save_trace_csv(plots_dir: Path, seq_id: str, sample_idx: int, sim: dict, amp_used: float):
    sub = _ensure_dir(plots_dir / seq_id)
    path = sub / f"sample_{sample_idx:04d}_trace.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["t", "u_max", "a_max", "peak_idx", "amp_used"])
        w.writeheader()
        for t, u, a, pk in zip(sim["t"], sim["u_max"], sim["a_max"], sim["peak_idx"]):
            w.writerow({
                "t": int(t),
                "u_max": float(u),
                "a_max": float(a),
                "peak_idx": int(pk),
                "amp_used": float(amp_used),
            })


# --------------------------- main runner ---------------------------

def run_black_box(
    config_file: Path,
    cert_file: Path,
    datasets_cfg: Path,
    dataset_name: str,
    split: str,
    out_dir: Path,
    max_frames: int | None = None,
    ema_alpha: float | None = None,
    verbose: bool = True,
    per_sample: bool = False,
    sample_steps: int = 600,
    print_every: int = 10,
    save_traces: bool = False,
) -> int:
    # Load configs
    cfg_obj = load_config(str(config_file))
    cfg = cfg_obj.__dict__.copy()
    gates = load_cert(str(cert_file))
    ds_all = yaml.safe_load(datasets_cfg.read_text())
    ds = ds_all["datasets"][dataset_name]

    # Ensure adapter plugin is imported so @register runs
    adapter_name = ds["adapter"]
    last_err = None
    for mod in (f"python_env.adapters.{adapter_name}.loader",
                f"python_env.adapters.{adapter_name}"):
        try:
            importlib.import_module(mod)
            break
        except Exception as e:
            last_err = e
    adapter = reg.get(adapter_name, **ds)

    # Outputs
    run_dir = _ensure_dir(out_dir / f"{dataset_name}_{split}_{_timestamp()}")
    plots_dir = _ensure_dir(run_dir / "plots")
    copies_dir = _ensure_dir(run_dir / "copies")
    samples_dir = _ensure_dir(run_dir / "samples")
    (copies_dir / "config.yaml").write_text(Path(config_file).read_text(), encoding="utf-8")
    (copies_dir / "certification.yaml").write_text(Path(cert_file).read_text(), encoding="utf-8")
    (copies_dir / "datasets.yaml").write_text(Path(datasets_cfg).read_text(), encoding="utf-8")

    # Aggregation for sequence-mode summary
    all_pass = True
    n_seq = 0
    agg = {"k90_ms": [], "k99_ms": [], "ripple_pct": [], "peak": []}

    if not per_sample:
        # ---------------------- SEQUENCE MODE ----------------------
        reg_jsonl = (run_dir / "registry.jsonl").open("w", encoding="utf-8", newline="\n")
        csv_path = run_dir / f"{dataset_name}_{split}_metrics.csv"
        csv_f = csv_path.open("w", newline="", encoding="utf-8")
        csv_w = csv.DictWriter(csv_f, fieldnames=[
            "seq_id", "pass", "k90_ms", "k99_ms", "ripple_pct", "peak", "islands", "fwhm_bins", "steps"
        ])
        csv_w.writeheader()

        for seq in _iter_sequences(adapter, split):
            n_seq += 1

            def stream():
                y = None
                for x in _iter_Iext_for_seq(adapter, seq, cfg, max_frames):
                    if ema_alpha is None:
                        yield x
                    else:
                        y = x if y is None else (1 - ema_alpha) * y + ema_alpha * x
                        yield y

            sim = step_dnf_stream(stream(), cfg)  # lite
            m = certify_sequence(sim, cfg_obj, gates=gates)

            for k in agg:
                agg[k].append(m[k])
            all_pass &= bool(m["pass"])

            csv_w.writerow({
                "seq_id": seq.seq_id, "pass": m["pass"],
                "k90_ms": f"{m['k90_ms']:.3f}", "k99_ms": f"{m['k99_ms']:.3f}",
                "ripple_pct": f"{m['ripple_pct']:.3f}", "peak": f"{m['peak']:.3f}",
                "islands": m.get("islands"), "fwhm_bins": m.get("fwhm_bins"),
                "steps": m.get("steps")
            })
            reg_jsonl.write(json.dumps({
                "ts": _timestamp(),
                "dataset": dataset_name, "split": split, "sequence": seq.seq_id,
                "params": cfg, "gates": gates, "metrics": m
            }) + "\n")
            if verbose:
                print(f"[{seq.seq_id}] pass={m['pass']} "
                      f"k99={m['k99_ms']:.2f}ms ripple={m['ripple_pct']:.2f}% peak={m['peak']:.2f}")

        csv_f.close()
        reg_jsonl.close()

    else:
        # ---------------------- PER-SAMPLE MODE ----------------------
        samples_csv = run_dir / f"{dataset_name}_{split}_samples.csv"
        f_csv = samples_csv.open("w", newline="", encoding="utf-8")
        cols = ["seq_id", "sample_idx", "pass", "k90_ms", "k99_ms",
                "ripple_pct", "peak", "islands", "fwhm_bins", "steps",
                "reasons", "amp_used", "pack_dir"]
        w = csv.DictWriter(f_csv, fieldnames=cols)
        w.writeheader()

        total_samples = 0
        passed = 0

        for seq in _iter_sequences(adapter, split):
            n_seq += 1
            for idx, frame_I in enumerate(_iter_Iext_for_seq(adapter, seq, cfg, max_frames)):
                total_samples += 1
                I = frame_I.astype("float32")  # 1-D I_ext for this image
                amp_used = float(np.max(I))  # log actual input amplitude

                # FULL traces for sample-pack emission
                sim = step_dnf_stream(_repeat_I(I, sample_steps), cfg, collect_full=True)
                m = certify_sequence(sim, cfg_obj, gates=gates)
                reasons_str = m.get("reasons", "")
                if m.get("pass", False):
                    passed += 1

                # Emit the pack
                sample_id = f"{seq.seq_id}_sample_{idx:04d}"
                rich, pack_dir = _emit_sample_pack(samples_dir, sample_id, sim, I, cfg_obj, seq.seq_id, idx)

                # enrich metrics.json with gate results + reasons + pass/fail
                rich.update({
                    "pass": bool(m.get("pass", False)),
                    "reasons": reasons_str,
                    "gates_used": m.get("gates_used", {}),
                    "k90_ms": m.get("k90_ms"), "k99_ms": m.get("k99_ms"),
                    "ripple_pct": m.get("ripple_pct"), "peak": m.get("peak"),
                    "islands": m.get("islands"), "fwhm_bins": m.get("fwhm_bins"),
                    "steps": m.get("steps"), "amp_used": amp_used
                })

                # plots and optional trace (legacy)
                _save_trace_plot(plots_dir, seq.seq_id, idx, sim, m)
                if save_traces:
                    _save_trace_csv(plots_dir, seq.seq_id, idx, sim, amp_used)

                # CSV row
                w.writerow({
                    "seq_id": seq.seq_id,
                    "sample_idx": idx,
                    "pass": m.get("pass", False),
                    "k90_ms": f"{m['k90_ms']:.3f}",
                    "k99_ms": f"{m['k99_ms']:.3f}",
                    "ripple_pct": f"{m['ripple_pct']:.3f}",
                    "peak": f"{m['peak']:.3f}",
                    "islands": m.get("islands"),
                    "fwhm_bins": m.get("fwhm_bins"),
                    "steps": m.get("steps"),
                    "reasons": reasons_str,
                    "amp_used": f"{amp_used:.6f}",
                    "pack_dir": str(pack_dir),
                })


                if verbose and (idx % max(1, print_every) == 0):
                    print(f"[{seq.seq_id} sample {idx:04d}] "
                          f"{'PASS' if m.get('pass', False) else 'FAIL'} | "
                          f"k99={m['k99_ms']:.2f}ms ripple={m['ripple_pct']:.2f}% "
                          f"peak={m['peak']:.2f} amp={amp_used:.4f} | {m.get('reasons','')}")

        f_csv.close()
        all_pass = (passed == total_samples)
        agg = {}  # not meaningful in per-sample mode
        print(f"\nPer-sample summary: {passed}/{total_samples} samples passed.")

    # Summary JSON (both modes)
    def _mean(xs): return float(np.mean(xs)) if xs else None
    summary = {
        "dataset": dataset_name,
        "split": split,
        "mode": "per-sample" if per_sample else "sequence",
        "n_sequences": n_seq,
        "overall_pass": bool(all_pass),
        "means": {k: _mean(v) for k, v in agg.items()} if agg else {},
        "config_file": str(config_file),
        "certification_file": str(cert_file),
        "datasets_file": str(datasets_cfg),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n== SUMMARY ==")
    print(json.dumps(summary, indent=2))

    return 0 if all_pass else 2


# ----------------------------- CLI entry -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cert", required=True)
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="artifacts/runs")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--ema-alpha", type=float, default=None,
                    help="EMA smoothing for sequence mode only.")
    ap.add_argument("--per-sample", action="store_true",
                    help="Evaluate each image independently (reset DNF per image).")
    ap.add_argument("--sample-steps", type=int, default=600,
                    help="DNF steps per image in per-sample mode.")
    ap.add_argument("--print-every", type=int, default=10,
                    help="Log every N samples in per-sample mode (use 1 to print all).")
    ap.add_argument("--save-traces", action="store_true",
                    help="Save per-sample trace CSV (t,u_max,a_max,peak_idx,amp_used).")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    code = run_black_box(
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
    sys.exit(code)


if __name__ == "__main__":
    main()


# # python_env/bb_run.py
# import argparse, json, csv, sys
# from pathlib import Path
# from datetime import datetime
# import importlib

# import yaml
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")  # headless-safe for savefig
# import matplotlib.pyplot as plt

# from python_env.bb_config import load_config, load_cert
# from python_env.bb_core import step_dnf_stream
# from python_env.bb_metrics import certify_sequence
# import python_env.adapters.registry as reg


# def _ensure_dir(p: Path):
#     p.mkdir(parents=True, exist_ok=True)
#     return p

# def _timestamp():
#     return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")

# def _iter_sequences(adapter, split: str):
#     for seq in adapter.iter_sequences(split):
#         yield seq

# def _iter_Iext_for_seq(adapter, seq, cfg: dict, max_frames: int | None):
#     n = 0
#     for frame in adapter.iter_frames(seq):
#         yield adapter.frame_to_Iext(frame, cfg)
#         n += 1
#         if max_frames is not None and n >= max_frames:
#             break

# def _repeat_I(I, steps: int):
#     for _ in range(int(steps)):
#         yield I

# def _reasons_from_metrics(m: dict, gates: dict) -> list[str]:
#     """Build human-readable failure reasons robustly."""
#     reasons = []
#     # Prefer explicit booleans if present:
#     if "peak_pass" in m and not m["peak_pass"]:
#         reasons.append("peak < min_peak")
#     if "ripple_pass" in m and not m["ripple_pass"]:
#         reasons.append("ripple > max_ripple")
#     if "k90_pass" in m and not m["k90_pass"]:
#         reasons.append("k90 > gate")
#     if "k99_pass" in m and not m["k99_pass"]:
#         reasons.append("k99 > gate")
#     if "islands_pass" in m and not m["islands_pass"]:
#         reasons.append("islands > gate")
#     if "fwhm_pass" in m and not m["fwhm_pass"]:
#         reasons.append("FWHM > gate")

#     # If no booleans, infer conservatively from values (support fraction/percent gate)
#     if not reasons:
#         if "peak" in m and "min_peak" in gates and m["peak"] < gates["min_peak"]:
#             reasons.append(f"peak={m['peak']:.3f} < {gates['min_peak']:.3f}")
#         if "ripple_pct" in m and "max_ripple" in gates:
#             gate_pct = gates["max_ripple"]*100.0 if gates["max_ripple"] <= 1.0 else gates["max_ripple"]
#             if m["ripple_pct"] > gate_pct:
#                 reasons.append(f"ripple={m['ripple_pct']:.2f}% > {gate_pct:.2f}%")
#         if "k90_ms" in m and "k90_ms" in gates and m["k90_ms"] > gates["k90_ms"]:
#             reasons.append(f"k90={m['k90_ms']:.1f}ms > {gates['k90_ms']:.1f}ms")
#         if "k99_ms" in m and "k99_ms" in gates and m["k99_ms"] > gates["k99_ms"]:
#             reasons.append(f"k99={m['k99_ms']:.1f}ms > {gates['k99_ms']:.1f}ms")
#     return reasons

# def _save_trace_plot(plots_dir: Path, seq_id: str, sample_idx: int, sim: dict, m: dict):
#     sub = _ensure_dir(plots_dir / seq_id)
#     fig = plt.figure(figsize=(6.0, 3.0))
#     plt.plot(sim["t"], sim["u_max"])
#     title = (f"{seq_id} | sample {sample_idx:04d} | "
#              f"k90={m['k90_ms']:.1f}ms k99={m['k99_ms']:.1f}ms "
#              f"ripple={m['ripple_pct']:.2f}% peak={m['peak']:.2f} "
#              f"{'PASS' if m.get('pass', False) else 'FAIL'}")
#     plt.title(title)
#     plt.xlabel("step"); plt.ylabel("u_max (a.u.)")
#     plt.tight_layout()
#     fig.savefig(sub / f"sample_{sample_idx:04d}.pdf", dpi=150, bbox_inches="tight")
#     plt.close(fig)

# def run_black_box(
#     config_file: Path,
#     cert_file: Path,
#     datasets_cfg: Path,
#     dataset_name: str,
#     split: str,
#     out_dir: Path,
#     max_frames: int | None = None,
#     ema_alpha: float | None = None,
#     verbose: bool = True,
#     per_sample: bool = False,
#     sample_steps: int = 600,
# ) -> int:
#     # ---- load configs
#     cfg_obj = load_config(str(config_file))
#     cfg = cfg_obj.__dict__.copy()
#     gates = load_cert(str(cert_file))
#     ds_all = yaml.safe_load(datasets_cfg.read_text())
#     ds = ds_all["datasets"][dataset_name]

#     # ---- ensure adapter plugin is imported so @register runs
#     adapter_name = ds["adapter"]
#     last_err = None
#     for mod in (f"python_env.adapters.{adapter_name}.loader",
#                 f"python_env.adapters.{adapter_name}"):
#         try:
#             importlib.import_module(mod)
#             break
#         except Exception as e:
#             last_err = e
#     # ---- build adapter (registry should now have it)
#     adapter = reg.get(adapter_name, **ds)

#     # ---- outputs
#     run_dir = _ensure_dir(out_dir / f"{dataset_name}_{split}_{_timestamp()}")
#     plots_dir = _ensure_dir(run_dir / "plots")
#     copies_dir = _ensure_dir(run_dir / "copies")
#     (copies_dir / "config.yaml").write_text(Path(config_file).read_text(), encoding="utf-8")
#     (copies_dir / "certification.yaml").write_text(Path(cert_file).read_text(), encoding="utf-8")
#     (copies_dir / "datasets.yaml").write_text(Path(datasets_cfg).read_text(), encoding="utf-8")

#     # common aggregations for overall summary (sequence-mode only)
#     all_pass = True
#     n_seq = 0
#     agg = {"k90_ms":[], "k99_ms":[], "ripple_pct":[], "peak":[]}

#     if not per_sample:
#         # ---------------------- SEQUENCE MODE (default) ----------------------
#         reg_jsonl = (run_dir / "registry.jsonl").open("w", encoding="utf-8", newline="\n")
#         csv_path = run_dir / f"{dataset_name}_{split}_metrics.csv"
#         csv_f = csv_path.open("w", newline="", encoding="utf-8")
#         csv_w = csv.DictWriter(csv_f, fieldnames=[
#             "seq_id","pass","k90_ms","k99_ms","ripple_pct","peak","islands","fwhm_bins","steps"
#         ])
#         csv_w.writeheader()

#         for seq in _iter_sequences(adapter, split):
#             n_seq += 1

#             def stream():
#                 y = None
#                 for x in _iter_Iext_for_seq(adapter, seq, cfg, max_frames):
#                     if ema_alpha is None:
#                         yield x
#                     else:
#                         y = x if y is None else (1-ema_alpha)*y + ema_alpha*x
#                         yield y

#             sim = step_dnf_stream(stream(), cfg)
#             m = certify_sequence(sim, cfg_obj, gates=gates)

#             for k in agg: agg[k].append(m[k])
#             all_pass &= bool(m["pass"])

#             csv_w.writerow({
#                 "seq_id": seq.seq_id, "pass": m["pass"],
#                 "k90_ms": f"{m['k90_ms']:.3f}", "k99_ms": f"{m['k99_ms']:.3f}",
#                 "ripple_pct": f"{m['ripple_pct']:.3f}", "peak": f"{m['peak']:.3f}",
#                 "islands": m.get("islands"), "fwhm_bins": m.get("fwhm_bins"),
#                 "steps": m.get("steps")
#             })
#             reg_jsonl.write(json.dumps({
#                 "ts": _timestamp(),
#                 "dataset": dataset_name, "split": split, "sequence": seq.seq_id,
#                 "params": cfg, "gates": gates, "metrics": m
#             }) + "\n")
#             if verbose:
#                 print(f"[{seq.seq_id}] pass={m['pass']} "
#                       f"k99={m['k99_ms']:.2f}ms ripple={m['ripple_pct']:.2f}% peak={m['peak']:.2f}")

#         csv_f.close(); reg_jsonl.close()

#     else:
#         # ---------------------- PER-SAMPLE MODE ------------------------------
#         # One run per image; reset the DNF for every image.
#         samples_csv = run_dir / f"{dataset_name}_{split}_samples.csv"
#         f_csv = samples_csv.open("w", newline="", encoding="utf-8")
#         cols = ["seq_id", "sample_idx", "pass", "k90_ms", "k99_ms",
#                 "ripple_pct", "peak", "islands", "fwhm_bins", "steps", "reasons"]
#         w = csv.DictWriter(f_csv, fieldnames=cols)
#         w.writeheader()

#         total_samples = 0
#         passed = 0

#         for seq in _iter_sequences(adapter, split):
#             n_seq += 1
#             # enumerate images as independent samples
#             idx = -1
#             for idx, frame in enumerate(_iter_Iext_for_seq(adapter, seq, cfg, max_frames)):
#                 total_samples += 1
#                 I = frame  # single image → 1-D I_ext
#                 # Optional temporal smoothing doesn't apply here (single sample)
#                 sim = step_dnf_stream(_repeat_I(I, sample_steps), cfg)
#                 m = certify_sequence(sim, cfg_obj, gates=gates)
#                 reasons = [] if m.get("pass", False) else _reasons_from_metrics(m, gates)
#                 if m.get("pass", False):
#                     passed += 1
                    
#                 # write row
#                 w.writerow({
#                     "seq_id": seq.seq_id,
#                     "sample_idx": idx,
#                     "pass": m.get("pass", False),
#                     "k90_ms": f"{m['k90_ms']:.3f}",
#                     "k99_ms": f"{m['k99_ms']:.3f}",
#                     "ripple_pct": f"{m['ripple_pct']:.3f}",
#                     "peak": f"{m['peak']:.3f}",
#                     "islands": m.get("islands"),
#                     "fwhm_bins": m.get("fwhm_bins"),
#                     "steps": m.get("steps"),
#                     "reasons": "; ".join(reasons)
#                 })

#                 # save per-sample plot
#                 _save_trace_plot(plots_dir, seq.seq_id, idx, sim, m)

#                 if verbose and (idx % 10 == 0):
#                     print(f"[{seq.seq_id} sample {idx:04d}] "
#                           f"{'PASS' if m.get('pass', False) else 'FAIL'} | "
#                           f"k99={m['k99_ms']:.2f}ms ripple={m['ripple_pct']:.2f}% peak={m['peak']:.2f}")

#         f_csv.close()
#         # summarize per-sample mode
#         all_pass = (passed == total_samples)
#         agg = {}  # not meaningful in per-sample; summary.json still records overall_pass
#         print(f"\nPer-sample summary: {passed}/{total_samples} samples passed.")

#     # ---- Summary JSON (covers both modes)
#     def _mean(xs): return float(np.mean(xs)) if xs else None
#     summary = {
#         "dataset": dataset_name,
#         "split": split,
#         "mode": "per-sample" if per_sample else "sequence",
#         "n_sequences": n_seq,
#         "overall_pass": bool(all_pass),
#         "means": {k: _mean(v) for k, v in agg.items()} if agg else {},
#         "config_file": str(config_file),
#         "certification_file": str(cert_file),
#         "datasets_file": str(datasets_cfg),
#     }
#     (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
#     print("\n== SUMMARY =="); print(json.dumps(summary, indent=2))

#     return 0 if all_pass else 2


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", required=True)
#     ap.add_argument("--cert", required=True)
#     ap.add_argument("--datasets", required=True)
#     ap.add_argument("--dataset", required=True)
#     ap.add_argument("--split", default="test")
#     ap.add_argument("--out", default="artifacts/runs")
#     ap.add_argument("--max-frames", type=int, default=None)
#     ap.add_argument("--ema-alpha", type=float, default=None)
#     ap.add_argument("--per-sample", action="store_true",
#                     help="Evaluate each image independently (reset DNF per image).")
#     ap.add_argument("--sample-steps", type=int, default=600,
#                     help="Number of DNF steps per image in per-sample mode.")
#     ap.add_argument("--quiet", action="store_true")
#     args = ap.parse_args()

#     code = run_black_box(
#         config_file=Path(args.config),
#         cert_file=Path(args.cert),
#         datasets_cfg=Path(args.datasets),
#         dataset_name=args.dataset,
#         split=args.split,
#         out_dir=Path(args.out),
#         max_frames=args.max_frames,
#         ema_alpha=args.ema_alpha,
#         verbose=not args.quiet,
#         per_sample=args.per_sample,
#         sample_steps=args.sample_steps,
#     )
#     sys.exit(code)

# if __name__ == "__main__":
#     main()
