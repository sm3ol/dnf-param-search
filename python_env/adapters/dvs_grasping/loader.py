# python/adapters/dvs_grasping/loader.py
import pathlib, numpy as np, glob
from ..registry import register
from ..base import Adapter, Sequence
from ...core.projectors import projector_by_name

def bin_events_to_frame(ev, H, W, bin_ms, t0):
    """
    ev: Nx4 array [t(us), x, y, p], returns H×W float image integrated over [t0, t0+bin_ms].
    """
    t1 = t0 + bin_ms*1000.0
    mask = (ev[:,0] >= t0) & (ev[:,0] < t1)
    sub = ev[mask]
    img = np.zeros((H, W), dtype="float32")
    # polarity: +1 for on, -1 for off; clip to avoid blow-up
    for _, x, y, p in sub:
        img[int(y), int(x)] += (1.0 if p > 0 else -1.0)
    # normalize per bin if desired
    return img

@register("dvs_grasping")
class DVSGraspingAdapter(Adapter):
    def __init__(self, root: str, mapping: dict, split_pattern: dict, davis: dict):
        self.root = pathlib.Path(root)
        self.N = mapping["N"]
        self.project = projector_by_name(mapping.get("projector","angle_ring"))
        self.pattern = {k: self.root / v for k, v in split_pattern.items()}
        self.H, self.W = davis["resolution"][1], davis["resolution"][0]
        self.bin_ms = float(davis.get("bin_ms", 1.0))

    def iter_sequences(self, split: str):
        for path in sorted(glob.glob(str(self.pattern[split]))):
            yield Sequence(seq_id=pathlib.Path(path).name, meta={"path": path})

    def iter_frames(self, seq: Sequence):
        # assume events in seq/path/events.npy → [t_us, x, y, p]
        ev = np.load(pathlib.Path(seq.meta["path"]) / "events.npy")  # Nx4
        t0, tmax = ev[0,0], ev[-1,0]
        t = t0
        while t < tmax:
            yield bin_events_to_frame(ev, self.H, self.W, self.bin_ms, t)
            t += self.bin_ms*1000.0

    # in python/adapters/ycbsight/loader.py (and similarly dvs_grasping)
    def frame_to_Iext(self, frame, cfg):
        idx_1d = self.project(frame, N=self.N)
        w = int(cfg.get("gaussian_stimulus_width", 5))
        xs = np.arange(self.N, dtype="float32")
        gauss = np.exp(-0.5 * ((xs - idx_1d)/max(1,w))**2).astype("float32")
        gain = float(cfg.get("intensity_scale_factor", cfg.get("stim_gain", 1.0)))
        return gain * gauss


    @staticmethod
    def _gaussian_1d(c, N, w):
        xs = np.arange(N, dtype="float32")
        return np.exp(-0.5 * ((xs - c)/max(1, w))**2).astype("float32")
