# core/projectors.py
import numpy as np

def _peak(frame):  # argmax index
    y, x = np.unravel_index(np.argmax(frame), frame.shape)
    return y, x

def raster_ring(frame, N):
    H, W = frame.shape
    y, x = _peak(frame)
    flat = y*W + x
    return int((flat / (H*W - 1)) * (N - 1))

def angle_ring(frame, N):
    H, W = frame.shape
    y, x = _peak(frame)
    cx, cy = (W-1)/2.0, (H-1)/2.0
    ang = np.arctan2(y-cy, x-cx)  # [-pi, pi]
    idx = (ang + np.pi) / (2*np.pi) * (N-1)
    return int(np.clip(idx, 0, N-1))

_PROJECTORS = {"raster_ring": raster_ring, "angle_ring": angle_ring}
def projector_by_name(name): return _PROJECTORS[name]
