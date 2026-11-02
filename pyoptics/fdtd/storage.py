#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Saving and loading monitor data.

Results are dictionaries of NumPy arrays plus scalar attributes. They are
written to compressed .npz files (no extra dependency) or, for files ending
in .h5/.hdf5, to HDF5 with gzip compression (needs the optional h5py,
``pip install pyoptics[hdf5]``). HDF5 also allows streaming snapshot frames
to disk during a run (Snapshots(path=...)).
"""

import numpy as np


def _is_hdf5(path):
    return str(path).lower().endswith((".h5", ".hdf5"))


def _h5py():
    try:
        import h5py
    except ImportError:   # pragma: no cover - depends on the installation
        raise ImportError("HDF5 files need h5py: pip install pyoptics[hdf5]")
    return h5py


def save(path, arrays, attrs=None):
    """Write a dict of arrays (and scalar/string attributes) to .npz or HDF5."""

    attrs = dict(attrs or {})
    if _is_hdf5(path):
        h5py = _h5py()
        with h5py.File(path, "w") as f:
            for k, v in arrays.items():
                v = np.asarray(v)
                if v.ndim:
                    f.create_dataset(k, data=v, compression="gzip", shuffle=True,
                                     chunks=True if v.size > 1 else None)
                else:
                    f.create_dataset(k, data=v)
            for k, v in attrs.items():
                f.attrs[k] = v
    else:
        data = {k: np.asarray(v) for k, v in arrays.items()}
        for k, v in attrs.items():
            data["attr__" + k] = np.asarray(v)
        np.savez_compressed(path, **data)


def load(path):
    """Read a file written by save(); returns (arrays, attrs)."""

    arrays, attrs = {}, {}
    if _is_hdf5(path):
        h5py = _h5py()
        with h5py.File(path, "r") as f:
            for k in f:
                arrays[k] = f[k][()]
            for k, v in f.attrs.items():
                attrs[k] = v.decode() if isinstance(v, bytes) else v
    else:
        with np.load(path, allow_pickle=False) as f:
            for k in f.files:
                if k.startswith("attr__"):
                    v = f[k]
                    attrs[k[6:]] = v.item() if v.ndim == 0 else v
                else:
                    arrays[k] = f[k]
    for k, v in attrs.items():
        if isinstance(v, np.str_):
            attrs[k] = str(v)
    return arrays, attrs


class FrameWriter(object):
    """Appends frames to a growing, chunked and gzip compressed HDF5
    dataset (used by Snapshots(path=...))."""

    def __init__(self, path, frame_shape, dtype, coords, attrs):
        h5py = _h5py()
        self.file = h5py.File(path, "w")
        self.frames = self.file.create_dataset("frames", shape=(0,) + tuple(frame_shape),
                                               maxshape=(None,) + tuple(frame_shape),
                                               dtype=dtype, chunks=(1,) + tuple(frame_shape),
                                               compression="gzip", shuffle=True)
        self.times = self.file.create_dataset("times", shape=(0,), maxshape=(None,), dtype=float)
        for i, c in enumerate(coords):
            self.file.create_dataset("xyz"[i], data=c)
        for k, v in attrs.items():
            self.file.attrs[k] = v

    def append(self, t, frame):
        n = self.frames.shape[0]
        self.frames.resize(n + 1, axis=0)
        self.frames[n] = frame
        self.times.resize(n + 1, axis=0)
        self.times[n] = t
        self.file.flush()

    def close(self):
        self.file.close()
