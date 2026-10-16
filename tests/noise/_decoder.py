"""A lookup-table decoder for the sanity tests.

Trained on samples of one circuit: for every observed detector pattern
the majority observable value. Decoding another circuit with the same
table gives a like-for-like comparison of two channels that share the
detector structure. It is not the board's pinned decoder (bposd-cs-10,
which needs ldpc); the tests here compare rates between profiles, not
against board values. Fine for the [[9,1,3]] one-round circuits, whose
DEMs have 8 detectors.
"""

from __future__ import annotations

import numpy as np


def _keys(dets):
    dets = np.asarray(dets, dtype=np.uint8)
    weights = (1 << np.arange(dets.shape[1], dtype=np.int64))
    return dets.astype(np.int64) @ weights


class LookupDecoder:
    def __init__(self, table: dict[int, int]):
        self.table = table

    @classmethod
    def train(cls, circuit, shots: int, seed: int) -> LookupDecoder:
        dets, obs = circuit.compile_detector_sampler(seed=seed).sample(
            shots, separate_observables=True)
        keys = _keys(dets)
        ones = {}
        counts = {}
        for k, o in zip(keys.tolist(), obs[:, 0].tolist()):
            counts[k] = counts.get(k, 0) + 1
            ones[k] = ones.get(k, 0) + int(o)
        table = {k: int(2 * ones[k] > counts[k]) for k in counts}
        return cls(table)

    def failures(self, circuit, shots: int, seed: int) -> int:
        dets, obs = circuit.compile_detector_sampler(seed=seed).sample(
            shots, separate_observables=True)
        keys = _keys(dets)
        pred = np.fromiter((self.table.get(k, 0) for k in keys.tolist()),
                           dtype=np.uint8, count=len(keys))
        return int(np.sum(pred != obs[:, 0].astype(np.uint8)))


def two_proportion_z(f1: int, n1: int, f2: int, n2: int) -> float:
    """z statistic of the difference of two failure fractions."""
    p1, p2 = f1 / n1, f2 / n2
    p = (f1 + f2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return 0.0 if se == 0 else float((p1 - p2) / se)
