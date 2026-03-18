# -*- coding: utf-8 -*-
"""
src.data.windowing — Fixed-length centered window builder for seq_bigru_residual.

Converts per-experiment or already-split arrays into windowed sequences
suitable for sequence-aware cVAE architectures.

Design constraints
------------------
- ``window_size`` must be odd (centered context, half = window_size // 2).
- Windows are built independently per (experiment, split-side) chunk.
- Windows never cross experiment boundaries.
- Windows never cross train/val boundaries.
- Windowing must happen AFTER the temporal per-experiment split.
- ``stride=1`` guarantees one output per original input sample.
- For sequence variants, only contiguous cap/reduction is compatible;
  ``balanced_blocks`` must not be used before windowing (broken temporal context).

Public API
----------
build_windows_single_experiment     Window a single contiguous slice.
build_windows_from_split_arrays     Window already-split arrays per experiment.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def build_windows_single_experiment(
    X: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    window_size: int,
    stride: int = 1,
    pad_mode: str = "edge",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build centered context windows for a single contiguous experiment slice.

    Parameters
    ----------
    X : (N, 2)          Sent I/Q signal — source of context windows.
    Y : (N, 2)          Received I/Q signal — prediction target.
    D : (N,) or (N, 1)  Distance condition (scalar per sample).
    C : (N,) or (N, 1)  Current condition (scalar per sample).
    window_size : int   Number of context samples per window.  Must be odd.
    stride : int        Step between consecutive output samples (default 1).
    pad_mode : str      NumPy pad mode for boundary handling (default "edge").

    Returns
    -------
    X_seq    : (N_out, window_size, 2)  Windowed context drawn from X.
    Y_center : (N_out, 2)              Center-sample target from Y.
    D_center : (N_out, 1)              Distance for each center sample.
    C_center : (N_out, 1)              Current for each center sample.

    Where ``N_out = len(range(0, N, stride))``.
    With ``stride=1``, ``N_out == N`` (one output per original sample).

    Notes
    -----
    The center position within each window is ``window_size // 2``.
    Edge padding fills positions outside [0, N) with the boundary value,
    so every sample — including those at the start and end — has a
    full-width window.  No context from outside the slice is used.
    """
    if window_size % 2 == 0:
        raise ValueError(
            f"window_size must be odd, got {window_size}."
        )
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}.")

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    D = np.asarray(D, dtype=np.float32).reshape(-1, 1)
    C = np.asarray(C, dtype=np.float32).reshape(-1, 1)

    N = X.shape[0]
    if N == 0:
        return (
            np.empty((0, window_size, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
        )

    half = window_size // 2

    # Pad X along the time axis only; I/Q axis is unchanged.
    # X_padded shape: (N + 2*half, 2)
    X_padded = np.pad(X, pad_width=[(half, half), (0, 0)], mode=pad_mode)

    # Sliding windows along axis=0 (time).
    # sliding_window_view result shape: (N, 2, window_size)
    # Transpose to (N, window_size, 2) and copy to make contiguous.
    X_win = np.lib.stride_tricks.sliding_window_view(X_padded, window_size, axis=0)
    X_win = X_win.transpose(0, 2, 1).copy()  # (N, window_size, 2)

    # Apply stride to all outputs.
    X_seq = X_win[::stride]    # (N_out, window_size, 2)
    Y_center = Y[::stride]     # (N_out, 2)
    D_center = D[::stride]     # (N_out, 1)
    C_center = C[::stride]     # (N_out, 1)

    return X_seq, Y_center, D_center, C_center


def build_windows_from_split_arrays(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    D_train: np.ndarray,
    C_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    D_val: np.ndarray,
    C_val: np.ndarray,
    df_split: pd.DataFrame,
    window_size: int,
    stride: int = 1,
    pad_mode: str = "edge",
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Apply per-experiment windowing to already-split concatenated arrays.

    Reconstructs per-experiment slices from the concatenated split arrays
    using ``df_split`` (must contain ``n_train`` and ``n_val`` columns, as
    produced by ``apply_split`` with ``strategy="per_experiment"``).

    Windowing is applied independently to each (experiment × split-side) chunk,
    guaranteeing no cross-experiment and no cross-train/val boundary leakage.

    Parameters
    ----------
    X_train, Y_train, D_train, C_train : Concatenated training-side arrays.
    X_val, Y_val, D_val, C_val         : Concatenated validation-side arrays.
    df_split : pd.DataFrame             Must have columns ``n_train`` and ``n_val``
                                        in experiment order.
    window_size : int                   Odd window length.
    stride : int                        Window step (default 1).
    pad_mode : str                      Boundary pad mode (default "edge").

    Returns
    -------
    (X_seq_tr, Y_tr, D_tr, C_tr, X_seq_va, Y_va, D_va, C_va)

    X_seq_tr : (N_tr_out, window_size, 2)
    X_seq_va : (N_va_out, window_size, 2)
    Y/D/C arrays have matching leading dimensions.

    With ``stride=1``, ``N_tr_out == len(X_train)`` and
    ``N_va_out == len(X_val)``.
    """
    if "n_train" not in df_split.columns or "n_val" not in df_split.columns:
        raise ValueError(
            "df_split must have 'n_train' and 'n_val' columns. "
            "Use apply_split with strategy='per_experiment'."
        )

    n_train_list = [int(v) for v in df_split["n_train"].tolist()]
    n_val_list = [int(v) for v in df_split["n_val"].tolist()]

    # Flatten all condition arrays for consistent indexing.
    D_train = np.asarray(D_train, dtype=np.float32).reshape(-1, 1)
    C_train = np.asarray(C_train, dtype=np.float32).reshape(-1, 1)
    D_val = np.asarray(D_val, dtype=np.float32).reshape(-1, 1)
    C_val = np.asarray(C_val, dtype=np.float32).reshape(-1, 1)

    tr_X, tr_Y, tr_D, tr_C = [], [], [], []
    va_X, va_Y, va_D, va_C = [], [], [], []

    tr_cursor = 0
    va_cursor = 0

    for n_tr, n_va in zip(n_train_list, n_val_list):
        if n_tr > 0:
            X_w, Y_w, D_w, C_w = build_windows_single_experiment(
                X_train[tr_cursor: tr_cursor + n_tr],
                Y_train[tr_cursor: tr_cursor + n_tr],
                D_train[tr_cursor: tr_cursor + n_tr],
                C_train[tr_cursor: tr_cursor + n_tr],
                window_size=window_size,
                stride=stride,
                pad_mode=pad_mode,
            )
            tr_X.append(X_w)
            tr_Y.append(Y_w)
            tr_D.append(D_w)
            tr_C.append(C_w)
        tr_cursor += n_tr

        if n_va > 0:
            X_w, Y_w, D_w, C_w = build_windows_single_experiment(
                X_val[va_cursor: va_cursor + n_va],
                Y_val[va_cursor: va_cursor + n_va],
                D_val[va_cursor: va_cursor + n_va],
                C_val[va_cursor: va_cursor + n_va],
                window_size=window_size,
                stride=stride,
                pad_mode=pad_mode,
            )
            va_X.append(X_w)
            va_Y.append(Y_w)
            va_D.append(D_w)
            va_C.append(C_w)
        va_cursor += n_va

    def _cat(lst, shape_tail):
        if lst:
            return np.concatenate(lst, axis=0)
        return np.empty((0,) + shape_tail, dtype=np.float32)

    return (
        _cat(tr_X, (window_size, 2)),
        _cat(tr_Y, (2,)),
        _cat(tr_D, (1,)),
        _cat(tr_C, (1,)),
        _cat(va_X, (window_size, 2)),
        _cat(va_Y, (2,)),
        _cat(va_D, (1,)),
        _cat(va_C, (1,)),
    )
