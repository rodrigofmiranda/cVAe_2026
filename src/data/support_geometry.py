# -*- coding: utf-8 -*-
"""Support-geometry helpers for support-aware VLC experiments.

These utilities define a single source of truth for:

- the training-side geometry scale ``a_train``
- derived support features used by support-aware conditioning
- edge/corner sample weights
- virtual disk filtering inside the original square support
- support-region labelling for diagnostics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


_EPS = 1e-12


@dataclass(frozen=True)
class SupportGeometryStats:
    """Training-derived geometry scale for one split."""

    a_train: float

    def to_dict(self) -> Dict[str, float]:
        return {"a_train": float(self.a_train)}


def compute_support_geometry_stats(X_train: np.ndarray) -> SupportGeometryStats:
    """Return the scalar support normalization inferred from the train split."""
    X = np.asarray(X_train, dtype=np.float32)
    if X.ndim < 2 or X.shape[-1] != 2:
        raise ValueError(f"Expected X_train shape (N, 2), got {X.shape}")
    if X.size == 0:
        raise ValueError("Cannot compute support geometry on an empty train split.")
    a_train = float(np.max(np.abs(X)))
    if not np.isfinite(a_train) or a_train <= 0.0:
        raise ValueError(f"Invalid support scale inferred from train split: {a_train!r}")
    return SupportGeometryStats(a_train=a_train)


def _center_iq(X: np.ndarray) -> np.ndarray:
    """Return the point-wise IQ array used for geometry calculations.

    Accepts:
    - point-wise arrays ``(N, 2)``
    - sequence arrays ``(N, W, 2)`` and extracts the center frame
    """
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 2:
        return arr[:, arr.shape[1] // 2, :]
    raise ValueError(f"Unsupported IQ shape for support geometry: {arr.shape}")


def support_feature_dict(
    X: np.ndarray,
    *,
    a_train: float,
) -> Dict[str, np.ndarray]:
    """Return derived support features for point-wise or sequence inputs."""
    Xc = _center_iq(X).astype(np.float32, copy=False)
    scale = max(float(a_train), _EPS)
    abs_i = np.abs(Xc[:, 0])
    abs_q = np.abs(Xc[:, 1])
    r_l2 = np.sqrt(np.sum(np.square(Xc, dtype=np.float32), axis=1, dtype=np.float32))
    r_inf = np.maximum(abs_i, abs_q)
    return {
        "r_l2_norm": (r_l2 / scale).astype(np.float32, copy=False),
        "r_inf_norm": (r_inf / scale).astype(np.float32, copy=False),
        "cornerness_norm": ((abs_i * abs_q) / (scale * scale)).astype(np.float32, copy=False),
    }


def support_feature_matrix(
    X: np.ndarray,
    *,
    a_train: float,
) -> np.ndarray:
    """Return ``(N, 3)`` = ``[r_l2_norm, r_inf_norm, cornerness_norm]``."""
    feats = support_feature_dict(X, a_train=a_train)
    return np.stack(
        [feats["r_l2_norm"], feats["r_inf_norm"], feats["cornerness_norm"]],
        axis=1,
    ).astype(np.float32, copy=False)


def support_sample_weights(
    X: np.ndarray,
    *,
    a_train: float,
    mode: str = "none",
    alpha: float = 1.5,
    tau: float = 0.75,
    tau_corner: float = 0.35,
    weight_max: float = 3.0,
) -> np.ndarray:
    """Return support-aware training weights for point-wise or sequence inputs."""
    mode_norm = str(mode or "none").strip().lower()
    n = int(_center_iq(X).shape[0])
    if mode_norm in {"", "none"}:
        return np.ones((n,), dtype=np.float32)

    feats = support_feature_dict(X, a_train=a_train)
    r_inf = feats["r_inf_norm"]
    edge_term = np.clip((r_inf - float(tau)) / max(1.0 - float(tau), _EPS), 0.0, 1.0)
    w_edge = 1.0 + float(alpha) * edge_term

    if mode_norm == "edge_rinf":
        weights = w_edge
    elif mode_norm == "edge_rinf_corner":
        cornerness = feats["cornerness_norm"]
        corner_term = np.clip(
            (cornerness - float(tau_corner)) / max(1.0 - float(tau_corner), _EPS),
            0.0,
            1.0,
        )
        weights = w_edge * (1.0 + 0.5 * float(alpha) * corner_term)
    else:
        raise ValueError(
            f"Unknown support_weight_mode={mode!r}. "
            "Expected one of ['none', 'edge_rinf', 'edge_rinf_corner']."
        )

    return np.clip(weights, 1.0, float(weight_max)).astype(np.float32, copy=False)


def support_filter_mask(
    X: np.ndarray,
    *,
    a_train: float,
    mode: str = "none",
) -> np.ndarray:
    """Return a boolean mask selecting the requested support subset."""
    mode_norm = str(mode or "none").strip().lower()
    n = int(_center_iq(X).shape[0])
    if mode_norm in {"", "none"}:
        return np.ones((n,), dtype=bool)

    if mode_norm != "disk_l2":
        raise ValueError(
            f"Unknown support_filter_mode={mode!r}. Expected 'none' or 'disk_l2'."
        )

    feats = support_feature_dict(X, a_train=a_train)
    r_l2 = feats["r_l2_norm"] * float(a_train)
    radius = float(a_train) * float(np.sqrt(4.0 / 3.0))
    return np.asarray(r_l2 <= radius, dtype=bool)


def support_region_labels(
    X: np.ndarray,
    *,
    a_train: float,
    edge_tau: float = 0.75,
    corner_tau: float = 0.35,
) -> np.ndarray:
    """Return categorical support labels: center / edge / corner."""
    feats = support_feature_dict(X, a_train=a_train)
    r_inf = feats["r_inf_norm"]
    cornerness = feats["cornerness_norm"]
    labels = np.full(r_inf.shape, "center", dtype=object)
    edge_mask = r_inf >= float(edge_tau)
    labels[edge_mask] = "edge"
    labels[edge_mask & (cornerness >= float(corner_tau))] = "corner"
    return labels


def support_experiment_config(
    *,
    feature_mode: Optional[str],
    weight_mode: Optional[str],
    weight_alpha: Optional[float],
    weight_tau: Optional[float],
    weight_tau_corner: Optional[float],
    weight_max: Optional[float],
    filter_mode: Optional[str],
    filter_eval_mode: Optional[str],
    diag_bins: Optional[int],
    a_train: Optional[float] = None,
) -> Dict[str, Any]:
    """Serialize support-aware runtime settings for manifests/state files."""
    return {
        "support_feature_mode": str(feature_mode or "none"),
        "support_weight_mode": str(weight_mode or "none"),
        "support_weight_alpha": float(1.5 if weight_alpha is None else weight_alpha),
        "support_weight_tau": float(0.75 if weight_tau is None else weight_tau),
        "support_weight_tau_corner": float(0.35 if weight_tau_corner is None else weight_tau_corner),
        "support_weight_max": float(3.0 if weight_max is None else weight_max),
        "support_filter_mode": str(filter_mode or "none"),
        "support_filter_eval_mode": str(filter_eval_mode or "matched_support_and_full"),
        "support_diag_bins": int(4 if diag_bins is None else diag_bins),
        "a_train": None if a_train is None else float(a_train),
    }
