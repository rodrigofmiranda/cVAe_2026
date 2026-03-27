# -*- coding: utf-8 -*-
"""
src/config/overrides.py — Centralised runtime overrides.

Single source of truth for every knob that the CLI, protocol runner,
or notebook may set.  Downstream modules receive a ``RunOverrides``
instance (or its ``.to_dict()`` for legacy consumers) instead of
reading CLI args directly.

Refactor: core — centralize runtime overrides.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional


@dataclass
class RunOverrides:
    """Runtime overrides applied on top of hard-coded defaults.

    All fields default to *None* (= "not specified — use default").
    A field that is *None* is **omitted** from :meth:`to_dict` so that
    downstream consumers can distinguish "user did not set" from
    "user explicitly set to X".

    Field groups
    ------------
    Training knobs:
        max_epochs, max_grids, grid_group, grid_tag, grid_preset,
        val_split, seed, patience, reduce_lr_patience

    Data knobs:
        max_experiments, max_samples_per_exp, max_val_samples_per_exp

    UI knobs:
        keras_verbose

    Evaluation / distribution-metric knobs:
        psd_nfft, max_dist_samples, gauss_alpha

    Regime-filtering knobs:
        dist_tol_m, curr_tol_mA

    Flags:
        dry_run, skip_eval, no_baseline, no_dist_metrics, no_data_reduction
    """

    # --- Training ---
    max_epochs: Optional[int] = None
    max_grids: Optional[int] = None
    grid_group: Optional[str] = None
    grid_tag: Optional[str] = None
    grid_preset: Optional[str] = None
    val_split: Optional[float] = None
    seed: Optional[int] = None
    patience: Optional[int] = None
    reduce_lr_patience: Optional[int] = None

    # --- Data ---
    max_experiments: Optional[int] = None
    max_samples_per_exp: Optional[int] = None
    max_val_samples_per_exp: Optional[int] = None

    # --- UI ---
    keras_verbose: Optional[int] = None

    # --- Eval / dist metrics ---
    batch_infer: Optional[int] = None
    psd_nfft: Optional[int] = None
    max_dist_samples: Optional[int] = None
    gauss_alpha: Optional[float] = None
    train_regime_diagnostics_enabled: Optional[bool] = None
    train_regime_diagnostics_every: Optional[int] = None
    train_regime_diagnostics_mc_samples: Optional[int] = None
    train_regime_diagnostics_max_samples_per_regime: Optional[int] = None
    train_regime_diagnostics_amplitude_bins: Optional[int] = None
    train_regime_diagnostics_focus_only_0p8m: Optional[bool] = None

    # --- Regime filtering ---
    dist_tol_m: Optional[float] = None
    curr_tol_mA: Optional[float] = None

    # --- Flags ---
    dry_run: Optional[bool] = None
    skip_eval: Optional[bool] = None
    no_baseline: Optional[bool] = None
    no_dist_metrics: Optional[bool] = None
    no_data_reduction: Optional[bool] = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunOverrides":
        """Build from a flat dict — unknown keys are silently ignored."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known and v is not None}
        return cls(**filtered)

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> "RunOverrides":
        """Build from an ``argparse.Namespace`` (CLI args).

        Boolean flags that default to *False* are kept as *None* when
        the user did not set them, so that protocol-level defaults can
        still win.
        """
        d: Dict[str, Any] = {}
        for f in fields(cls):
            val = getattr(ns, f.name, None)
            if val is None:
                continue
            # Store booleans only when truthy (argparse store_true
            # defaults to False, which should remain "not set").
            if isinstance(val, bool) and not val:
                continue
            d[f.name] = val
        return cls(**d)

    @classmethod
    def merge(
        cls,
        protocol_globals: Optional[Dict[str, Any]] = None,
        cli: Optional["RunOverrides"] = None,
    ) -> "RunOverrides":
        """Layer protocol global_settings under CLI overrides.

        Priority: CLI > protocol globals > None.
        """
        base = cls.from_dict(protocol_globals or {})
        if cli is None:
            return base
        merged: Dict[str, Any] = {}
        for f in fields(cls):
            cli_val = getattr(cli, f.name)
            base_val = getattr(base, f.name)
            merged[f.name] = cli_val if cli_val is not None else base_val
        return cls(**merged)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self, *, drop_none: bool = True) -> Dict[str, Any]:
        """Convert to a plain dict.

        Parameters
        ----------
        drop_none : bool
            If *True* (default), keys whose value is *None* are omitted.
            This preserves the legacy semantics where ``"max_epochs" in _ov``
            means "the user explicitly set it".
        """
        d = asdict(self)
        if drop_none:
            return {k: v for k, v in d.items() if v is not None}
        return d

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def effective_keras_verbose(self, default: int = 2) -> int:
        """Return keras verbosity, falling back to *default*."""
        return self.keras_verbose if self.keras_verbose is not None else default

    def __repr__(self) -> str:  # pragma: no cover
        set_fields = {k: v for k, v in asdict(self).items() if v is not None}
        pairs = ", ".join(f"{k}={v!r}" for k, v in set_fields.items())
        return f"RunOverrides({pairs})"
