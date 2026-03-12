# -*- coding: utf-8 -*-
"""
src.config.schema — Typed configuration containers (pure dataclasses).

Provides ``TrainConfig``, ``DataConfig``, and ``RunMeta`` with
``from_dict()`` class methods that coerce types and fill missing keys
from :mod:`src.config.defaults`.

No pydantic dependency — only stdlib ``dataclasses``.

Commit: refactor(step1).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.config.defaults import (
    TRAINING_DEFAULTS,
    MODEL_DEFAULTS,
    DATA_REDUCTION_DEFAULTS,
    ANALYSIS_DEFAULTS,
)


# =====================================================================
# Helpers
# =====================================================================

def _get(src: dict, key: str, default: Any, typ: type | None = None) -> Any:
    """Fetch *key* from *src* with *default*, optionally coercing to *typ*."""
    val = src.get(key, default)
    if val is None:
        return default
    if typ is not None:
        try:
            val = typ(val)
        except (TypeError, ValueError):
            val = default
    return val


# =====================================================================
# TrainConfig
# =====================================================================

@dataclass
class TrainConfig:
    """Hyperparameters for a single cVAE training run."""

    # model architecture
    layer_sizes: List[int] = field(default_factory=lambda: list(MODEL_DEFAULTS["layer_sizes"]))
    latent_dim: int = MODEL_DEFAULTS["latent_dim"]
    activation: str = MODEL_DEFAULTS["activation"]
    dropout: float = MODEL_DEFAULTS["dropout"]

    # loss / regularisation
    beta: float = MODEL_DEFAULTS["beta"]
    free_bits: float = MODEL_DEFAULTS["free_bits"]
    kl_anneal_epochs: int = MODEL_DEFAULTS["kl_anneal_epochs"]

    # optimiser / schedule
    lr: float = MODEL_DEFAULTS["lr"]
    batch_size: int = MODEL_DEFAULTS["batch_size"]
    epochs: int = TRAINING_DEFAULTS["epochs"]
    patience: int = TRAINING_DEFAULTS["patience"]
    reduce_lr_patience: int = TRAINING_DEFAULTS["reduce_lr_patience"]
    early_stop_warmup: int = TRAINING_DEFAULTS["early_stop_warmup"]

    # split
    validation_split: float = TRAINING_DEFAULTS["validation_split"]
    split_mode: str = TRAINING_DEFAULTS["split_mode"]
    per_experiment_split_order: str = TRAINING_DEFAULTS["per_experiment_split_order"]
    within_experiment_shuffle: bool = TRAINING_DEFAULTS["within_experiment_shuffle"]
    shuffle_train_batches: bool = TRAINING_DEFAULTS["shuffle_train_batches"]

    seed: int = TRAINING_DEFAULTS["seed"]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        """Build from a flat dictionary, coercing types and filling defaults."""
        d = d or {}
        return cls(
            layer_sizes=list(d.get("layer_sizes", MODEL_DEFAULTS["layer_sizes"])),
            latent_dim=int(_get(d, "latent_dim", MODEL_DEFAULTS["latent_dim"], int)),
            activation=str(_get(d, "activation", MODEL_DEFAULTS["activation"])),
            dropout=float(_get(d, "dropout", MODEL_DEFAULTS["dropout"], float)),
            beta=float(_get(d, "beta", MODEL_DEFAULTS["beta"], float)),
            free_bits=float(_get(d, "free_bits", MODEL_DEFAULTS["free_bits"], float)),
            kl_anneal_epochs=int(_get(d, "kl_anneal_epochs", MODEL_DEFAULTS["kl_anneal_epochs"], int)),
            lr=float(_get(d, "lr", MODEL_DEFAULTS["lr"], float)),
            batch_size=int(_get(d, "batch_size", MODEL_DEFAULTS["batch_size"], int)),
            epochs=int(_get(d, "epochs", TRAINING_DEFAULTS["epochs"], int)),
            patience=int(_get(d, "patience", TRAINING_DEFAULTS["patience"], int)),
            reduce_lr_patience=int(_get(d, "reduce_lr_patience", TRAINING_DEFAULTS["reduce_lr_patience"], int)),
            early_stop_warmup=int(_get(d, "early_stop_warmup", TRAINING_DEFAULTS["early_stop_warmup"], int)),
            validation_split=float(_get(d, "validation_split", TRAINING_DEFAULTS["validation_split"], float)),
            split_mode=str(_get(d, "split_mode", TRAINING_DEFAULTS["split_mode"])),
            per_experiment_split_order=str(_get(d, "per_experiment_split_order", TRAINING_DEFAULTS["per_experiment_split_order"])),
            within_experiment_shuffle=bool(_get(d, "within_experiment_shuffle", TRAINING_DEFAULTS["within_experiment_shuffle"])),
            shuffle_train_batches=bool(_get(d, "shuffle_train_batches", TRAINING_DEFAULTS["shuffle_train_batches"])),
            seed=int(_get(d, "seed", TRAINING_DEFAULTS["seed"], int)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# DataConfig
# =====================================================================

@dataclass
class DataConfig:
    """Data loading / reduction parameters."""

    target_samples_per_experiment: int = DATA_REDUCTION_DEFAULTS["target_samples_per_experiment"]
    min_samples_per_experiment: int = DATA_REDUCTION_DEFAULTS["min_samples_per_experiment"]
    mode: str = DATA_REDUCTION_DEFAULTS["mode"]
    block_len: int = DATA_REDUCTION_DEFAULTS["block_len"]
    time_spread: bool = DATA_REDUCTION_DEFAULTS["time_spread"]
    min_gap_blocks: int = DATA_REDUCTION_DEFAULTS["min_gap_blocks"]
    max_samples_per_experiment: int = DATA_REDUCTION_DEFAULTS["max_samples_per_experiment"]
    seed: int = DATA_REDUCTION_DEFAULTS["seed"]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataConfig":
        d = d or {}
        return cls(
            target_samples_per_experiment=int(_get(d, "target_samples_per_experiment", DATA_REDUCTION_DEFAULTS["target_samples_per_experiment"], int)),
            min_samples_per_experiment=int(_get(d, "min_samples_per_experiment", DATA_REDUCTION_DEFAULTS["min_samples_per_experiment"], int)),
            mode=str(_get(d, "mode", DATA_REDUCTION_DEFAULTS["mode"])),
            block_len=int(_get(d, "block_len", DATA_REDUCTION_DEFAULTS["block_len"], int)),
            time_spread=bool(_get(d, "time_spread", DATA_REDUCTION_DEFAULTS["time_spread"])),
            min_gap_blocks=int(_get(d, "min_gap_blocks", DATA_REDUCTION_DEFAULTS["min_gap_blocks"], int)),
            max_samples_per_experiment=int(_get(d, "max_samples_per_experiment", DATA_REDUCTION_DEFAULTS["max_samples_per_experiment"], int)),
            seed=int(_get(d, "seed", DATA_REDUCTION_DEFAULTS["seed"], int)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# AnalysisConfig
# =====================================================================

@dataclass
class AnalysisConfig:
    """Quick analysis / ranking configuration shared by train and eval."""

    n_eval_samples: int = ANALYSIS_DEFAULTS["n_eval_samples"]
    batch_infer: int = ANALYSIS_DEFAULTS["batch_infer"]
    rank_mode: str = ANALYSIS_DEFAULTS["rank_mode"]
    mc_samples: int = ANALYSIS_DEFAULTS["mc_samples"]
    dist_metrics: bool = ANALYSIS_DEFAULTS["dist_metrics"]
    psd_nfft: int = ANALYSIS_DEFAULTS["psd_nfft"]
    w_psd: float = ANALYSIS_DEFAULTS["w_psd"]
    w_skew: float = ANALYSIS_DEFAULTS["w_skew"]
    w_kurt: float = ANALYSIS_DEFAULTS["w_kurt"]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnalysisConfig":
        d = d or {}
        return cls(
            n_eval_samples=int(_get(d, "n_eval_samples", ANALYSIS_DEFAULTS["n_eval_samples"], int)),
            batch_infer=int(_get(d, "batch_infer", ANALYSIS_DEFAULTS["batch_infer"], int)),
            rank_mode=str(_get(d, "rank_mode", ANALYSIS_DEFAULTS["rank_mode"])),
            mc_samples=int(_get(d, "mc_samples", ANALYSIS_DEFAULTS["mc_samples"], int)),
            dist_metrics=bool(_get(d, "dist_metrics", ANALYSIS_DEFAULTS["dist_metrics"])),
            psd_nfft=int(_get(d, "psd_nfft", ANALYSIS_DEFAULTS["psd_nfft"], int)),
            w_psd=float(_get(d, "w_psd", ANALYSIS_DEFAULTS["w_psd"], float)),
            w_skew=float(_get(d, "w_skew", ANALYSIS_DEFAULTS["w_skew"], float)),
            w_kurt=float(_get(d, "w_kurt", ANALYSIS_DEFAULTS["w_kurt"], float)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# EvalProtocolConfig
# =====================================================================

@dataclass
class EvalProtocolConfig:
    """Evaluation-time inference protocol stored in ``state_run.json``."""

    n_eval_samples: int = ANALYSIS_DEFAULTS["n_eval_samples"]
    batch_infer: int = ANALYSIS_DEFAULTS["batch_infer"]
    eval_slice: str = "stratified"
    deterministic_inference: bool = False
    rank_mode: str = ANALYSIS_DEFAULTS["rank_mode"]
    mc_samples: int = ANALYSIS_DEFAULTS["mc_samples"]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalProtocolConfig":
        d = d or {}
        rank_mode = str(_get(d, "rank_mode", ANALYSIS_DEFAULTS["rank_mode"]))
        deterministic = bool(
            _get(
                d,
                "deterministic_inference",
                str(rank_mode).lower() == "det",
            )
        )
        return cls(
            n_eval_samples=int(_get(d, "n_eval_samples", ANALYSIS_DEFAULTS["n_eval_samples"], int)),
            batch_infer=int(_get(d, "batch_infer", ANALYSIS_DEFAULTS["batch_infer"], int)),
            eval_slice=str(_get(d, "eval_slice", "stratified")),
            deterministic_inference=deterministic,
            rank_mode=rank_mode,
            mc_samples=int(_get(d, "mc_samples", ANALYSIS_DEFAULTS["mc_samples"], int)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# RunMeta
# =====================================================================

@dataclass
class RunMeta:
    """Metadata for a single training / evaluation run."""

    run_id: str = ""
    run_dir: str = ""
    dataset_root: str = ""
    output_base: str = ""
    timestamp: str = ""

    # Sub-configs (stored as dicts for JSON compat)
    training_config: Dict[str, Any] = field(default_factory=dict)
    data_reduction_config: Dict[str, Any] = field(default_factory=dict)
    analysis_quick: Dict[str, Any] = field(default_factory=dict)
    normalization: Optional[Dict[str, float]] = None
    data_split: Dict[str, Any] = field(default_factory=dict)
    eval_protocol: Dict[str, Any] = field(default_factory=dict)
    grid: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, str] = field(default_factory=dict)
    model_constraints: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunMeta":
        """Reconstruct from a ``state_run.json``-style dict."""
        d = d or {}
        return cls(
            run_id=str(d.get("run_id", "")),
            run_dir=str(d.get("run_dir", "")),
            dataset_root=str(d.get("dataset_root", "")),
            output_base=str(d.get("output_base", "")),
            timestamp=str(d.get("timestamp", "")),
            training_config=dict(d.get("training_config", {})),
            data_reduction_config=dict(d.get("data_reduction_config", {})),
            analysis_quick=dict(d.get("analysis_quick", {})),
            normalization=d.get("normalization"),
            data_split=dict(d.get("data_split", {})),
            eval_protocol=dict(d.get("eval_protocol", {})),
            grid=dict(d.get("grid", {})),
            artifacts=dict(d.get("artifacts", {})),
            paths=dict(d.get("paths", {})),
            model_constraints=dict(d.get("model_constraints", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
