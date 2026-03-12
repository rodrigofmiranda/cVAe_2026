# -*- coding: utf-8 -*-
"""Runtime builders for the canonical training and evaluation engines."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from src.config.defaults import (
    DATA_REDUCTION_DEFAULTS,
    TRAINING_DEFAULTS,
)
from src.config.io import ensure_state_run_compat
from src.config.overrides import RunOverrides
from src.config.schema import AnalysisConfig, EvalProtocolConfig, RunMeta


def _override_dict(
    overrides: Optional[Union[RunOverrides, Mapping[str, Any]]],
) -> Dict[str, Any]:
    if overrides is None:
        return {}
    if isinstance(overrides, RunOverrides):
        return overrides.to_dict()
    return dict(overrides)


@dataclass(frozen=True)
class TrainingRuntime:
    """Explicit runtime bundle for the training engine."""

    dataset_root_hint: Path
    output_base: Path
    run_id: Optional[str]
    overrides: Dict[str, Any]
    training_config: Dict[str, Any]
    data_reduction_config: Dict[str, Any]
    analysis_quick: Dict[str, Any]
    eval_protocol: Dict[str, Any]


@dataclass(frozen=True)
class EvaluationRuntime:
    """Explicit runtime bundle for the evaluation engine."""

    run_dir: Path
    dataset_root: Path
    state: Dict[str, Any]
    overrides: Dict[str, Any]
    analysis_quick: Dict[str, Any]
    eval_protocol: Dict[str, Any]
    training_config: Dict[str, Any]


def build_training_runtime(
    dataset_root: str | Path,
    output_base: str | Path,
    *,
    run_id: Optional[str] = None,
    overrides: Optional[Union[RunOverrides, Mapping[str, Any]]] = None,
) -> TrainingRuntime:
    """Build the canonical training runtime bundle from defaults + overrides."""
    ov = _override_dict(overrides)

    training_config = dict(TRAINING_DEFAULTS)
    if ov.get("val_split") is not None:
        training_config["validation_split"] = float(ov["val_split"])
    if ov.get("seed") is not None:
        training_config["seed"] = int(ov["seed"])
    if ov.get("max_epochs") is not None:
        training_config["epochs"] = int(ov["max_epochs"])
    if ov.get("patience") is not None:
        training_config["patience"] = int(ov["patience"])
    if ov.get("reduce_lr_patience") is not None:
        training_config["reduce_lr_patience"] = int(ov["reduce_lr_patience"])
    if ov.get("_split_strategy") is not None:
        training_config["split_mode"] = str(ov["_split_strategy"])

    data_reduction_config = dict(DATA_REDUCTION_DEFAULTS)
    data_reduction_config.update(
        {
            "enabled": True,
            "n_blocks": 10,
            "bins_r": 10,
            "blocks_per_bin": 5,
        }
    )
    if bool(ov.get("no_data_reduction", False)):
        data_reduction_config["enabled"] = False

    analysis_cfg = AnalysisConfig.from_dict({})
    if ov.get("psd_nfft") is not None:
        analysis_cfg.psd_nfft = int(ov["psd_nfft"])
    eval_cfg = EvalProtocolConfig(
        n_eval_samples=int(analysis_cfg.n_eval_samples),
        batch_infer=int(analysis_cfg.batch_infer),
        eval_slice="stratified",
        deterministic_inference=(str(analysis_cfg.rank_mode).lower() == "det"),
        rank_mode=str(analysis_cfg.rank_mode).lower(),
        mc_samples=int(analysis_cfg.mc_samples),
    )

    return TrainingRuntime(
        dataset_root_hint=Path(dataset_root),
        output_base=Path(output_base),
        run_id=run_id,
        overrides=ov,
        training_config=training_config,
        data_reduction_config=data_reduction_config,
        analysis_quick=analysis_cfg.to_dict(),
        eval_protocol=eval_cfg.to_dict(),
    )


def build_evaluation_runtime(
    run_dir: str | Path,
    *,
    dataset_root: str | Path | None = None,
    state: Optional[Dict[str, Any]] = None,
    overrides: Optional[Union[RunOverrides, Mapping[str, Any]]] = None,
) -> EvaluationRuntime:
    """Build the canonical evaluation runtime bundle from state + overrides."""
    ov = _override_dict(overrides)
    state_data = copy.deepcopy(state or {})
    ensure_state_run_compat(state_data)
    meta = RunMeta.from_dict(state_data)

    resolved_dataset_root = dataset_root or meta.dataset_root
    if not resolved_dataset_root:
        raise ValueError("dataset_root must be provided explicitly or stored in state_run.json")

    analysis_cfg = AnalysisConfig.from_dict(state_data.get("analysis_quick", {}))
    eval_cfg = EvalProtocolConfig.from_dict(state_data.get("eval_protocol", {}))
    training_config = dict(state_data.get("training_config", {}))

    return EvaluationRuntime(
        run_dir=Path(run_dir),
        dataset_root=Path(resolved_dataset_root),
        state=state_data,
        overrides=ov,
        analysis_quick=analysis_cfg.to_dict(),
        eval_protocol=eval_cfg.to_dict(),
        training_config=training_config,
    )
