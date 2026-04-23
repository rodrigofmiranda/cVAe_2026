# -*- coding: utf-8 -*-
"""Canonical cVAE training pipeline without env-driven orchestration."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config.runtime import build_training_runtime
from src.data.loading import (
    find_dataset_root,
    load_experiments_as_list,
    reduce_aligned_arrays,
)
from src.data.support_geometry import (
    compute_support_geometry_stats,
    support_experiment_config,
)
from src.data.normalization import normalize_conditions
from src.data.splits import (
    apply_caps_to_df_split,
    cap_train_samples_per_experiment,
    cap_val_samples_per_experiment,
)
from src.models.cvae import build_cvae
from src.protocol.split_strategies import apply_split
from src.training.grid_plan import select_grid
from src.training.gridsearch import run_gridsearch
from src.training.logging import bootstrap_run, write_state_run


def _filter_selected_experiments(
    exps,
    selected_experiments: List[str],
):
    if not selected_experiments:
        return exps
    selected = [str(p) for p in selected_experiments]
    before = len(exps)

    def _dataset_rel(path_str: str) -> str:
        marker = "dataset_fullsquare_organized/"
        norm = str(path_str).replace("\\", "/")
        if marker in norm:
            return norm.split(marker, 1)[1].rstrip("/")
        return norm.rstrip("/")

    selected_rel = [_dataset_rel(p) for p in selected]

    # Support:
    # - exact absolute match
    # - absolute prefix match (curr_100mA/ -> curr_100mA/full_square_...)
    # - dataset-relative suffix match so protocol JSONs remain portable across
    #   different clone roots (e.g. /home/rodrigo/cVAe_2026_shape_fullcircle
    #   vs /mnt/clone_a/cVAe_2026_shape_fullcircle).
    def _matches(exp_path: str) -> bool:
        exp_norm = str(exp_path).replace("\\", "/")
        exp_rel = _dataset_rel(exp_norm)
        return any(
            exp_norm == s
            or exp_norm.startswith(s.rstrip("/") + "/")
            or exp_rel == sr
            or exp_rel.startswith(sr.rstrip("/") + "/")
            for s, sr in zip(selected, selected_rel)
        )

    filtered = [(X, Y, D, C, p) for X, Y, D, C, p in exps if _matches(str(p))]
    print(
        f"⚡ selected_experiments filter: {before} → {len(filtered)} experiment(s) "
        f"(selected_experiments={list(selected)})"
    )
    return filtered


def _assert_aligned(name: str, *arrays) -> None:
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) != 1:
        raise AssertionError(f"{name} alignment mismatch: lengths={lengths}")


def _build_grid_plan(grid: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "grid_id": i + 1,
                "group": item["group"],
                "tag": item["tag"],
                "cfg_json": json.dumps(item["cfg"], ensure_ascii=False),
                "analysis_quick_overrides_json": json.dumps(
                    item.get("analysis_quick_overrides", {}),
                    ensure_ascii=False,
                ),
                **item["cfg"],
            }
            for i, item in enumerate(grid)
        ]
    )


def _resolve_grid_analysis_quick(
    base_analysis_quick: Dict[str, Any],
    grid: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Resolve the base analysis_quick dict passed into grid-search.

    Grid-search itself supports per-candidate ``analysis_quick_overrides``.
    This helper only decides whether the run-level base dict can be safely
    pre-merged for all candidates, or whether candidate-specific merging must
    be deferred to ``run_gridsearch``.
    """
    merged = dict(base_analysis_quick)
    overrides = [
        dict(item.get("analysis_quick_overrides", {}))
        for item in grid
        if item.get("analysis_quick_overrides")
    ]
    if not overrides:
        return merged

    canonical = overrides[0]
    for other in overrides[1:]:
        if other != canonical:
            print(
                "⚡ analysis_quick overrides vary across candidates; "
                "gridsearch will apply them per candidate."
            )
            return merged

    merged.update(canonical)
    print(f"⚡ analysis_quick overrides from preset: {canonical}")
    return merged


def _resolve_best_candidate_analysis_quick(
    base_analysis_quick: Dict[str, Any],
    grid: List[Dict[str, Any]],
    best_grid_tag: str,
) -> Dict[str, Any]:
    """Return the merged analysis_quick dict for the champion candidate."""
    merged = dict(base_analysis_quick)
    if not best_grid_tag:
        return merged
    for item in grid:
        if str(item.get("tag", "")) != str(best_grid_tag):
            continue
        merged.update(dict(item.get("analysis_quick_overrides", {})))
        return merged
    return merged


def _log_regime_tolerance(overrides: Dict[str, Any], d_unique, c_unique) -> None:
    target_d = overrides.get("_regime_distance_m")
    target_c = overrides.get("_regime_current_mA")
    tol_d = float(overrides.get("dist_tol_m", 0.05))
    tol_c = float(overrides.get("curr_tol_mA", 25.0))

    if target_d is not None:
        for distance in d_unique:
            if abs(distance - float(target_d)) > tol_d:
                print(
                    f"⚠️  D={distance:.3f}m exceeds tolerance of regime target "
                    f"{target_d}m ± {tol_d}m"
                )
    if target_c is not None:
        for current in c_unique:
            if abs(current - float(target_c)) > tol_c:
                print(
                    f"⚠️  C={current:.1f}mA exceeds tolerance of regime target "
                    f"{target_c}mA ± {tol_c}mA"
                )


def _build_support_config(X_train: np.ndarray, overrides: Dict[str, Any]) -> Dict[str, Any]:
    stats = compute_support_geometry_stats(X_train)
    return support_experiment_config(
        feature_mode=overrides.get("support_feature_mode"),
        weight_mode=overrides.get("support_weight_mode"),
        weight_alpha=overrides.get("support_weight_alpha"),
        weight_tau=overrides.get("support_weight_tau"),
        weight_tau_corner=overrides.get("support_weight_tau_corner"),
        weight_max=overrides.get("support_weight_max"),
        filter_mode=overrides.get("support_filter_mode"),
        filter_eval_mode=overrides.get("support_filter_eval_mode"),
        diag_bins=overrides.get("support_diag_bins"),
        a_train=stats.a_train,
    )


def _inject_support_defaults_into_grid(
    grid: List[Dict[str, Any]],
    support_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not grid:
        return grid
    enriched: List[Dict[str, Any]] = []
    for item in grid:
        cfg = dict(item.get("cfg", {}))
        cfg.setdefault("support_feature_mode", str(support_config.get("support_feature_mode", "none")))
        cfg.setdefault("support_weight_mode", str(support_config.get("support_weight_mode", "none")))
        cfg.setdefault("support_weight_alpha", float(support_config.get("support_weight_alpha", 1.5)))
        cfg.setdefault("support_weight_tau", float(support_config.get("support_weight_tau", 0.75)))
        cfg.setdefault("support_weight_tau_corner", float(support_config.get("support_weight_tau_corner", 0.35)))
        cfg.setdefault("support_weight_max", float(support_config.get("support_weight_max", 3.0)))
        cfg.setdefault("support_filter_mode", str(support_config.get("support_filter_mode", "none")))
        cfg["support_feature_scale"] = float(support_config["a_train"])
        enriched.append({**item, "cfg": cfg})
    return enriched


def run_training_pipeline(
    dataset_root: str | Path,
    output_base: str | Path,
    *,
    run_id: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the canonical training pipeline and return a summary dict."""
    runtime = build_training_runtime(
        dataset_root=dataset_root,
        output_base=output_base,
        run_id=run_id,
        overrides=overrides,
    )
    ov = dict(runtime.overrides)
    shared_logs_dir = ov.get("_logs_dir")
    run_paths = bootstrap_run(
        output_base=runtime.output_base,
        run_id=runtime.run_id,
        logs_dir=shared_logs_dir,
    )

    print(f"📁 DATASET_ROOT (hint) = {runtime.dataset_root_hint}")
    print(f"📦 OUTPUT_BASE        = {runtime.output_base}")
    print(f"🏷️  RUN_ID            = {run_paths.run_id}")
    print(f"📌 RUN_DIR            = {run_paths.run_dir}")

    np.random.seed(int(runtime.training_config["seed"]))
    tf.random.set_seed(int(runtime.training_config["seed"]))

    print("\n🔎 Localizando dataset...")
    resolved_dataset_root = find_dataset_root(
        marker_dirname="dataset_fullsquare_organized",
        dataset_root_hint=runtime.dataset_root_hint,
        verbose=True,
    )

    print("\n📦 Carregando experimentos (sem redução; split por experimento)...")
    exps, _df_info = load_experiments_as_list(
        resolved_dataset_root,
        verbose=True,
        reduction_config=None,
    )
    available_paths = [str(p) for *_, p in exps]

    selected_experiments = list(ov.get("_selected_experiments", []))
    exps = _filter_selected_experiments(exps, selected_experiments)
    if selected_experiments and not exps:
        raise RuntimeError(
            "No loaded experiments match _selected_experiments. "
            f"Available paths: {available_paths}"
        )

    if ov.get("max_experiments") is not None:
        max_experiments = int(ov["max_experiments"])
        exps = exps[:max_experiments]
        print(f"⚡ max_experiments aplicado: {len(exps)} experiment(s)")

    if not exps:
        raise RuntimeError("Training pipeline resolved zero experiments after filtering.")

    (
        X_train,
        Y_train,
        D_train,
        C_train,
        X_val,
        Y_val,
        D_val,
        C_val,
        df_split,
    ) = apply_split(
        exps=exps,
        strategy=str(runtime.training_config["split_mode"]),
        val_split=float(runtime.training_config["validation_split"]),
        seed=int(runtime.training_config["seed"]),
        within_exp_shuffle=bool(runtime.training_config["within_experiment_shuffle"]),
    )

    print(
        f"\n✓ Dados ({runtime.training_config['split_mode']}): "
        f"{len(X_train):,} treino | {len(X_val):,} validação"
    )

    _df_train_cap = None
    _df_val_cap = None
    if ov.get("max_samples_per_exp") is not None:
        max_samples = int(ov["max_samples_per_exp"])
        X_train, Y_train, D_train, C_train, _df_train_cap = cap_train_samples_per_experiment(
            X_train,
            Y_train,
            D_train,
            C_train,
            df_split,
            max_samples,
        )
        print(
            "⚡ train capped pós-split "
            f"(max_samples_per_exp={max_samples}) | "
            f"train={len(X_train):,} | val={len(X_val):,} (val pre-cap)"
        )

    if ov.get("max_val_samples_per_exp") is not None:
        max_val_samples = int(ov["max_val_samples_per_exp"])
        X_val, Y_val, D_val, C_val, _df_val_cap = cap_val_samples_per_experiment(
            X_val,
            Y_val,
            D_val,
            C_val,
            df_split,
            max_val_samples,
        )
        print(
            "⚡ val capped pós-split "
            f"(max_val_samples_per_exp={max_val_samples}) | "
            f"train={len(X_train):,} | val={len(X_val):,}"
        )

    if _df_train_cap is not None or _df_val_cap is not None:
        df_split = apply_caps_to_df_split(
            df_split,
            df_train_cap=_df_train_cap,
            df_val_cap=_df_val_cap,
        )

    # Guard: seq_bigru_residual is incompatible with balanced_blocks (early check via overrides).
    _arch_override = str(ov.get("arch_variant", "")).strip().lower()
    if _arch_override == "seq_bigru_residual":
        _dr_enabled = bool(runtime.data_reduction_config.get("enabled", False))
        _dr_mode = str(runtime.data_reduction_config.get("mode", "balanced_blocks")).lower()
        if _dr_enabled and _dr_mode == "balanced_blocks":
            raise ValueError(
                "arch_variant='seq_bigru_residual' requires contiguous temporal data; "
                "mode='balanced_blocks' breaks temporal context needed for windowing. "
                "Use no_data_reduction=True or set data_reduction mode='center_crop'."
            )

    if runtime.data_reduction_config.get("enabled", False):
        rng_red = np.random.default_rng(int(runtime.training_config.get("seed", 42)))
        X_train, Y_train, D_train, C_train = reduce_aligned_arrays(
            X_train,
            Y_train,
            D_train,
            C_train,
            cfg=runtime.data_reduction_config,
            rng=rng_red,
        )
        print(
            f"✓ Data reduction pós-split: train={len(X_train):,} | "
            f"val={len(X_val):,} (val intocado)"
        )

    _assert_aligned("Train", X_train, Y_train, D_train, C_train)
    _assert_aligned("Val", X_val, Y_val, D_val, C_val)

    Dn_train, Cn_train, Dn_val, Cn_val, norm_params = normalize_conditions(
        D_train, C_train, D_val, C_val
    )

    d_unique = sorted(np.unique(D_train).tolist())
    c_unique = sorted(np.unique(C_train).tolist())
    support_config = _build_support_config(X_train, ov)
    print(
        f"✓ Distância (treino): [{norm_params['D_min']:.3f}, {norm_params['D_max']:.3f}] m  "
        f"unique={d_unique}"
    )
    print(
        f"✓ Corrente  (treino): [{norm_params['C_min']:.1f}, {norm_params['C_max']:.1f}] mA  "
        f"unique={c_unique}"
    )

    if selected_experiments:
        _log_regime_tolerance(ov, d_unique, c_unique)

    grid = select_grid(ov)
    grid = _inject_support_defaults_into_grid(grid, support_config)

    # Guard: comprehensive check after grid selection — catches arch_variant set per-cfg.
    _seq_in_grid = any(
        str(item["cfg"].get("arch_variant", "")).strip().lower() == "seq_bigru_residual"
        for item in grid
    )
    if _seq_in_grid:
        _dr_enabled = bool(runtime.data_reduction_config.get("enabled", False))
        _dr_mode = str(runtime.data_reduction_config.get("mode", "balanced_blocks")).lower()
        if _dr_enabled and _dr_mode == "balanced_blocks":
            raise ValueError(
                "arch_variant='seq_bigru_residual' is incompatible with "
                "data_reduction mode='balanced_blocks': non-contiguous block selection "
                "breaks temporal context required for windowing. "
                "Use no_data_reduction=True or set data_reduction mode='center_crop'."
            )

    analysis_quick = _resolve_grid_analysis_quick(runtime.analysis_quick, grid)
    df_plan = _build_grid_plan(grid)

    if ov.get("dry_run", False):
        first_cfg = grid[0]["cfg"] if grid else {}
        if first_cfg:
            vae, _ = build_cvae(first_cfg)
            print(f"🔍 dry_run: model built | params={vae.count_params():,}")
            vae.summary(print_fn=lambda s: print("  " + s))
            del vae
        dry_info = {
            "dry_run": True,
            "overrides": {k: v for k, v in ov.items()},
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_grid": int(len(grid)),
        }
        run_paths.write_json("logs/dry_run.json", dry_info)
        print(f"✅ dry_run complete — wrote {run_paths.logs_dir / 'dry_run.json'}")
        gc.collect()
        return {
            "run_id": run_paths.run_id,
            "run_dir": str(run_paths.run_dir),
            "status": "dry_run",
        }

    df_results = run_gridsearch(
        grid=grid,
        training_config=runtime.training_config,
        analysis_quick=analysis_quick,
        X_train=X_train,
        Y_train=Y_train,
        Dn_train=Dn_train,
        Cn_train=Cn_train,
        X_val=X_val,
        Y_val=Y_val,
        Dn_val=Dn_val,
        Cn_val=Cn_val,
        D_train_raw=D_train,
        C_train_raw=C_train,
        D_val_raw=D_val,
        C_val_raw=C_val,
        run_paths=run_paths,
        overrides=ov,
        df_plan=df_plan,
        df_split=df_split,
        support_config=support_config,
    )

    grid_csv_path = run_paths.run_dir / "tables" / "gridsearch_results.csv"
    grid_xlsx_path = run_paths.run_dir / "tables" / "gridsearch_results.xlsx"
    grid_diag_csv_path = run_paths.run_dir / "tables" / "grid_training_diagnostics.csv"
    training_dashboard_path = (
        run_paths.run_dir / "plots" / "training" / "dashboard_analysis_complete.png"
    )
    best_grid_tag = (
        str(df_results.iloc[0]["tag"])
        if isinstance(df_results, pd.DataFrame) and not df_results.empty and "tag" in df_results.columns
        else ""
    )
    best_score_v2 = (
        float(df_results.iloc[0]["score_v2"])
        if isinstance(df_results, pd.DataFrame)
        and not df_results.empty
        and "score_v2" in df_results.columns
        and np.isfinite(df_results.iloc[0]["score_v2"])
        else float("nan")
    )
    champion_analysis_quick = _resolve_best_candidate_analysis_quick(
        analysis_quick,
        grid,
        best_grid_tag,
    )
    artifacts = {
        "grid_results_csv": str(grid_csv_path),
        "grid_results_xlsx": str(grid_xlsx_path),
        "grid_training_diagnostics_csv": str(grid_diag_csv_path),
        "training_dashboard_png": str(training_dashboard_path),
        "best_model_full": str(run_paths.models_dir / "best_model_full.keras"),
        "best_decoder": str(run_paths.models_dir / "best_decoder.keras"),
        "best_prior_net": str(run_paths.models_dir / "best_prior_net.keras"),
        "training_history_json": str(run_paths.logs_dir / "training_history.json"),
    }
    data_split = {
        "split_mode": str(runtime.training_config["split_mode"]),
        "per_experiment_split_order": str(runtime.training_config["per_experiment_split_order"]),
        "within_experiment_shuffle": bool(runtime.training_config["within_experiment_shuffle"]),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "validation_split": float(runtime.training_config["validation_split"]),
        "seed": int(runtime.training_config["seed"]),
        "seq_variant_detected": bool(_seq_in_grid),
    }
    grid_state = {
        "n_models": int(len(grid)),
        "grid_results_csv": str(grid_csv_path),
        "grid_results_xlsx": str(grid_xlsx_path),
        "grid_training_diagnostics_csv": str(grid_diag_csv_path),
        "training_dashboard_png": str(training_dashboard_path),
        "best_grid_tag": best_grid_tag,
        "best_score_v2": best_score_v2,
    }
    best_support_config = dict(support_config)
    if best_grid_tag:
        for item in grid:
            if str(item.get("tag", "")) != str(best_grid_tag):
                continue
            best_support_config.update(
                {
                    "support_feature_mode": str(item["cfg"].get("support_feature_mode", best_support_config["support_feature_mode"])),
                    "support_weight_mode": str(item["cfg"].get("support_weight_mode", best_support_config["support_weight_mode"])),
                    "support_weight_alpha": float(item["cfg"].get("support_weight_alpha", best_support_config["support_weight_alpha"])),
                    "support_weight_tau": float(item["cfg"].get("support_weight_tau", best_support_config["support_weight_tau"])),
                    "support_weight_tau_corner": float(item["cfg"].get("support_weight_tau_corner", best_support_config["support_weight_tau_corner"])),
                    "support_weight_max": float(item["cfg"].get("support_weight_max", best_support_config["support_weight_max"])),
                    "support_filter_mode": str(item["cfg"].get("support_filter_mode", best_support_config["support_filter_mode"])),
                }
            )
            break

    state_path = write_state_run(
        run_paths.run_dir,
        run_id=run_paths.run_id,
        dataset_root=str(resolved_dataset_root),
        output_base=str(runtime.output_base.resolve()),
        training_config=runtime.training_config,
        data_reduction_config=runtime.data_reduction_config,
        analysis_quick=champion_analysis_quick,
        normalization={k: float(v) for k, v in norm_params.items()},
        data_split=data_split,
        eval_protocol=runtime.eval_protocol,
        grid=grid_state,
        artifacts=artifacts,
        extra={"support_config": best_support_config},
        logs_dir=run_paths.logs_dir,
    )
    print(f"✓ state_run.json salvo: {state_path}")

    gc.collect()
    return {
        "run_id": run_paths.run_id,
        "run_dir": str(run_paths.run_dir),
        "status": "completed",
        "state_run_path": str(state_path),
    }
