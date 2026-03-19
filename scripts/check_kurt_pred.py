from __future__ import annotations

"""
Extrai e compara kurtosis real vs predita (MC-concat) para um run de regime.

Uso:
    python scripts/check_kurt_pred.py
    python scripts/check_kurt_pred.py outputs/exp_.../eval/<regime_id>

O script reconstrói o split de validacao a partir de ``state_run.json`` e
usa ``_quick_cvae_predict(..., mode="mc_concat")`` para verificar kurtosis
na distribuicao preditiva, nao no decoder deterministico.

Criterio G4:
    kurt_excess_pred deve ter o mesmo sinal que kurt_excess_real.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loading import find_dataset_root, load_experiments_as_list
from src.protocol.run import (
    _filter_experiments_for_regime,
    _parse_regime_id_physical,
    _quick_cvae_predict,
)
from src.protocol.split_strategies import apply_split


def _latest_regime_dir() -> Path:
    patterns = [
        "outputs/exp_*/eval/*",
        "outputs/exp_*/eval/*/*",
        "outputs/exp_*/studies/*/regimes/*",
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(ROOT.glob(pattern))
    candidates = [p for p in candidates if (p / "state_run.json").exists()]
    if not candidates:
        raise FileNotFoundError("No regime run found under outputs/exp_*/eval/* or legacy studies/*/regimes/*")
    return candidates[-1].resolve()


def _resolve_path(raw: str | None) -> Path:
    if raw is None:
        return _latest_regime_dir()
    path = Path(raw)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_exp_dir(regime_dir: Path) -> Path:
    for parent in regime_dir.parents:
        if parent.name.startswith("exp_") and (parent / "manifest.json").exists():
            return parent
    raise FileNotFoundError(f"Could not locate exp_* directory for {regime_dir}")


def _load_manifest_entry(exp_dir: Path, regime_dir: Path) -> dict:
    manifest = _read_json(exp_dir / "manifest.json")
    regime_dir_resolved = str(regime_dir.resolve())

    for regime in manifest.get("regimes", []):
        run_dir = regime.get("run_dir")
        if run_dir and str(Path(run_dir).resolve()) == regime_dir_resolved:
            return regime

    for regime in manifest.get("regimes", []):
        if str(regime.get("run_id", "")) == regime_dir.name:
            return regime

    raise KeyError(f"Regime entry not found in {exp_dir / 'manifest.json'} for {regime_dir}")


def _resolve_dataset_root(state: dict) -> Path:
    raw = state.get("dataset_root")
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        if path.exists():
            return path
    return find_dataset_root(verbose=False)


def _stratified_eval_idx(df_split, n_eval: int, seed: int, n_val_total: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n_eval >= n_val_total:
        return np.arange(n_val_total, dtype=np.int64)

    if df_split is None or "n_val" not in getattr(df_split, "columns", []):
        idx = rng.choice(n_val_total, size=n_eval, replace=False)
        idx.sort()
        return idx

    n_val_list = [int(v) for v in df_split["n_val"].tolist()]
    if not n_val_list or int(np.sum(n_val_list)) != int(n_val_total):
        idx = rng.choice(n_val_total, size=n_eval, replace=False)
        idx.sort()
        return idx

    n_exps = len(n_val_list)
    base = n_eval // n_exps
    rem = n_eval % n_exps
    target = np.full(n_exps, base, dtype=np.int64)
    if rem > 0:
        rem_idx = rng.permutation(n_exps)[:rem]
        target[rem_idx] += 1

    cap = np.asarray(n_val_list, dtype=np.int64)
    take = np.minimum(target, cap)
    avail = cap - take
    left = int(n_eval - int(take.sum()))

    while left > 0 and int(avail.sum()) > 0:
        progressed = False
        for i in rng.permutation(n_exps):
            if left <= 0:
                break
            if avail[i] > 0:
                take[i] += 1
                avail[i] -= 1
                left -= 1
                progressed = True
        if not progressed:
            break

    idx_parts = []
    cursor = 0
    for i, n_i in enumerate(n_val_list):
        k_i = int(take[i])
        if k_i > 0:
            if k_i < n_i:
                local = np.sort(rng.choice(n_i, size=k_i, replace=False))
            else:
                local = np.arange(n_i, dtype=np.int64)
            idx_parts.append(cursor + local)
        cursor += n_i

    if not idx_parts:
        idx = rng.choice(n_val_total, size=n_eval, replace=False)
        idx.sort()
        return idx

    idx = np.concatenate(idx_parts, axis=0)
    idx.sort()
    return idx


def _selected_regime_experiments(all_exps: list, entry: dict) -> list:
    selected_paths = [
        str(Path(p).resolve()) for p in entry.get("selected_experiments", []) if p
    ]
    if selected_paths:
        selected_set = set(selected_paths)
        return [exp for exp in all_exps if str(Path(exp[4]).resolve()) in selected_set]

    regime_label = str(entry.get("regime_label") or entry.get("regime_id") or "")
    regime_label = regime_label.split("/")[-1]
    dist_m, curr_mA = _parse_regime_id_physical(regime_label)
    regime = {
        "regime_id": regime_label,
        "distance_m": dist_m,
        "current_mA": curr_mA,
    }
    return _filter_experiments_for_regime(all_exps, regime)


def _build_validation_slice(regime_dir: Path, entry: dict, state: dict, n_eval: int, seed: int):
    dataset_root = _resolve_dataset_root(state)
    all_exps, _ = load_experiments_as_list(dataset_root, verbose=False, reduction_config=None)
    exps = _selected_regime_experiments(all_exps, entry)
    if not exps:
        raise RuntimeError("No experiments available to rebuild the regime validation split")

    data_split = state.get("data_split", {})
    training_cfg = state.get("training_config", {})
    split_strategy = str(
        entry.get("split_strategy")
        or data_split.get("split_mode")
        or "per_experiment"
    )
    val_split = float(data_split.get("validation_split", training_cfg.get("validation_split", 0.2)))
    within_exp_shuffle = bool(data_split.get("within_experiment_shuffle", False))

    (
        _X_train,
        _Y_train,
        _D_train,
        _C_train,
        X_val,
        Y_val,
        D_val,
        C_val,
        df_split,
    ) = apply_split(
        exps,
        strategy=split_strategy,
        val_split=val_split,
        seed=seed,
        within_exp_shuffle=within_exp_shuffle,
    )

    if len(X_val) == 0:
        raise RuntimeError("Validation split is empty")

    n_eval_eff = min(int(n_eval), len(X_val))
    if split_strategy == "per_experiment":
        idx_eval = _stratified_eval_idx(df_split, n_eval_eff, seed, len(X_val))
    else:
        rng = np.random.default_rng(seed)
        idx_eval = rng.choice(len(X_val), size=n_eval_eff, replace=False)
        idx_eval.sort()

    return (
        X_val[idx_eval],
        Y_val[idx_eval],
        D_val[idx_eval],
        C_val[idx_eval],
        dataset_root,
        len(exps),
        len(X_val),
    )


def _same_sign(a: float, b: float) -> bool:
    return np.sign(a) == np.sign(b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check residual kurtosis sign on MC-concat predictions.")
    parser.add_argument(
        "run_dir",
        nargs="?",
        help="Regime directory under outputs/exp_.../eval/<regime_id>",
    )
    parser.add_argument("--n_eval", type=int, default=None, help="Validation samples to inspect.")
    parser.add_argument("--mc_samples", type=int, default=16, help="Number of MC samples.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for split/eval/inference.")
    args = parser.parse_args()

    regime_dir = _resolve_path(args.run_dir)
    state_path = regime_dir / "state_run.json"
    if not state_path.exists():
        print(f"FAIL: state_run.json not found in {regime_dir}")
        sys.exit(1)

    state = _read_json(state_path)
    exp_dir = _find_exp_dir(regime_dir)
    entry = _load_manifest_entry(exp_dir, regime_dir)

    analysis_quick = state.get("analysis_quick", {})
    training_cfg = state.get("training_config", {})
    data_split = state.get("data_split", {})

    seed = int(
        args.seed
        if args.seed is not None
        else data_split.get("seed", training_cfg.get("seed", 42))
    )
    n_eval = int(
        args.n_eval
        if args.n_eval is not None
        else analysis_quick.get("n_eval_samples", 40_000)
    )
    mc_samples = max(1, int(args.mc_samples))

    X_val, Y_val, D_val, C_val, dataset_root, n_exps, n_val_total = _build_validation_slice(
        regime_dir=regime_dir,
        entry=entry,
        state=state,
        n_eval=n_eval,
        seed=seed,
    )

    pred = _quick_cvae_predict(
        regime_dir,
        X_val,
        D_val,
        C_val,
        mc_samples=mc_samples,
        seed=seed,
        mode="mc_concat",
    )
    if pred is None:
        print("FAIL: _quick_cvae_predict returned None")
        sys.exit(1)

    Y_concat, X_tiled, _D_tiled, _C_tiled = pred
    res_real = Y_val - X_val
    res_pred = Y_concat - X_tiled

    k_real_i = float(stats.kurtosis(res_real[:, 0], fisher=True))
    k_real_q = float(stats.kurtosis(res_real[:, 1], fisher=True))
    k_pred_i = float(stats.kurtosis(res_pred[:, 0], fisher=True))
    k_pred_q = float(stats.kurtosis(res_pred[:, 1], fisher=True))
    delta_kurt_l2 = float(np.linalg.norm([k_pred_i - k_real_i, k_pred_q - k_real_q]))

    sign_ok = _same_sign(k_pred_i, k_real_i) and _same_sign(k_pred_q, k_real_q)

    print(f"Regime: {regime_dir.name}")
    print(f"Run dir: {regime_dir}")
    print(f"Dataset root: {dataset_root}")
    print(f"Selected experiments: {n_exps}  val_total={n_val_total:,}  val_eval={len(X_val):,}")
    print(f"mc_samples={mc_samples}  seed={seed}")

    print()
    print("Residual REAL:")
    print(f"  kurt_I={k_real_i:.4f}  kurt_Q={k_real_q:.4f}")
    print(f"  var_I={np.var(res_real[:, 0]):.6f}  var_Q={np.var(res_real[:, 1]):.6f}")

    print()
    print(f"Residual PREDICTED (MC-concat, {mc_samples} samples):")
    print(f"  kurt_I={k_pred_i:.4f}  kurt_Q={k_pred_q:.4f}")
    print(f"  var_I={np.var(res_pred[:, 0]):.6f}  var_Q={np.var(res_pred[:, 1]):.6f}")

    print()
    print(f"Delta_kurt_l2: {delta_kurt_l2:.4f}")
    if sign_ok:
        print("Kurtosis sign match: OK")
        return

    print("Kurtosis sign match: FAIL - predicted distribution has the wrong kurtosis sign")
    sys.exit(1)


if __name__ == "__main__":
    main()
