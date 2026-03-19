"""
Verifica _quick_cvae_predict com dados REAIS de validacao (tail por experimento).

Objetivo:
- Validar contrato de shape da concatenacao MC.
- Evitar diagnostico fragil com X sintetico.
- Reportar kurtosis do residuo real vs predito no proprio regime do run.

Uso:
    python scripts/verify_mc_predict.py
    python scripts/verify_mc_predict.py --run_dir outputs/exp_.../eval/dist_1m__curr_300mA
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loading import load_experiments_as_list
from src.protocol.run import _quick_cvae_predict
from src.protocol.split_strategies import apply_split


def _latest_regime_dir() -> Path:
    patterns = [
        str(ROOT / "outputs/exp_*/eval/*"),
        str(ROOT / "outputs/exp_*/eval/*/*"),
        str(ROOT / "outputs/exp_*/studies/within_regime/regimes/*"),
    ]
    cands = []
    for pattern in patterns:
        cands.extend(glob.glob(pattern))
    if not cands:
        raise FileNotFoundError("Nenhum regime em outputs/exp_*/eval/* nem no layout legado studies/.../regimes/*")
    return Path(cands[-1]).resolve()


def _find_exp_dir(regime_dir: Path) -> Path:
    for p in regime_dir.parents:
        if p.name.startswith("exp_") and (p / "manifest.json").exists():
            return p
    raise FileNotFoundError(f"Nao achei exp_dir para regime {regime_dir}")


def _load_manifest_entry(exp_dir: Path, regime_dir: Path) -> dict:
    manifest_path = exp_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    regime_dir_res = str(regime_dir.resolve())
    entries = manifest.get("regimes", [])

    for r in entries:
        run_dir = r.get("run_dir", "")
        if run_dir and str(Path(run_dir).resolve()) == regime_dir_res:
            return r

    run_id = regime_dir.name
    for r in entries:
        if str(r.get("run_id", "")) == run_id:
            return r

    raise KeyError(f"Run do regime {regime_dir} nao encontrado em {manifest_path}")


def _stratified_eval_idx(df_split, n_eval: int, seed: int, n_val_total: int) -> np.ndarray:
    """Replica a ideia de estratificacao por experimento da avaliacao."""
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

    idx_eval = np.concatenate(idx_parts, axis=0)
    idx_eval.sort()
    return idx_eval


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Verifica MC predict com tail real de validacao.")
    p.add_argument("--run_dir", type=str, default=None, help="Diretorio do regime no formato outputs/.../regimes/<id>")
    p.add_argument("--n_eval", type=int, default=None, help="Numero de amostras de validacao (padrao: analysis_quick.n_eval_samples)")
    p.add_argument("--mc_samples", type=int, default=None, help="Numero de amostras MC (padrao: analysis_quick.mc_samples)")
    p.add_argument("--seed", type=int, default=None, help="Seed para split/amostragem/inferencia")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    regime_dir = Path(args.run_dir).resolve() if args.run_dir else _latest_regime_dir()
    state_path = regime_dir / "state_run.json"
    if not state_path.exists():
        raise FileNotFoundError(f"state_run.json ausente em {regime_dir}")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    data_split = state.get("data_split", {})
    analysis_quick = state.get("analysis_quick", {})
    training_cfg = state.get("training_config", {})

    dataset_root = Path(state.get("dataset_root", "")).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root invalido em state_run.json: {dataset_root}")

    val_split = float(data_split.get("validation_split", training_cfg.get("validation_split", 0.2)))
    seed = int(args.seed if args.seed is not None else data_split.get("seed", training_cfg.get("seed", 42)))
    within_exp_shuffle = bool(data_split.get("within_experiment_shuffle", False))
    n_eval = int(args.n_eval if args.n_eval is not None else analysis_quick.get("n_eval_samples", 40_000))
    mc_samples = max(1, int(args.mc_samples if args.mc_samples is not None else analysis_quick.get("mc_samples", 8)))

    exp_dir = _find_exp_dir(regime_dir)
    entry = _load_manifest_entry(exp_dir, regime_dir)
    selected_paths = [str(Path(p).resolve()) for p in entry.get("selected_experiments", [])]

    all_exps, _df_info = load_experiments_as_list(dataset_root, verbose=False, reduction_config=None)
    if selected_paths:
        selected_set = set(selected_paths)
        exps = [e for e in all_exps if str(Path(e[4]).resolve()) in selected_set]
    else:
        exps = list(all_exps)

    if not exps:
        raise RuntimeError("Nenhum experimento selecionado para reproduzir o split de validacao.")

    (
        _X_tr, _Y_tr, _D_tr, _C_tr,
        X_val, Y_val, D_val, C_val,
        df_split,
    ) = apply_split(
        exps,
        strategy="per_experiment",
        val_split=val_split,
        seed=seed,
        within_exp_shuffle=within_exp_shuffle,
    )

    if len(X_val) == 0:
        raise RuntimeError("Split produziu validacao vazia.")

    n_eval_eff = min(n_eval, len(X_val))
    idx_eval = _stratified_eval_idx(df_split, n_eval_eff, seed, len(X_val))
    Xv, Yv, Dv, Cv = X_val[idx_eval], Y_val[idx_eval], D_val[idx_eval], C_val[idx_eval]

    result = _quick_cvae_predict(regime_dir, Xv, Dv, Cv, mc_samples=mc_samples, seed=seed)
    if result is None:
        raise RuntimeError(f"_quick_cvae_predict retornou None em {regime_dir}")

    Y_concat, X_tiled, D_tiled, C_tiled = result
    expected_n = len(Xv) * mc_samples
    assert Y_concat.shape == (expected_n, 2), f"Y_concat esperado {(expected_n, 2)}, obtido {Y_concat.shape}"
    assert X_tiled.shape == (expected_n, 2), f"X_tiled esperado {(expected_n, 2)}, obtido {X_tiled.shape}"
    assert D_tiled.shape[0] == expected_n and C_tiled.shape[0] == expected_n

    res_real = Yv - Xv
    res_pred = Y_concat - X_tiled

    ki_real = float(stats.kurtosis(res_real[:, 0], fisher=True))
    kq_real = float(stats.kurtosis(res_real[:, 1], fisher=True))
    ki_pred = float(stats.kurtosis(res_pred[:, 0], fisher=True))
    kq_pred = float(stats.kurtosis(res_pred[:, 1], fisher=True))
    delta_kurt_l2 = float(np.linalg.norm([ki_pred - ki_real, kq_pred - kq_real]))

    print(f"run_dir={regime_dir}")
    print(f"dataset_root={dataset_root}")
    print(f"selected_experiments={len(exps)}  n_val_total={len(X_val):,}  n_eval={len(Xv):,}")
    print(f"mc_samples={mc_samples}  seed={seed}")
    print(f"REAL kurt_excess: I={ki_real:.4f}  Q={kq_real:.4f}")
    print(f"PRED kurt_excess: I={ki_pred:.4f}  Q={kq_pred:.4f}")
    print(f"delta_kurt_l2={delta_kurt_l2:.6f}")
    print("✅ MC predict OK (tail real de validacao)")


if __name__ == "__main__":
    main()
