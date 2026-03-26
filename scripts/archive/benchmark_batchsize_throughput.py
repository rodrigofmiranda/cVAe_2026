#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark de throughput vs batch_size para cVAE (VLC digital twin).

Objetivo:
- Medir ganho real de throughput ao aumentar batch_size.
- Comparar pipeline numpy (fit com arrays) vs tf.data (cache + prefetch).
- Verificar se a qualidade basica (EVM/SNR/variancia) degrada com batch muito alto.

Principios metodologicos preservados:
- Split temporal por experimento (head=train, tail=val).
- Sem leakage temporal.
- Cap/reducao aplicados apenas no treino.
- Validacao nunca reduzida/capada.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

# Allow `python scripts/benchmark_batchsize_throughput.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loading import load_experiments_as_list
from src.data.normalization import normalize_conditions
from src.data.splits import cap_train_samples_per_experiment
from src.evaluation.metrics import (
    calculate_evm,
    calculate_snr,
    residual_distribution_metrics,
)
from src.models.cvae import build_cvae, create_inference_model_from_full
from src.protocol.selector_engine import select_experiments
from src.protocol.split_strategies import apply_split


def _parse_regime_id(rid: str) -> Tuple[float, float]:
    m = re.match(r"^dist_([0-9p.]+)m__curr_([0-9]+)mA$", str(rid).strip())
    if m is None:
        raise ValueError(
            f"regime_id invalido: {rid!r}. Use formato dist_1p0m__curr_300mA."
        )
    dist = float(m.group(1).replace("p", "."))
    curr = float(m.group(2))
    return dist, curr


def _parse_int_list(s: str) -> List[int]:
    vals = []
    for t in str(s).split(","):
        t = t.strip()
        if not t:
            continue
        vals.append(int(t))
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("Lista de batch sizes vazia.")
    return vals


def _parse_pipelines(s: str) -> List[str]:
    vals = []
    for t in str(s).split(","):
        t = t.strip().lower()
        if not t:
            continue
        if t not in {"numpy", "tfdata"}:
            raise ValueError(f"pipeline invalido: {t!r}. Use numpy,tfdata.")
        vals.append(t)
    vals = list(dict.fromkeys(vals))
    if not vals:
        raise ValueError("Lista de pipelines vazia.")
    return vals


class EpochTimer(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_times_s: List[float] = []
        self._t0: Optional[float] = None

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        if self._t0 is None:
            return
        dt = time.perf_counter() - self._t0
        self.epoch_times_s.append(float(dt))


def _stratified_val_indices_by_experiment(
    *,
    n_total: int,
    n_val_total: int,
    df_split: Optional[pd.DataFrame],
    seed: int,
) -> np.ndarray:
    n_total = int(max(1, min(n_total, n_val_total)))
    rng = np.random.default_rng(int(seed))
    idx_eval = None

    if isinstance(df_split, pd.DataFrame) and "n_val" in df_split.columns:
        n_val_list = [int(v) for v in df_split["n_val"].tolist()]
        if n_val_list and int(np.sum(n_val_list)) == int(n_val_total):
            n_exps = len(n_val_list)
            base = n_total // n_exps
            rem = n_total % n_exps

            target = np.full(n_exps, base, dtype=np.int64)
            if rem > 0:
                rem_idx = rng.permutation(n_exps)[:rem]
                target[rem_idx] += 1

            cap = np.asarray(n_val_list, dtype=np.int64)
            take = np.minimum(target, cap)
            avail = cap - take
            left = int(n_total - int(take.sum()))

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

            if idx_parts:
                idx_eval = np.concatenate(idx_parts, axis=0)
                idx_eval.sort()

    if idx_eval is None:
        idx_eval = rng.choice(n_val_total, size=n_total, replace=False)
        idx_eval.sort()
    return idx_eval


@dataclass
class TrialResult:
    pipeline: str
    batch_size: int
    n_train: int
    n_val: int
    epochs_total: int
    warmup_epochs: int
    measured_epochs: int
    epoch_time_mean_s: float
    epoch_time_std_s: float
    steps_per_epoch: int
    step_time_mean_ms: float
    samples_per_s: float
    val_loss_last: float
    val_recon_last: float
    evm_real_pct: float
    evm_pred_pct: float
    delta_evm_pp: float
    snr_real_db: float
    snr_pred_db: float
    delta_snr_db: float
    var_real_delta: float
    var_pred_delta: float
    var_ratio_pred_real: float


def _build_tfdata(
    *,
    X_train: np.ndarray,
    Dn_train: np.ndarray,
    Cn_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Dn_val: np.ndarray,
    Cn_val: np.ndarray,
    Y_val: np.ndarray,
    batch_size: int,
    seed: int,
    cache: bool,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    dtr = Dn_train.reshape(-1, 1)
    ctr = Cn_train.reshape(-1, 1)
    dva = Dn_val.reshape(-1, 1)
    cva = Cn_val.reshape(-1, 1)

    train_ds = tf.data.Dataset.from_tensor_slices(
        ((X_train, dtr, ctr, Y_train), Y_train)
    )
    if cache:
        train_ds = train_ds.cache()
    shuffle_buffer = min(len(X_train), max(int(batch_size) * 10, 65_536))
    train_ds = train_ds.shuffle(
        buffer_size=int(shuffle_buffer),
        seed=int(seed),
        reshuffle_each_iteration=True,
    )
    train_ds = train_ds.batch(int(batch_size), drop_remainder=False)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        ((X_val, dva, cva, Y_val), Y_val)
    )
    if cache:
        val_ds = val_ds.cache()
    val_ds = val_ds.batch(int(batch_size), drop_remainder=False)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    train_ds = train_ds.with_options(opts)
    val_ds = val_ds.with_options(opts)
    return train_ds, val_ds


def _run_trial(
    *,
    pipeline: str,
    batch_size: int,
    cfg: Dict,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    Dn_train: np.ndarray,
    Cn_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    Dn_val: np.ndarray,
    Cn_val: np.ndarray,
    df_split: pd.DataFrame,
    epochs: int,
    warmup_epochs: int,
    seed: int,
    keras_verbose: int,
    use_cache: bool,
    n_eval_samples: int,
) -> TrialResult:
    tf.keras.backend.clear_session()
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))

    cfg_local = dict(cfg)
    cfg_local["batch_size"] = int(batch_size)
    vae, kl_cb = build_cvae(cfg_local)

    timer_cb = EpochTimer()
    callbacks = [kl_cb, timer_cb]

    t0 = time.perf_counter()
    if pipeline == "tfdata":
        train_ds, val_ds = _build_tfdata(
            X_train=X_train, Dn_train=Dn_train, Cn_train=Cn_train, Y_train=Y_train,
            X_val=X_val, Dn_val=Dn_val, Cn_val=Cn_val, Y_val=Y_val,
            batch_size=int(batch_size), seed=int(seed), cache=bool(use_cache),
        )
        hist = vae.fit(
            train_ds,
            validation_data=val_ds,
            epochs=int(epochs),
            verbose=int(keras_verbose),
            callbacks=callbacks,
        )
        steps_per_epoch = int(math.ceil(len(X_train) / int(batch_size)))
    else:
        hist = vae.fit(
            [X_train, Dn_train, Cn_train, Y_train],
            Y_train,
            validation_data=([X_val, Dn_val, Cn_val, Y_val], Y_val),
            epochs=int(epochs),
            batch_size=int(batch_size),
            verbose=int(keras_verbose),
            callbacks=callbacks,
            shuffle=True,
        )
        steps_per_epoch = int(math.ceil(len(X_train) / int(batch_size)))
    _ = time.perf_counter() - t0

    times = timer_cb.epoch_times_s
    measured = times[int(warmup_epochs):] if len(times) > int(warmup_epochs) else times
    if not measured:
        measured = times if times else [float("nan")]
    epoch_time_mean_s = float(np.nanmean(measured))
    epoch_time_std_s = float(np.nanstd(measured))
    samples_per_s = float(len(X_train) / epoch_time_mean_s) if epoch_time_mean_s > 0 else float("nan")
    step_time_ms = float((epoch_time_mean_s / max(steps_per_epoch, 1)) * 1000.0) if epoch_time_mean_s > 0 else float("nan")

    idx_eval = _stratified_val_indices_by_experiment(
        n_total=min(int(n_eval_samples), len(X_val)),
        n_val_total=len(X_val),
        df_split=df_split,
        seed=int(seed),
    )
    Xv = X_val[idx_eval]
    Yv = Y_val[idx_eval]
    Dv = Dn_val[idx_eval]
    Cv = Cn_val[idx_eval]

    infer = create_inference_model_from_full(vae, deterministic=True)
    Yp = infer.predict([Xv, Dv, Cv], batch_size=int(max(2048, batch_size)), verbose=0)

    evm_real, _ = calculate_evm(Xv, Yv)
    evm_pred, _ = calculate_evm(Xv, Yp)
    snr_real = calculate_snr(Xv, Yv)
    snr_pred = calculate_snr(Xv, Yp)
    distm = residual_distribution_metrics(Xv, Yv, Yp, psd_nfft=2048)
    var_real = float(distm["var_real_delta"])
    var_pred = float(distm["var_pred_delta"])
    var_ratio = float(var_pred / var_real) if var_real > 0 else float("nan")

    val_loss_hist = hist.history.get("val_loss", [])
    val_recon_hist = hist.history.get("val_recon_loss", [])
    val_loss_last = float(val_loss_hist[-1]) if val_loss_hist else float("nan")
    val_recon_last = float(val_recon_hist[-1]) if val_recon_hist else float("nan")

    del infer, vae, hist
    tf.keras.backend.clear_session()

    return TrialResult(
        pipeline=str(pipeline),
        batch_size=int(batch_size),
        n_train=int(len(X_train)),
        n_val=int(len(X_val)),
        epochs_total=int(epochs),
        warmup_epochs=int(warmup_epochs),
        measured_epochs=int(max(0, epochs - warmup_epochs)),
        epoch_time_mean_s=epoch_time_mean_s,
        epoch_time_std_s=epoch_time_std_s,
        steps_per_epoch=int(steps_per_epoch),
        step_time_mean_ms=step_time_ms,
        samples_per_s=samples_per_s,
        val_loss_last=val_loss_last,
        val_recon_last=val_recon_last,
        evm_real_pct=float(evm_real),
        evm_pred_pct=float(evm_pred),
        delta_evm_pp=float(evm_pred - evm_real),
        snr_real_db=float(snr_real),
        snr_pred_db=float(snr_pred),
        delta_snr_db=float(snr_pred - snr_real),
        var_real_delta=var_real,
        var_pred_delta=var_pred,
        var_ratio_pred_real=var_ratio,
    )


def _default_cfg() -> Dict:
    # Config alinhado ao grid de referencia.
    return {
        "activation": "leaky_relu",
        "kl_anneal_epochs": 80,
        "batch_size": 16384,  # sobrescrito por trial
        "lr": 3e-4,
        "dropout": 0.0,
        "free_bits": 0.10,
        "layer_sizes": [128, 256, 512],
        "latent_dim": 4,
        "beta": 0.003,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark de throughput cVAE por batch_size (numpy vs tf.data)."
    )
    p.add_argument("--dataset_root", type=str, default="data/dataset_fullsquare_organized")
    p.add_argument("--regime_id", type=str, default="dist_1p0m__curr_300mA")
    p.add_argument("--dist_tol_m", type=float, default=0.05)
    p.add_argument("--curr_tol_mA", type=float, default=25.0)
    p.add_argument("--max_experiments", type=int, default=1)
    p.add_argument("--val_split", type=float, default=0.20)
    p.add_argument("--max_samples_per_exp", type=int, default=200_000)
    p.add_argument("--batch_sizes", type=str, default="2048,4096,8192,16384,32768")
    p.add_argument("--pipelines", type=str, default="numpy,tfdata")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument("--n_eval_samples", type=int, default=40_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keras_verbose", type=int, default=2, choices=[0, 1, 2])
    p.add_argument("--tfdata_cache", action="store_true")
    p.add_argument("--output_base", type=str, default="outputs/benchmarks")
    return p.parse_args()


def main():
    args = parse_args()
    batch_sizes = _parse_int_list(args.batch_sizes)
    pipelines = _parse_pipelines(args.pipelines)
    dist_m, curr_mA = _parse_regime_id(args.regime_id)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU devices: {len(gpus)}")
    for i, g in enumerate(gpus):
        print(f"  - GPU[{i}]: {g}")

    dataset_root = Path(args.dataset_root).resolve()
    print(f"Dataset root: {dataset_root}")

    print("Carregando inventario de experimentos...")
    exps, _ = load_experiments_as_list(dataset_root, verbose=False, reduction_config=None)
    exps = select_experiments(
        exps,
        selector={"distance_m": dist_m, "current_mA": curr_mA},
        dist_tol=float(args.dist_tol_m),
        curr_tol=float(args.curr_tol_mA),
        label=args.regime_id,
    )
    if args.max_experiments is not None:
        exps = exps[: int(args.max_experiments)]
    if not exps:
        raise RuntimeError("Nenhum experimento selecionado para benchmark.")
    print(f"Experimentos selecionados: {len(exps)}")

    X_train, Y_train, D_train, C_train, X_val, Y_val, D_val, C_val, df_split = apply_split(
        exps=exps,
        strategy="per_experiment",
        val_split=float(args.val_split),
        seed=int(args.seed),
        within_exp_shuffle=False,
    )
    X_train, Y_train, D_train, C_train, _ = cap_train_samples_per_experiment(
        X_train, Y_train, D_train, C_train, df_split, int(args.max_samples_per_exp)
    )

    Dn_train, Cn_train, Dn_val, Cn_val, norm_params = normalize_conditions(
        D_train, C_train, D_val, C_val
    )

    X_train = X_train.astype(np.float32, copy=False)
    Y_train = Y_train.astype(np.float32, copy=False)
    X_val = X_val.astype(np.float32, copy=False)
    Y_val = Y_val.astype(np.float32, copy=False)
    Dn_train = Dn_train.astype(np.float32, copy=False)
    Cn_train = Cn_train.astype(np.float32, copy=False)
    Dn_val = Dn_val.astype(np.float32, copy=False)
    Cn_val = Cn_val.astype(np.float32, copy=False)

    print(
        f"Split pronto | train={len(X_train):,} | val={len(X_val):,} | "
        f"norm D=[{norm_params['D_min']:.3f},{norm_params['D_max']:.3f}] "
        f"C=[{norm_params['C_min']:.1f},{norm_params['C_max']:.1f}]"
    )

    cfg = _default_cfg()
    rows: List[TrialResult] = []
    total_trials = len(batch_sizes) * len(pipelines)
    t_all = time.perf_counter()
    tid = 0
    for pipe in pipelines:
        for bs in batch_sizes:
            tid += 1
            print("\n" + "=" * 88)
            print(f"Trial {tid}/{total_trials} | pipeline={pipe} | batch_size={bs}")
            print("=" * 88)
            row = _run_trial(
                pipeline=pipe,
                batch_size=int(bs),
                cfg=cfg,
                X_train=X_train,
                Y_train=Y_train,
                Dn_train=Dn_train,
                Cn_train=Cn_train,
                X_val=X_val,
                Y_val=Y_val,
                Dn_val=Dn_val,
                Cn_val=Cn_val,
                df_split=df_split,
                epochs=int(args.epochs),
                warmup_epochs=int(args.warmup_epochs),
                seed=int(args.seed),
                keras_verbose=int(args.keras_verbose),
                use_cache=bool(args.tfdata_cache),
                n_eval_samples=int(args.n_eval_samples),
            )
            rows.append(row)
            print(
                f"samples/s={row.samples_per_s:.1f} | epoch_mean={row.epoch_time_mean_s:.3f}s | "
                f"dEVM={row.delta_evm_pp:+.3f}pp | dSNR={row.delta_snr_db:+.3f}dB | "
                f"var_ratio={row.var_ratio_pred_real:.3f}"
            )

    dt_all = time.perf_counter() - t_all
    df = pd.DataFrame([asdict(r) for r in rows])
    df = df.sort_values(["pipeline", "samples_per_s"], ascending=[True, False]).reset_index(drop=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_base).resolve() / f"batchsize_benchmark_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "results.csv", index=False)
    df.to_excel(out_dir / "results.xlsx", index=False)

    summary = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "duration_s": round(float(dt_all), 3),
        "args": vars(args),
        "regime_target": {"regime_id": args.regime_id, "distance_m": dist_m, "current_mA": curr_mA},
        "n_trials": int(total_trials),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "gpu_devices": [str(g) for g in gpus],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Recomendacao simples: menor batch com >=95% do throughput maximo por pipeline.
    rec_lines = []
    for pipe in pipelines:
        d = df[df["pipeline"] == pipe].sort_values("batch_size")
        if d.empty:
            continue
        thr_max = float(d["samples_per_s"].max())
        knee = d[d["samples_per_s"] >= 0.95 * thr_max].sort_values("batch_size").head(1)
        best = d.sort_values("samples_per_s", ascending=False).head(1)
        if not knee.empty and not best.empty:
            k = knee.iloc[0]
            b = best.iloc[0]
            rec_lines.append(
                f"[{pipe}] best_throughput: bs={int(b['batch_size'])}, samples/s={float(b['samples_per_s']):.1f}; "
                f"knee_95: bs={int(k['batch_size'])}, samples/s={float(k['samples_per_s']):.1f}"
            )

    (out_dir / "recommendation.txt").write_text("\n".join(rec_lines) + "\n", encoding="utf-8")

    print("\n" + "-" * 88)
    print(f"Benchmark finalizado em {dt_all:.1f}s")
    print(f"Resultados: {out_dir / 'results.csv'}")
    print(f"Resumo:     {out_dir / 'summary.json'}")
    print(f"Recomend.:  {out_dir / 'recommendation.txt'}")
    print("-" * 88)


if __name__ == "__main__":
    main()
