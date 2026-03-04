# -*- coding: utf-8 -*-
"""
Deterministic MLP baseline — x→y regression for VLC channel.

Serves as a "deterministic + heteroscedastic noise" reference
to compare against the cVAE digital twin.  Does NOT reuse any
VAE code.

Commit 3N.
"""

import time
from typing import Dict, Optional

import numpy as np


def _build_mlp(input_dim: int, output_dim: int, hidden: list, dropout: float = 0.0):
    """Build a small Keras sequential MLP (imported lazily)."""
    import tensorflow as tf

    model = tf.keras.Sequential(name="baseline_mlp")
    for i, units in enumerate(hidden):
        model.add(tf.keras.layers.Dense(
            units, activation="relu",
            kernel_initializer="he_normal",
            name=f"dense_{i}",
        ))
        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout, name=f"drop_{i}"))
    model.add(tf.keras.layers.Dense(output_dim, activation="linear", name="output"))

    # Build explicitly so summary works before fit
    model.build(input_shape=(None, input_dim))
    return model


def run_deterministic_baseline(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    config: Optional[Dict] = None,
) -> Dict:
    """
    Train a small MLP (MSE loss) and evaluate EVM/SNR.

    Parameters
    ----------
    X_train, Y_train : ndarray (N, 2)
        Training I/Q pairs.
    X_val, Y_val : ndarray (N, 2)
        Validation I/Q pairs.
    config : dict, optional
        Keys (all optional, sensible defaults):
        - hidden: list[int], MLP hidden layer sizes (default [128, 64])
        - dropout: float (default 0.0)
        - epochs: int (default 50)
        - batch_size: int (default 1024)
        - learning_rate: float (default 1e-3)
        - verbose: int, keras verbose (default 0)
        - return_predictions: bool, if True include '_Y_pred' key (default False)

    Returns
    -------
    dict
        Metrics and metadata:
        - evm_real_%, evm_pred_%, delta_evm_%
        - snr_real_db, snr_pred_db, delta_snr_db
        - residual_mean_l2, residual_std_l2
        - best_val_loss, epochs_run, train_time_s
        - _Y_pred (ndarray, only when return_predictions=True)
    """
    import tensorflow as tf
    from src.evaluation.metrics import calculate_evm, calculate_snr

    cfg = config or {}
    hidden = cfg.get("hidden", [128, 64])
    dropout = float(cfg.get("dropout", 0.0))
    epochs = int(cfg.get("epochs", 50))
    batch_size = int(cfg.get("batch_size", 1024))
    lr = float(cfg.get("learning_rate", 1e-3))
    _return_preds = bool(cfg.get("return_predictions", False))
    verbose = int(cfg.get("verbose", 0))

    # ---- Build & compile ----
    model = _build_mlp(
        input_dim=X_train.shape[1],
        output_dim=Y_train.shape[1],
        hidden=hidden,
        dropout=dropout,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
    )

    # ---- Train ----
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True,
    )

    t0 = time.time()
    hist = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose,
        shuffle=True,
    )
    train_time = time.time() - t0

    val_losses = hist.history.get("val_loss", [])
    best_val = float(min(val_losses)) if val_losses else float("nan")
    epochs_run = len(val_losses)

    # ---- Predict ----
    Y_pred = model.predict(X_val, batch_size=batch_size, verbose=0)

    # ---- Metrics ----
    evm_real, _ = calculate_evm(X_val, Y_val)
    evm_pred, _ = calculate_evm(X_val, Y_pred)
    snr_real = calculate_snr(X_val, Y_val)
    snr_pred = calculate_snr(X_val, Y_pred)

    residual_real = Y_val - X_val
    residual_pred = Y_pred - X_val
    diff = residual_pred - residual_real
    res_mean_l2 = float(np.mean(np.sqrt(np.sum(diff ** 2, axis=1))))
    res_std_l2 = float(np.std(np.sqrt(np.sum(diff ** 2, axis=1))))

    # ---- Build result ----
    out = {
        "evm_real_%": float(evm_real),
        "evm_pred_%": float(evm_pred),
        "delta_evm_%": float(evm_pred - evm_real),
        "snr_real_db": float(snr_real),
        "snr_pred_db": float(snr_pred),
        "delta_snr_db": float(snr_pred - snr_real),
        "residual_mean_l2": res_mean_l2,
        "residual_std_l2": res_std_l2,
        "best_val_loss": best_val,
        "epochs_run": epochs_run,
        "train_time_s": round(train_time, 2),
    }
    if _return_preds:
        out["_Y_pred"] = Y_pred  # large array — caller should pop()

    # ---- Cleanup ----
    del model
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass

    return out
