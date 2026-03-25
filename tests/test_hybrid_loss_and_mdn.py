import numpy as np
import tensorflow as tf

from src.models.cvae_sequence import create_seq_inference_model, load_seq_model
from src.models.losses import (
    CondPriorVAELoss,
    axis_moment_loss_tf,
    spectral_psd_loss_tf,
)
from src.models.cvae import build_cvae


def _base_loss_inputs():
    y_true = tf.constant(
        [[0.20, -0.10], [0.05, 0.15], [0.01, -0.04], [0.12, 0.07]],
        dtype=tf.float32,
    )
    out_params = tf.constant(
        [
            [0.08, -0.03, np.log(0.09), np.log(0.16)],
            [0.01, 0.05, np.log(0.04), np.log(0.25)],
            [0.03, -0.01, np.log(0.05), np.log(0.05)],
            [0.10, 0.09, np.log(0.08), np.log(0.07)],
        ],
        dtype=tf.float32,
    )
    z_mean_q = tf.zeros((4, 4), dtype=tf.float32)
    z_log_var_q = tf.zeros((4, 4), dtype=tf.float32)
    z_mean_p = tf.zeros((4, 4), dtype=tf.float32)
    z_log_var_p = tf.zeros((4, 4), dtype=tf.float32)
    x_center = tf.constant(
        [[0.10, -0.20], [-0.02, 0.12], [0.00, -0.01], [0.03, 0.02]],
        dtype=tf.float32,
    )
    return y_true, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p, x_center


def test_axis_and_spectral_losses_are_zero_for_matching_residuals():
    r = tf.constant(np.linspace(-1.0, 1.0, 256).reshape(128, 2), dtype=tf.float32)

    axis = axis_moment_loss_tf(r, r).numpy()
    psd = spectral_psd_loss_tf(r, r).numpy()

    assert abs(axis) < 1e-7
    assert abs(psd) < 1e-7


def test_condprior_loss_tracks_axis_and_psd_metrics():
    layer = CondPriorVAELoss(
        beta=1.0,
        free_bits=0.0,
        lambda_axis=0.1,
        lambda_psd=0.02,
    )
    _ = layer(_base_loss_inputs())
    metric_names = {metric.name for metric in layer.metrics}

    assert "axis_loss" in metric_names
    assert "psd_loss" in metric_names
    cfg = layer.get_config()
    assert cfg["lambda_axis"] == 0.1
    assert cfg["lambda_psd"] == 0.02


def test_seq_mdn_model_builds_saves_loads_and_predicts(tmp_path):
    cfg = {
        "arch_variant": "seq_bigru_residual",
        "layer_sizes": [16, 16],
        "latent_dim": 2,
        "beta": 0.003,
        "free_bits": 0.10,
        "lr": 3e-4,
        "dropout": 0.0,
        "kl_anneal_epochs": 4,
        "activation": "leaky_relu",
        "window_size": 7,
        "window_stride": 1,
        "window_pad_mode": "edge",
        "seq_hidden_size": 4,
        "seq_num_layers": 1,
        "seq_bidirectional": True,
        "decoder_distribution": "mdn",
        "mdn_components": 3,
        "lambda_axis": 0.05,
        "lambda_psd": 0.0,
    }
    model, _ = build_cvae(cfg)

    x = np.zeros((2, 7, 2), dtype=np.float32)
    d = np.zeros((2, 1), dtype=np.float32)
    c = np.zeros((2, 1), dtype=np.float32)
    y = np.zeros((2, 2), dtype=np.float32)

    y_out = model.predict([x, d, c, y], verbose=0)
    assert y_out.shape == (2, 2)
    assert model.get_layer("decoder").output_shape[-1] == 15

    save_path = tmp_path / "seq_mdn.keras"
    model.save(save_path)
    loaded = load_seq_model(save_path)

    inf_det = create_seq_inference_model(loaded, deterministic=True)
    inf_sto = create_seq_inference_model(loaded, deterministic=False)

    y_det = inf_det.predict([x, d, c], verbose=0)
    y_sto = inf_sto.predict([x, d, c], verbose=0)

    assert y_det.shape == (2, 2)
    assert y_sto.shape == (2, 2)
