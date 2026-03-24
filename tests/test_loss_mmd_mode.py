import numpy as np
import tensorflow as tf

from src.models import losses as losses_mod


def _base_inputs():
    y_true = tf.constant(
        [[0.20, -0.10], [0.05, 0.15]],
        dtype=tf.float32,
    )
    out_params = tf.constant(
        [
            [0.08, -0.03, np.log(0.09), np.log(0.16)],
            [0.01, 0.05, np.log(0.04), np.log(0.25)],
        ],
        dtype=tf.float32,
    )
    z_mean_q = tf.zeros((2, 4), dtype=tf.float32)
    z_log_var_q = tf.zeros((2, 4), dtype=tf.float32)
    z_mean_p = tf.zeros((2, 4), dtype=tf.float32)
    z_log_var_p = tf.zeros((2, 4), dtype=tf.float32)
    x_center = tf.constant(
        [[0.10, -0.20], [-0.02, 0.12]],
        dtype=tf.float32,
    )
    return y_true, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p, x_center


def test_condprior_vae_loss_mean_residual_mmd_uses_mean(monkeypatch):
    captured = {}

    def fake_mmd2_tf(r_real, r_gen, n_sub=512, bandwidth=None):
        captured["r_real"] = r_real.numpy()
        captured["r_gen"] = r_gen.numpy()
        return tf.constant(0.0, dtype=tf.float32)

    monkeypatch.setattr(losses_mod, "mmd2_tf", fake_mmd2_tf)

    layer = losses_mod.CondPriorVAELoss(
        beta=1.0,
        free_bits=0.0,
        lambda_mmd=1.0,
        mmd_mode="mean_residual",
    )
    inputs = _base_inputs()
    y_true, out_params, *_rest, x_center = inputs

    _ = layer(inputs)

    expected_real = y_true.numpy() - x_center.numpy()
    expected_gen = out_params.numpy()[:, :2] - x_center.numpy()

    assert np.allclose(captured["r_real"], expected_real)
    assert np.allclose(captured["r_gen"], expected_gen)
    assert layer.get_config()["mmd_mode"] == "mean_residual"


def test_condprior_vae_loss_sampled_residual_mmd_uses_sample(monkeypatch):
    captured = {}

    def fake_mmd2_tf(r_real, r_gen, n_sub=512, bandwidth=None):
        captured["r_gen"] = r_gen.numpy()
        return tf.constant(0.0, dtype=tf.float32)

    monkeypatch.setattr(losses_mod, "mmd2_tf", fake_mmd2_tf)
    tf.random.set_seed(20260324)

    layer = losses_mod.CondPriorVAELoss(
        beta=1.0,
        free_bits=0.0,
        lambda_mmd=1.0,
        mmd_mode="sampled_residual",
    )
    inputs = _base_inputs()
    _y_true, out_params, *_rest, x_center = inputs

    _ = layer(inputs)

    expected_mean = out_params.numpy()[:, :2] - x_center.numpy()

    assert captured["r_gen"].shape == expected_mean.shape
    assert not np.allclose(captured["r_gen"], expected_mean)
    assert layer.get_config()["mmd_mode"] == "sampled_residual"


def test_condprior_delta_loss_sampled_residual_mmd_uses_sample(monkeypatch):
    captured = {}

    def fake_mmd2_tf(r_real, r_gen, n_sub=512, bandwidth=None):
        captured["r_gen"] = r_gen.numpy()
        return tf.constant(0.0, dtype=tf.float32)

    monkeypatch.setattr(losses_mod, "mmd2_tf", fake_mmd2_tf)
    tf.random.set_seed(20260324)

    layer = losses_mod.CondPriorDeltaVAELoss(
        beta=1.0,
        free_bits=0.0,
        lambda_mmd=1.0,
        mmd_mode="sampled_residual",
    )
    y_true, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p, x_true = _base_inputs()

    _ = layer([y_true, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p, x_true])

    expected_delta_mean = out_params.numpy()[:, :2]

    assert captured["r_gen"].shape == expected_delta_mean.shape
    assert not np.allclose(captured["r_gen"], expected_delta_mean)
    assert layer.get_config()["mmd_mode"] == "sampled_residual"


def test_resolve_mmd_mode_rejects_unknown_mode():
    try:
        losses_mod._resolve_mmd_mode("unknown_mode")
    except ValueError as exc:
        assert "mmd_mode must be one of" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid mmd_mode")
