import numpy as np
import pytest
import tensorflow as tf

from src.evaluation.report import decoder_sensitivity
from src.models.cvae_sequence import create_seq_inference_model, load_seq_model
from src.models.losses import (
    CondPriorDiffusionVAELoss,
    CondPriorVAELoss,
    axis_coverage_tail_loss_tf,
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


def test_axis_coverage_tail_loss_penalizes_miscalibrated_residuals():
    rng = np.random.default_rng(7)
    r_real = tf.constant(rng.normal(scale=0.35, size=(512, 2)), dtype=tf.float32)
    r_good = tf.constant(rng.normal(scale=0.35, size=(512, 2)), dtype=tf.float32)
    r_bad = tf.constant(0.15 * rng.normal(size=(512, 2)) + 0.35, dtype=tf.float32)

    good = axis_coverage_tail_loss_tf(r_real, r_good).numpy()
    bad = axis_coverage_tail_loss_tf(r_real, r_bad).numpy()

    assert np.isfinite(good)
    assert np.isfinite(bad)
    assert good < bad


def test_condprior_loss_tracks_axis_psd_and_coverage_metrics():
    layer = CondPriorVAELoss(
        beta=1.0,
        free_bits=0.0,
        lambda_axis=0.1,
        lambda_psd=0.02,
        lambda_coverage=0.05,
    )
    _ = layer(_base_loss_inputs())
    metric_names = {metric.name for metric in layer.metrics}

    assert "axis_loss" in metric_names
    assert "psd_loss" in metric_names
    assert "coverage_loss" in metric_names
    cfg = layer.get_config()
    assert cfg["lambda_axis"] == 0.1
    assert cfg["lambda_psd"] == 0.02
    assert cfg["lambda_coverage"] == 0.05
    assert cfg["coverage_levels"] == [0.5, 0.8, 0.95]
    assert cfg["tail_levels"] == [0.05, 0.95]


def test_condprior_diffusion_loss_tracks_core_metrics():
    layer = CondPriorDiffusionVAELoss(
        beta=1.0,
        free_bits=0.1,
        diffusion_steps=4,
        diffusion_beta_start=1e-4,
        diffusion_beta_end=1e-2,
    )
    x_center = tf.zeros((4, 2), dtype=tf.float32)
    residual_noisy = tf.ones((4, 2), dtype=tf.float32) * 0.1
    noise_target = tf.zeros((4, 2), dtype=tf.float32)
    sqrt_alpha_bar = tf.ones((4, 1), dtype=tf.float32) * 0.95
    sqrt_one_minus_alpha_bar = tf.ones((4, 1), dtype=tf.float32) * 0.2
    eps_pred = tf.zeros((4, 2), dtype=tf.float32)
    z_mean_q = tf.zeros((4, 2), dtype=tf.float32)
    z_log_var_q = tf.zeros((4, 2), dtype=tf.float32)
    z_mean_p = tf.zeros((4, 2), dtype=tf.float32)
    z_log_var_p = tf.zeros((4, 2), dtype=tf.float32)

    y_hat = layer(
        [
            x_center,
            residual_noisy,
            noise_target,
            sqrt_alpha_bar,
            sqrt_one_minus_alpha_bar,
            eps_pred,
            z_mean_q,
            z_log_var_q,
            z_mean_p,
            z_log_var_p,
        ]
    )
    assert y_hat.shape == (4, 2)
    metric_names = {metric.name for metric in layer.metrics}
    assert metric_names == {"recon_loss", "kl_loss"}
    cfg = layer.get_config()
    assert cfg["diffusion_steps"] == 4


def test_condprior_diffusion_loss_supports_vpred_without_kl():
    layer = CondPriorDiffusionVAELoss(
        beta=1.0,
        free_bits=0.1,
        diffusion_steps=4,
        diffusion_beta_start=1e-4,
        diffusion_beta_end=1e-2,
        diffusion_target="v",
        use_kl=False,
    )
    x_center = tf.zeros((4, 2), dtype=tf.float32)
    residual_noisy = tf.ones((4, 2), dtype=tf.float32) * 0.1
    target_tensor = tf.zeros((4, 2), dtype=tf.float32)
    sqrt_alpha_bar = tf.ones((4, 1), dtype=tf.float32) * 0.95
    sqrt_one_minus_alpha_bar = tf.ones((4, 1), dtype=tf.float32) * 0.2
    target_pred = tf.zeros((4, 2), dtype=tf.float32)
    z_mean_q = tf.zeros((4, 2), dtype=tf.float32)
    z_log_var_q = tf.zeros((4, 2), dtype=tf.float32)
    z_mean_p = tf.zeros((4, 2), dtype=tf.float32)
    z_log_var_p = tf.zeros((4, 2), dtype=tf.float32)

    y_hat = layer(
        [
            x_center,
            residual_noisy,
            target_tensor,
            sqrt_alpha_bar,
            sqrt_one_minus_alpha_bar,
            target_pred,
            z_mean_q,
            z_log_var_q,
            z_mean_p,
            z_log_var_p,
        ]
    )
    assert y_hat.shape == (4, 2)
    cfg = layer.get_config()
    assert cfg["diffusion_target"] == "v"
    assert cfg["use_kl"] is False


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


def test_seq_diffusion_model_builds_saves_loads_and_predicts(tmp_path):
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
        "decoder_distribution": "diffusion",
        "diffusion_steps": 4,
        "diffusion_hidden_size": 16,
        "lambda_mmd": 0.0,
        "lambda_axis": 0.0,
        "lambda_psd": 0.0,
        "lambda_coverage": 0.0,
    }
    model, _ = build_cvae(cfg)

    x = np.zeros((2, 7, 2), dtype=np.float32)
    d = np.zeros((2, 1), dtype=np.float32)
    c = np.zeros((2, 1), dtype=np.float32)
    y = np.zeros((2, 2), dtype=np.float32)

    y_out = model.predict([x, d, c, y], verbose=0)
    assert y_out.shape == (2, 2)
    decoder = model.get_layer("decoder")
    assert decoder.output_shape[-1] == 2
    assert len(decoder.inputs) == 6

    save_path = tmp_path / "seq_diffusion.keras"
    model.save(save_path)
    loaded = load_seq_model(save_path)

    inf_det = create_seq_inference_model(loaded, deterministic=True)
    inf_sto = create_seq_inference_model(loaded, deterministic=False)

    y_det = inf_det.predict([x, d, c], verbose=0)
    y_sto = inf_sto.predict([x, d, c], verbose=0)

    assert y_det.shape == (2, 2)
    assert y_sto.shape == (2, 2)


def test_seq_direct_diffusion_model_builds_saves_loads_and_predicts(tmp_path):
    cfg = {
        "arch_variant": "seq_bigru_residual",
        "layer_sizes": [16, 16],
        "latent_dim": 4,
        "beta": 0.0,
        "free_bits": 0.0,
        "lr": 2e-4,
        "dropout": 0.0,
        "kl_anneal_epochs": 1,
        "activation": "leaky_relu",
        "window_size": 7,
        "window_stride": 1,
        "window_pad_mode": "edge",
        "seq_hidden_size": 4,
        "seq_num_layers": 1,
        "seq_bidirectional": True,
        "decoder_distribution": "diffusion_direct",
        "diffusion_target": "v",
        "diffusion_steps": 4,
        "diffusion_hidden_size": 16,
        "lambda_mmd": 0.0,
        "lambda_axis": 0.0,
        "lambda_psd": 0.0,
        "lambda_coverage": 0.0,
    }
    model, _ = build_cvae(cfg)

    with pytest.raises(ValueError):
        model.get_layer("encoder")

    x = np.zeros((2, 7, 2), dtype=np.float32)
    d = np.zeros((2, 1), dtype=np.float32)
    c = np.zeros((2, 1), dtype=np.float32)
    y = np.zeros((2, 2), dtype=np.float32)

    y_out = model.predict([x, d, c, y], verbose=0)
    assert y_out.shape == (2, 2)
    decoder = model.get_layer("decoder")
    assert decoder.output_shape[-1] == 2
    assert len(decoder.inputs) == 6
    assert getattr(decoder, "_diffusion_direct") is True
    assert getattr(decoder, "_diffusion_target") == "v"

    save_path = tmp_path / "seq_diffusion_direct.keras"
    model.save(save_path)
    loaded = load_seq_model(save_path)

    inf_det = create_seq_inference_model(loaded, deterministic=True)
    inf_sto = create_seq_inference_model(loaded, deterministic=False)

    y_det = inf_det.predict([x, d, c], verbose=0)
    y_sto = inf_sto.predict([x, d, c], verbose=0)

    assert y_det.shape == (2, 2)
    assert y_sto.shape == (2, 2)


def test_decoder_sensitivity_is_finite_for_seq_gaussian_mdn_and_diffusion():
    base_cfg = {
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
    }
    x = np.zeros((4, 7, 2), dtype=np.float32)
    d = np.zeros((4, 1), dtype=np.float32)
    c = np.zeros((4, 1), dtype=np.float32)

    for decoder_distribution, mdn_components, diffusion_steps, diffusion_target in (
        ("gaussian", 1, None, None),
        ("mdn", 3, None, None),
        ("diffusion", 1, 4, None),
        ("diffusion_direct", 1, 4, "v"),
    ):
        cfg = dict(base_cfg)
        cfg["decoder_distribution"] = decoder_distribution
        cfg["mdn_components"] = mdn_components
        if diffusion_steps is not None:
            cfg["diffusion_steps"] = diffusion_steps
            cfg["diffusion_hidden_size"] = 16
            if diffusion_target is not None:
                cfg["diffusion_target"] = diffusion_target
                cfg["beta"] = 0.0
                cfg["free_bits"] = 0.0
            cfg["lambda_mmd"] = 0.0
            cfg["lambda_axis"] = 0.0
            cfg["lambda_psd"] = 0.0
            cfg["lambda_coverage"] = 0.0
        model, _ = build_cvae(cfg)
        sens = decoder_sensitivity(
            model.get_layer("prior_net"),
            model.get_layer("decoder"),
            x,
            d,
            c,
            n_mc_z=4,
            batch_size=2,
            arch_variant="seq_bigru_residual",
        )
        assert sens["status"] == "ok"
        assert np.isfinite(sens["decoder_output_variance_mean"])
        assert np.isfinite(sens["decoder_output_rms_std"])


def test_decoder_sensitivity_flags_unsupported_decoder_interface():
    class _FakePrior:
        def predict(self, inputs, batch_size=4096, verbose=0):
            n = len(inputs[0])
            return np.zeros((n, 2), dtype=np.float32), np.zeros((n, 2), dtype=np.float32)

    class _FakeDecoder:
        inputs = [object(), object(), object()]

    x = np.zeros((2, 2), dtype=np.float32)
    d = np.zeros((2, 1), dtype=np.float32)
    c = np.zeros((2, 1), dtype=np.float32)
    sens = decoder_sensitivity(_FakePrior(), _FakeDecoder(), x, d, c)

    assert sens["status"] == "unsupported_decoder_interface"
    assert np.isnan(sens["decoder_output_variance_mean"])
    assert np.isnan(sens["decoder_output_rms_std"])
