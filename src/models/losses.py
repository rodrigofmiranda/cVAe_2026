# -*- coding: utf-8 -*-
"""
src/models/losses.py — Loss functions for the heteroscedastic cVAE.

Extracted from ``cvae_components.py`` (refactor step 3).

The formulas are **identical** to the monolith — no algorithmic changes.

Public API
----------
reconstruction_loss      Heteroscedastic Gaussian NLL
kl_divergence            KL(q ‖ p) per sample
kl_to_standard_normal    KL(q ‖ N(0, I)) per sample
kl_with_freebits         Free-bits thresholded KL
compute_total_loss        recon + β · min(kl, cap)
CondPriorVAELoss         Keras layer (for training graph)
CondPriorDeltaVAELoss    Explicit residual-target Keras layer
StdNormalHeteroscedasticVAELoss
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.config.defaults import (
    DECODER_LOGVAR_CLAMP_HI,
    DECODER_LOGVAR_CLAMP_LO,
)


# ======================================================================
# Standalone functional forms (TF tensors — usable in eager or graph)
# ======================================================================

def reconstruction_loss(
    y_true: tf.Tensor,
    y_mean: tf.Tensor,
    y_log_var: tf.Tensor,
) -> tf.Tensor:
    """Heteroscedastic Gaussian NLL (mean over batch).

    .. math::
        \\mathrm{NLL} = \\frac{1}{2}\\sum_d
        \\bigl(\\log\\sigma^2_d + (y_d - \\mu_d)^2 / \\sigma^2_d
        + \\log 2\\pi \\bigr)

    Parameters
    ----------
    y_true : (N, 2)
    y_mean : (N, 2)
    y_log_var : (N, 2)  — clipped externally if needed.

    Returns
    -------
    scalar Tensor — mean NLL over batch.
    """
    y_log_var = tf.clip_by_value(
        y_log_var, DECODER_LOGVAR_CLAMP_LO, DECODER_LOGVAR_CLAMP_HI
    )
    y_var = tf.exp(y_log_var) + 1e-6
    nll = 0.5 * tf.reduce_sum(
        y_log_var + tf.square(y_true - y_mean) / y_var + tf.math.log(2.0 * np.pi),
        axis=-1,
    )
    return tf.reduce_mean(nll)


def kl_divergence(
    z_mean_q: tf.Tensor,
    z_log_var_q: tf.Tensor,
    z_mean_p: tf.Tensor,
    z_log_var_p: tf.Tensor,
) -> tf.Tensor:
    """KL(q ‖ p) per sample, where q and p are diagonal Gaussians.

    Returns
    -------
    kl_per_sample : (N,)
    """
    vq = tf.exp(tf.clip_by_value(z_log_var_q, -20.0, 20.0))
    vp = tf.exp(tf.clip_by_value(z_log_var_p, -20.0, 20.0))
    kl_dim = 0.5 * (
        tf.math.log(vp + 1e-12) - tf.math.log(vq + 1e-12)
        + (vq + tf.square(z_mean_q - z_mean_p)) / (vp + 1e-12)
        - 1.0
    )
    return tf.reduce_sum(kl_dim, axis=-1)


def kl_with_freebits(
    kl_per_sample: tf.Tensor,
    free_bits: float = 0.0,
) -> tf.Tensor:
    """Apply free-bits thresholding to per-sample KL.

    Returns
    -------
    kl_fb : (N,)  — ``max(kl - free_bits, 0)``
    """
    fb = tf.cast(free_bits, kl_per_sample.dtype)
    return tf.maximum(kl_per_sample - fb, 0.0)


def kl_to_standard_normal(
    z_mean_q: tf.Tensor,
    z_log_var_q: tf.Tensor,
) -> tf.Tensor:
    """KL(q ‖ N(0, I)) per sample for a diagonal Gaussian q.

    Returns
    -------
    kl_per_sample : (N,)
    """
    z_log_var_q = tf.clip_by_value(z_log_var_q, -20.0, 20.0)
    kl_dim = 0.5 * (
        tf.exp(z_log_var_q) + tf.square(z_mean_q) - 1.0 - z_log_var_q
    )
    return tf.reduce_sum(kl_dim, axis=-1)


def compute_total_loss(
    recon: tf.Tensor,
    kl: tf.Tensor,
    beta: float | tf.Tensor,
    kl_cap: float = 200.0,
) -> tf.Tensor:
    """Total ELBO loss = recon + β · min(kl, cap).

    Parameters
    ----------
    recon : scalar — mean reconstruction NLL.
    kl    : scalar — mean KL.
    beta  : current β weight.
    kl_cap : safety clamp.

    Returns
    -------
    scalar Tensor.
    """
    return recon + beta * tf.minimum(kl, kl_cap)


# ======================================================================
# Mini-batch MMD² (TF ops — safe inside training graph)
# ======================================================================

def mmd2_tf(
    r_real: tf.Tensor,
    r_gen: tf.Tensor,
    n_sub: int = 512,
    bandwidth: float | None = None,
) -> tf.Tensor:
    """Unbiased mini-batch MMD² with RBF kernel, in pure TF ops.

    Parameters
    ----------
    r_real : (N, 2) — real channel residuals  (Y_real − X)
    r_gen  : (N, 2) — model residuals         (Y_pred − X)
    n_sub  : number of samples to sub-sample per call (default 512)
    bandwidth : RBF bandwidth σ². None → median heuristic computed inline.

    Returns
    -------
    scalar Tensor — unbiased MMD²
    """
    N = tf.shape(r_real)[0]
    n = tf.minimum(n_sub, N)

    # Independent sub-samples from real and generated pools
    idx_r = tf.random.shuffle(tf.range(N))[:n]
    idx_g = tf.random.shuffle(tf.range(N))[:n]
    x = tf.cast(tf.gather(r_real, idx_r), tf.float32)
    y = tf.cast(tf.gather(r_gen,  idx_g), tf.float32)

    def _sq_dists(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        aa = tf.reduce_sum(tf.square(a), axis=1, keepdims=True)
        bb = tf.reduce_sum(tf.square(b), axis=1, keepdims=True)
        ab = tf.matmul(a, b, transpose_b=True)
        return tf.maximum(aa + tf.transpose(bb) - 2.0 * ab, 0.0)

    if bandwidth is None:
        # Median heuristic on cross-set distances
        d2_xy = _sq_dists(x, y)
        flat = tf.reshape(d2_xy, [-1])
        mid = tf.cast(tf.shape(flat)[0] // 2, tf.int32)
        bw = tf.maximum(tf.sort(flat)[mid], 1e-3)
    else:
        bw = tf.constant(float(bandwidth), dtype=tf.float32)

    def _K(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return tf.exp(-_sq_dists(a, b) / (2.0 * bw))

    Kxx = _K(x, x)
    Kyy = _K(y, y)
    Kxy = _K(x, y)

    nf = tf.cast(n, tf.float32)
    mask = 1.0 - tf.eye(n)
    term_xx = tf.reduce_sum(Kxx * mask) / (nf * (nf - 1.0))
    term_yy = tf.reduce_sum(Kyy * mask) / (nf * (nf - 1.0))
    term_xy = tf.reduce_sum(Kxy) / (nf * nf)
    return term_xx + term_yy - 2.0 * term_xy


def _resolve_mmd_mode(mode: str | None) -> str:
    """Return the canonical MMD residual-matching mode."""
    mode_norm = str(mode or "mean_residual").strip().lower()
    aliases = {
        "mean": "mean_residual",
        "mean_residual": "mean_residual",
        "sample": "sampled_residual",
        "sampled": "sampled_residual",
        "sampled_residual": "sampled_residual",
    }
    if mode_norm not in aliases:
        raise ValueError(
            "mmd_mode must be one of {'mean_residual', 'sampled_residual'}; "
            f"got {mode!r}"
        )
    return aliases[mode_norm]


def _sample_heteroscedastic(mean: tf.Tensor, log_var: tf.Tensor) -> tf.Tensor:
    """Draw a differentiable sample from N(mean, diag(exp(log_var)))."""
    log_var = tf.clip_by_value(
        log_var, DECODER_LOGVAR_CLAMP_LO, DECODER_LOGVAR_CLAMP_HI
    )
    std = tf.exp(0.5 * log_var)
    eps = tf.random.normal(tf.shape(mean), dtype=mean.dtype)
    return mean + std * eps


def _resolve_decoder_distribution(mode: str | None) -> str:
    """Return the canonical decoder distribution family."""
    mode_norm = str(mode or "gaussian").strip().lower()
    aliases = {
        "gaussian": "gaussian",
        "heteroscedastic": "gaussian",
        "mdn": "mdn",
        "mixture": "mdn",
        "mixture_density": "mdn",
        "flow": "flow",
        "normalizing_flow": "flow",
        "conditional_flow": "flow",
    }
    if mode_norm not in aliases:
        raise ValueError(
            "decoder_distribution must be one of {'gaussian', 'mdn', 'flow'}; "
            f"got {mode!r}"
        )
    return aliases[mode_norm]


def _resolve_flow_family(mode: str | None) -> str:
    """Return the canonical flow decoder family."""
    mode_norm = str(mode or "coupling_2d").strip().lower()
    aliases = {
        "coupling": "coupling_2d",
        "coupling_2d": "coupling_2d",
        "affine_coupling": "coupling_2d",
        "affine_coupling_2d": "coupling_2d",
        "joint_affine_coupling": "coupling_2d",
        "sinh_arcsinh": "sinh_arcsinh",
        "sinh_arcsinh_flow": "sinh_arcsinh",
        "sas": "sinh_arcsinh",
        "axis_sas": "sinh_arcsinh",
        "legacy_sas": "sinh_arcsinh",
        "spline": "spline_2d",
        "spline_2d": "spline_2d",
        "rq_spline": "spline_2d",
        "rqs": "spline_2d",
        "neural_spline": "spline_2d",
        "conditional_neural_spline": "spline_2d",
    }
    if mode_norm not in aliases:
        raise ValueError(
            "flow_family must be one of {'coupling_2d', 'sinh_arcsinh', 'spline_2d'}; "
            f"got {mode!r}"
        )
    return aliases[mode_norm]


def _unpack_gaussian_params(out_params: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Split Gaussian decoder params into mean/log-variance tensors."""
    return out_params[:, :2], out_params[:, 2:]


def _unpack_mdn_params(
    out_params: tf.Tensor,
    mdn_components: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Split MDN output into logits, component means, and component log-vars."""
    k = int(mdn_components)
    if k <= 0:
        raise ValueError(f"mdn_components must be > 0; got {mdn_components!r}")
    logits = out_params[:, :k]
    mean_flat = out_params[:, k : k + 2 * k]
    log_var_flat = out_params[:, k + 2 * k : k + 4 * k]
    comp_mean = tf.reshape(mean_flat, (-1, k, 2))
    comp_log_var = tf.reshape(log_var_flat, (-1, k, 2))
    return logits, comp_mean, comp_log_var


def mdn_reconstruction_loss(
    y_true: tf.Tensor,
    logits: tf.Tensor,
    comp_mean: tf.Tensor,
    comp_log_var: tf.Tensor,
) -> tf.Tensor:
    """Mixture-density negative log-likelihood averaged over the batch."""
    comp_log_var = tf.clip_by_value(
        comp_log_var, DECODER_LOGVAR_CLAMP_LO, DECODER_LOGVAR_CLAMP_HI
    )
    y_true = tf.expand_dims(y_true, axis=1)  # (N,1,2)
    inv_var = tf.exp(-comp_log_var)
    quad = tf.reduce_sum(tf.square(y_true - comp_mean) * inv_var, axis=-1)
    log_det = tf.reduce_sum(comp_log_var, axis=-1)
    log_norm = -0.5 * (
        log_det + quad + tf.cast(2.0 * np.log(2.0 * np.pi), tf.float32)
    )
    log_pi = tf.nn.log_softmax(logits, axis=-1)
    log_prob = tf.reduce_logsumexp(log_pi + log_norm, axis=-1)
    return -tf.reduce_mean(log_prob)


def _mdn_expected_mean(logits: tf.Tensor, comp_mean: tf.Tensor) -> tf.Tensor:
    """Return E[y] under a diagonal-Gaussian mixture."""
    probs = tf.nn.softmax(logits, axis=-1)
    return tf.reduce_sum(tf.expand_dims(probs, axis=-1) * comp_mean, axis=1)


def _sample_mdn(
    logits: tf.Tensor,
    comp_mean: tf.Tensor,
    comp_log_var: tf.Tensor,
) -> tf.Tensor:
    """Draw one sample from a diagonal-Gaussian mixture."""
    batch = tf.shape(logits)[0]
    idx = tf.random.categorical(logits, 1)
    idx = tf.cast(tf.squeeze(idx, axis=1), tf.int32)
    gather_idx = tf.stack([tf.range(batch, dtype=tf.int32), idx], axis=1)
    mean_sel = tf.gather_nd(comp_mean, gather_idx)
    log_var_sel = tf.gather_nd(comp_log_var, gather_idx)
    return _sample_heteroscedastic(mean_sel, log_var_sel)


FLOW_LOG_SCALE_CLAMP_LO = -8.0
FLOW_LOG_SCALE_CLAMP_HI = 3.0
FLOW_SKEW_CLAMP = 3.0
FLOW_LOG_TAIL_CLAMP_LO = np.log(0.5)
FLOW_LOG_TAIL_CLAMP_HI = np.log(2.0)
FLOW_COUPLING_SHIFT_PARAM_CLAMP = 3.0
FLOW_COUPLING_SHIFT_EFFECT_CLAMP = 6.0
FLOW_COUPLING_LOG_SCALE_PARAM_CLAMP = 2.0
FLOW_COUPLING_LOG_SCALE_EFFECT_CLAMP = 2.5
FLOW_SPLINE_DEFAULT_BOUND = 3.5
FLOW_SPLINE_MIN_BIN_WIDTH = 1e-2
FLOW_SPLINE_MIN_BIN_HEIGHT = 1e-2
FLOW_SPLINE_MIN_DERIVATIVE = 1e-2


def _flow_layer_family_from_names(layer_names) -> str:
    """Infer the flow family from decoder layer names."""
    names = {str(name) for name in (layer_names or [])}
    if any("flow_spline_" in name for name in names):
        return "spline_2d"
    if any("flow_coupling_" in name for name in names):
        return "coupling_2d"
    if "flow_skew" in names or "flow_log_tail" in names:
        return "sinh_arcsinh"
    return "coupling_2d"


def _unpack_flow_params(
    out_params: tf.Tensor,
    *,
    flow_family: str = "coupling_2d",
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Split conditional flow output into family-specific parameter blocks."""
    family = _resolve_flow_family(flow_family)
    y_loc = out_params[:, :2]
    if family == "spline_2d":
        total_dim = out_params.shape[-1]
        if total_dim is None:
            raise ValueError("spline_2d flow requires a static decoder output dimension")
        if total_dim < 12 or total_dim % 6 != 0:
            raise ValueError(
                "spline_2d flow output dimension must be divisible by 6 and >= 12; "
                f"got {total_dim}"
            )
        n_bins = total_dim // 6
        widths_end = 2 + 2 * n_bins
        heights_end = widths_end + 2 * n_bins
        p1 = out_params[:, 2:widths_end]
        p2 = out_params[:, widths_end:heights_end]
        p3 = out_params[:, heights_end:]
    else:
        p1 = out_params[:, 2:4]
        p2 = out_params[:, 4:6]
        p3 = out_params[:, 6:8]
    return y_loc, p1, p2, p3


def _clamp_sas_flow_params(
    log_scale: tf.Tensor,
    skew: tf.Tensor,
    log_tail: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Clamp legacy sinh-arcsinh flow parameters to a stable band."""
    log_scale = tf.clip_by_value(
        log_scale, FLOW_LOG_SCALE_CLAMP_LO, FLOW_LOG_SCALE_CLAMP_HI
    )
    skew = tf.clip_by_value(skew, -FLOW_SKEW_CLAMP, FLOW_SKEW_CLAMP)
    log_tail = tf.clip_by_value(
        log_tail, FLOW_LOG_TAIL_CLAMP_LO, FLOW_LOG_TAIL_CLAMP_HI
    )
    return log_scale, skew, log_tail


def _flow_forward_sinh_arcsinh(
    eps: tf.Tensor,
    *,
    y_loc: tf.Tensor,
    log_scale: tf.Tensor,
    skew: tf.Tensor,
    log_tail: tf.Tensor,
) -> tf.Tensor:
    """Transform base Gaussian samples into observations via per-axis SAS flow."""
    log_scale, skew, log_tail = _clamp_sas_flow_params(log_scale, skew, log_tail)
    scale = tf.exp(log_scale)
    tail = tf.exp(log_tail)
    warped = tf.sinh((tf.math.asinh(eps) + skew) / tail)
    return y_loc + scale * warped


def _flow_inverse_sinh_arcsinh(
    y: tf.Tensor,
    *,
    y_loc: tf.Tensor,
    log_scale: tf.Tensor,
    skew: tf.Tensor,
    log_tail: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Map observations back to the standard-normal base for SAS flow."""
    log_scale, skew, log_tail = _clamp_sas_flow_params(log_scale, skew, log_tail)
    scale = tf.exp(log_scale)
    tail = tf.exp(log_tail)
    x = (y - y_loc) / (scale + 1e-6)
    base = tf.sinh(tail * tf.math.asinh(x) - skew)
    log_abs_det = (
        log_tail
        + tf.math.log(tf.cosh(tail * tf.math.asinh(x) - skew) + 1e-6)
        - 0.5 * tf.math.log1p(tf.square(x))
        - log_scale
    )
    return base, tf.reduce_sum(log_abs_det, axis=-1)


def _clamp_coupling_flow_params(
    base_log_scale: tf.Tensor,
    coupling_shift: tf.Tensor,
    coupling_log_scale: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Clamp 2-D affine-coupling flow parameters."""
    base_log_scale = tf.clip_by_value(
        base_log_scale, FLOW_LOG_SCALE_CLAMP_LO, FLOW_LOG_SCALE_CLAMP_HI
    )
    coupling_shift = tf.clip_by_value(
        coupling_shift,
        -FLOW_COUPLING_SHIFT_PARAM_CLAMP,
        FLOW_COUPLING_SHIFT_PARAM_CLAMP,
    )
    coupling_log_scale = tf.clip_by_value(
        coupling_log_scale,
        -FLOW_COUPLING_LOG_SCALE_PARAM_CLAMP,
        FLOW_COUPLING_LOG_SCALE_PARAM_CLAMP,
    )
    return base_log_scale, coupling_shift, coupling_log_scale


def _coupling_effects(
    base0: tf.Tensor,
    coupling_shift: tf.Tensor,
    coupling_log_scale: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return shift/log-scale effects for the second axis in the 2-D coupling flow."""
    shift = coupling_shift[:, :1] + coupling_shift[:, 1:2] * tf.tanh(base0)
    log_scale = (
        coupling_log_scale[:, :1]
        + coupling_log_scale[:, 1:2] * tf.tanh(base0)
    )
    shift = tf.clip_by_value(
        shift,
        -FLOW_COUPLING_SHIFT_EFFECT_CLAMP,
        FLOW_COUPLING_SHIFT_EFFECT_CLAMP,
    )
    log_scale = tf.clip_by_value(
        log_scale,
        -FLOW_COUPLING_LOG_SCALE_EFFECT_CLAMP,
        FLOW_COUPLING_LOG_SCALE_EFFECT_CLAMP,
    )
    return shift, log_scale


def _flow_forward_coupling_2d(
    eps: tf.Tensor,
    *,
    y_loc: tf.Tensor,
    base_log_scale: tf.Tensor,
    coupling_shift: tf.Tensor,
    coupling_log_scale: tf.Tensor,
) -> tf.Tensor:
    """Transform base Gaussian samples via a simple joint 2-D affine coupling flow."""
    base_log_scale, coupling_shift, coupling_log_scale = _clamp_coupling_flow_params(
        base_log_scale,
        coupling_shift,
        coupling_log_scale,
    )
    base_scale = tf.exp(base_log_scale)
    base0 = eps[:, :1]
    y0 = y_loc[:, :1] + base_scale[:, :1] * base0
    shift, eff_log_scale = _coupling_effects(base0, coupling_shift, coupling_log_scale)
    inner1 = tf.exp(eff_log_scale) * eps[:, 1:2] + shift
    y1 = y_loc[:, 1:2] + base_scale[:, 1:2] * inner1
    return tf.concat([y0, y1], axis=-1)


def _flow_inverse_coupling_2d(
    y: tf.Tensor,
    *,
    y_loc: tf.Tensor,
    base_log_scale: tf.Tensor,
    coupling_shift: tf.Tensor,
    coupling_log_scale: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Map observations back to the standard-normal base for the coupling flow."""
    base_log_scale, coupling_shift, coupling_log_scale = _clamp_coupling_flow_params(
        base_log_scale,
        coupling_shift,
        coupling_log_scale,
    )
    base_scale = tf.exp(base_log_scale)
    base0 = (y[:, :1] - y_loc[:, :1]) / (base_scale[:, :1] + 1e-6)
    shift, eff_log_scale = _coupling_effects(base0, coupling_shift, coupling_log_scale)
    inner1 = (y[:, 1:2] - y_loc[:, 1:2]) / (base_scale[:, 1:2] + 1e-6)
    base1 = (inner1 - shift) * tf.exp(-eff_log_scale)
    base = tf.concat([base0, base1], axis=-1)
    log_abs_det = -(
        base_log_scale[:, :1] + base_log_scale[:, 1:2] + eff_log_scale
    )
    return base, tf.squeeze(log_abs_det, axis=-1)


def _prepare_spline_params(
    raw_widths: tf.Tensor,
    raw_heights: tf.Tensor,
    raw_derivatives: tf.Tensor,
    *,
    tail_bound: float = FLOW_SPLINE_DEFAULT_BOUND,
    min_bin_width: float = FLOW_SPLINE_MIN_BIN_WIDTH,
    min_bin_height: float = FLOW_SPLINE_MIN_BIN_HEIGHT,
    min_derivative: float = FLOW_SPLINE_MIN_DERIVATIVE,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Convert raw per-axis spline parameters into normalized knot statistics."""
    n_bins = raw_widths.shape[-1]
    if n_bins is None:
        raise ValueError("spline flow requires a static bin count")
    n_bins = int(n_bins)
    if n_bins < 2:
        raise ValueError(f"spline flow requires at least 2 bins; got {n_bins}")

    total_width = 2.0 * float(tail_bound)
    total_height = 2.0 * float(tail_bound)
    min_total_width = float(min_bin_width) * n_bins
    min_total_height = float(min_bin_height) * n_bins
    if min_total_width >= total_width or min_total_height >= total_height:
        raise ValueError("spline min bin sizes exceed the configured tail bound")

    widths = tf.nn.softmax(raw_widths, axis=-1)
    widths = float(min_bin_width) + (total_width - min_total_width) * widths
    heights = tf.nn.softmax(raw_heights, axis=-1)
    heights = float(min_bin_height) + (total_height - min_total_height) * heights

    inner_derivatives = float(min_derivative) + tf.nn.softplus(raw_derivatives)
    boundary = tf.ones((tf.shape(inner_derivatives)[0], 1), dtype=inner_derivatives.dtype)
    derivatives = tf.concat([boundary, inner_derivatives, boundary], axis=-1)

    cumwidths = tf.pad(
        tf.cumsum(widths, axis=-1),
        paddings=[[0, 0], [1, 0]],
        constant_values=0.0,
    )
    cumwidths = cumwidths + (-float(tail_bound))
    cumheights = tf.pad(
        tf.cumsum(heights, axis=-1),
        paddings=[[0, 0], [1, 0]],
        constant_values=0.0,
    )
    cumheights = cumheights + (-float(tail_bound))
    return widths, heights, derivatives, cumwidths, cumheights


def _search_spline_bin(cumknot_inner: tf.Tensor, values: tf.Tensor) -> tf.Tensor:
    """Return the per-row spline bin index for ``values``."""
    return tf.squeeze(
        tf.searchsorted(cumknot_inner, tf.expand_dims(values, axis=-1), side="right"),
        axis=-1,
    )


def _rational_quadratic_spline_forward_flat(
    x: tf.Tensor,
    *,
    widths: tf.Tensor,
    heights: tf.Tensor,
    derivatives: tf.Tensor,
    cumwidths: tf.Tensor,
    cumheights: tf.Tensor,
    tail_bound: float = FLOW_SPLINE_DEFAULT_BOUND,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply an element-wise monotonic rational-quadratic spline."""
    x = tf.cast(x, widths.dtype)
    inside = tf.logical_and(x > -float(tail_bound), x < float(tail_bound))
    x_safe = tf.where(inside, x, tf.zeros_like(x))
    bin_idx = _search_spline_bin(cumwidths[:, 1:-1], x_safe)

    x0 = tf.gather(cumwidths[:, :-1], bin_idx, batch_dims=1)
    y0 = tf.gather(cumheights[:, :-1], bin_idx, batch_dims=1)
    w = tf.gather(widths, bin_idx, batch_dims=1)
    h = tf.gather(heights, bin_idx, batch_dims=1)
    d0 = tf.gather(derivatives[:, :-1], bin_idx, batch_dims=1)
    d1 = tf.gather(derivatives[:, 1:], bin_idx, batch_dims=1)

    theta = (x_safe - x0) / (w + 1e-6)
    one_minus_theta = 1.0 - theta
    delta = h / (w + 1e-6)
    curvature = d0 + d1 - 2.0 * delta
    denominator = delta + curvature * theta * one_minus_theta
    numerator = h * (
        delta * tf.square(theta) + d0 * theta * one_minus_theta
    )
    y_inside = y0 + numerator / (denominator + 1e-6)

    derivative_numer = tf.square(delta) * (
        d1 * tf.square(theta)
        + 2.0 * delta * theta * one_minus_theta
        + d0 * tf.square(one_minus_theta)
    )
    logabsdet_inside = tf.math.log(derivative_numer + 1e-6) - 2.0 * tf.math.log(
        denominator + 1e-6
    )

    y = tf.where(inside, y_inside, x)
    logabsdet = tf.where(inside, logabsdet_inside, tf.zeros_like(x))
    return y, logabsdet


def _rational_quadratic_spline_inverse_flat(
    y: tf.Tensor,
    *,
    widths: tf.Tensor,
    heights: tf.Tensor,
    derivatives: tf.Tensor,
    cumwidths: tf.Tensor,
    cumheights: tf.Tensor,
    tail_bound: float = FLOW_SPLINE_DEFAULT_BOUND,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Invert an element-wise monotonic rational-quadratic spline."""
    y = tf.cast(y, widths.dtype)
    inside = tf.logical_and(y > -float(tail_bound), y < float(tail_bound))
    y_safe = tf.where(inside, y, tf.zeros_like(y))
    bin_idx = _search_spline_bin(cumheights[:, 1:-1], y_safe)

    x0 = tf.gather(cumwidths[:, :-1], bin_idx, batch_dims=1)
    y0 = tf.gather(cumheights[:, :-1], bin_idx, batch_dims=1)
    w = tf.gather(widths, bin_idx, batch_dims=1)
    h = tf.gather(heights, bin_idx, batch_dims=1)
    d0 = tf.gather(derivatives[:, :-1], bin_idx, batch_dims=1)
    d1 = tf.gather(derivatives[:, 1:], bin_idx, batch_dims=1)

    delta = h / (w + 1e-6)
    delta_y = y_safe - y0
    curvature = d0 + d1 - 2.0 * delta
    a = delta_y * curvature + h * (delta - d0)
    b = h * d0 - delta_y * curvature
    c = -delta * delta_y
    discriminant = tf.maximum(tf.square(b) - 4.0 * a * c, 0.0)
    sqrt_discriminant = tf.sqrt(discriminant + 1e-12)
    theta_quadratic = (2.0 * c) / (-b - sqrt_discriminant + 1e-6)
    theta_linear = -c / (b + 1e-6)
    theta = tf.where(tf.abs(a) > 1e-6, theta_quadratic, theta_linear)
    theta = tf.clip_by_value(theta, 0.0, 1.0)

    x_inside = x0 + theta * w
    one_minus_theta = 1.0 - theta
    denominator = delta + curvature * theta * one_minus_theta
    derivative_numer = tf.square(delta) * (
        d1 * tf.square(theta)
        + 2.0 * delta * theta * one_minus_theta
        + d0 * tf.square(one_minus_theta)
    )
    logabsdet_inside = -(
        tf.math.log(derivative_numer + 1e-6)
        - 2.0 * tf.math.log(denominator + 1e-6)
    )

    x = tf.where(inside, x_inside, y)
    logabsdet = tf.where(inside, logabsdet_inside, tf.zeros_like(y))
    return x, logabsdet


def _flow_forward_spline_2d(
    eps: tf.Tensor,
    *,
    y_loc: tf.Tensor,
    raw_widths: tf.Tensor,
    raw_heights: tf.Tensor,
    raw_derivatives: tf.Tensor,
) -> tf.Tensor:
    """Transform base Gaussian samples through an independent 2-D RQ spline."""
    n_bins = raw_widths.shape[-1]
    if n_bins is None:
        raise ValueError("spline_2d flow requires a static raw_widths shape")
    n_bins = int(n_bins) // 2
    eps_flat = tf.reshape(eps, (-1,))
    widths, heights, derivatives, cumwidths, cumheights = _prepare_spline_params(
        tf.reshape(raw_widths, (-1, n_bins)),
        tf.reshape(raw_heights, (-1, n_bins)),
        tf.reshape(raw_derivatives, (-1, n_bins - 1)),
    )
    y_flat, _ = _rational_quadratic_spline_forward_flat(
        eps_flat,
        widths=widths,
        heights=heights,
        derivatives=derivatives,
        cumwidths=cumwidths,
        cumheights=cumheights,
    )
    y_rel = tf.reshape(y_flat, tf.shape(eps))
    return y_loc + y_rel


def _flow_inverse_spline_2d(
    y: tf.Tensor,
    *,
    y_loc: tf.Tensor,
    raw_widths: tf.Tensor,
    raw_heights: tf.Tensor,
    raw_derivatives: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Map spline-decoder observations back to the standard-normal base."""
    n_bins = raw_widths.shape[-1]
    if n_bins is None:
        raise ValueError("spline_2d flow requires a static raw_widths shape")
    n_bins = int(n_bins) // 2
    centered = y - y_loc
    centered_flat = tf.reshape(centered, (-1,))
    widths, heights, derivatives, cumwidths, cumheights = _prepare_spline_params(
        tf.reshape(raw_widths, (-1, n_bins)),
        tf.reshape(raw_heights, (-1, n_bins)),
        tf.reshape(raw_derivatives, (-1, n_bins - 1)),
    )
    base_flat, logabsdet_flat = _rational_quadratic_spline_inverse_flat(
        centered_flat,
        widths=widths,
        heights=heights,
        derivatives=derivatives,
        cumwidths=cumwidths,
        cumheights=cumheights,
    )
    base = tf.reshape(base_flat, tf.shape(centered))
    logabsdet = tf.reshape(logabsdet_flat, tf.shape(centered))
    return base, tf.reduce_sum(logabsdet, axis=-1)


def flow_reconstruction_loss(
    y_true: tf.Tensor,
    out_params: tf.Tensor,
    *,
    flow_family: str = "coupling_2d",
) -> tf.Tensor:
    """Exact NLL for the configured conditional flow decoder."""
    family = _resolve_flow_family(flow_family)
    y_loc, p1, p2, p3 = _unpack_flow_params(out_params, flow_family=family)
    if family == "sinh_arcsinh":
        base, log_abs_det = _flow_inverse_sinh_arcsinh(
            y_true,
            y_loc=y_loc,
            log_scale=p1,
            skew=p2,
            log_tail=p3,
        )
    elif family == "spline_2d":
        base, log_abs_det = _flow_inverse_spline_2d(
            y_true,
            y_loc=y_loc,
            raw_widths=p1,
            raw_heights=p2,
            raw_derivatives=p3,
        )
    else:
        base, log_abs_det = _flow_inverse_coupling_2d(
            y_true,
            y_loc=y_loc,
            base_log_scale=p1,
            coupling_shift=p2,
            coupling_log_scale=p3,
        )
    log_base = -0.5 * (
        tf.square(base) + tf.cast(np.log(2.0 * np.pi), tf.float32)
    )
    log_prob = tf.reduce_sum(log_base, axis=-1) + log_abs_det
    return -tf.reduce_mean(log_prob)


def _flow_deterministic_point(
    out_params: tf.Tensor,
    *,
    flow_family: str = "coupling_2d",
) -> tf.Tensor:
    """Median-like deterministic representative for the configured flow family."""
    family = _resolve_flow_family(flow_family)
    y_loc, p1, p2, p3 = _unpack_flow_params(out_params, flow_family=family)
    zeros = tf.zeros_like(y_loc)
    if family == "sinh_arcsinh":
        return _flow_forward_sinh_arcsinh(
            zeros,
            y_loc=y_loc,
            log_scale=p1,
            skew=p2,
            log_tail=p3,
        )
    if family == "spline_2d":
        return _flow_forward_spline_2d(
            zeros,
            y_loc=y_loc,
            raw_widths=p1,
            raw_heights=p2,
            raw_derivatives=p3,
        )
    return _flow_forward_coupling_2d(
        zeros,
        y_loc=y_loc,
        base_log_scale=p1,
        coupling_shift=p2,
        coupling_log_scale=p3,
    )


def _sample_flow(
    out_params: tf.Tensor,
    *,
    flow_family: str = "coupling_2d",
) -> tf.Tensor:
    """Draw one stochastic sample from the configured conditional flow decoder."""
    family = _resolve_flow_family(flow_family)
    y_loc, p1, p2, p3 = _unpack_flow_params(out_params, flow_family=family)
    eps = tf.random.normal(tf.shape(y_loc), dtype=y_loc.dtype)
    if family == "sinh_arcsinh":
        return _flow_forward_sinh_arcsinh(
            eps,
            y_loc=y_loc,
            log_scale=p1,
            skew=p2,
            log_tail=p3,
        )
    if family == "spline_2d":
        return _flow_forward_spline_2d(
            eps,
            y_loc=y_loc,
            raw_widths=p1,
            raw_heights=p2,
            raw_derivatives=p3,
        )
    return _flow_forward_coupling_2d(
        eps,
        y_loc=y_loc,
        base_log_scale=p1,
        coupling_shift=p2,
        coupling_log_scale=p3,
    )


def _batch_axis_stats(x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return std/skew/kurtosis along the batch axis for each output channel."""
    x = tf.cast(x, tf.float32)
    x_center = x - tf.reduce_mean(x, axis=0, keepdims=True)
    var = tf.reduce_mean(tf.square(x_center), axis=0)
    std = tf.sqrt(var + 1e-6)
    z = x_center / std
    skew = tf.reduce_mean(tf.pow(z, 3.0), axis=0)
    kurt = tf.reduce_mean(tf.pow(z, 4.0), axis=0) - 3.0
    return std, skew, kurt


def axis_moment_loss_tf(
    r_real: tf.Tensor,
    r_gen: tf.Tensor,
    *,
    std_weight: float = 1.0,
    skew_weight: float = 0.25,
    kurt_weight: float = 0.10,
) -> tf.Tensor:
    """Axis-wise distribution proxy using std, skew, and kurtosis."""
    std_real, skew_real, kurt_real = _batch_axis_stats(tf.stop_gradient(r_real))
    std_gen, skew_gen, kurt_gen = _batch_axis_stats(r_gen)

    std_term = tf.reduce_mean(
        tf.square(tf.math.log((std_gen + 1e-6) / (std_real + 1e-6)))
    )
    skew_term = tf.reduce_mean(tf.square(skew_gen - skew_real))
    kurt_term = tf.reduce_mean(tf.square(kurt_gen - kurt_real))
    return (
        tf.cast(std_weight, tf.float32) * std_term
        + tf.cast(skew_weight, tf.float32) * skew_term
        + tf.cast(kurt_weight, tf.float32) * kurt_term
    )


def kurt_only_loss_tf(r_real: tf.Tensor, r_gen: tf.Tensor) -> tf.Tensor:
    """Kurtosis-only MSE loss between real and generated residuals.

    Computes excess kurtosis for each output channel and returns the mean
    squared difference.  Isolates the 4th-moment signal without std/skew
    interference so ``lambda_kurt`` can be tuned independently.
    """
    _, _, kurt_real = _batch_axis_stats(tf.stop_gradient(r_real))
    _, _, kurt_gen = _batch_axis_stats(r_gen)
    return tf.reduce_mean(tf.square(kurt_gen - kurt_real))


def _quantile_axis0(x: tf.Tensor, q: float) -> tf.Tensor:
    """Approximate per-axis quantile along the batch dimension."""
    x = tf.cast(x, tf.float32)
    x_sorted = tf.sort(x, axis=0)
    n = tf.shape(x_sorted)[0]
    idx = tf.cast(
        tf.round(tf.cast(n - 1, tf.float32) * tf.cast(q, tf.float32)),
        tf.int32,
    )
    return tf.gather(x_sorted, idx, axis=0)


def axis_coverage_tail_loss_tf(
    r_real: tf.Tensor,
    r_gen: tf.Tensor,
    *,
    coverage_levels: tuple[float, ...] = (0.50, 0.80, 0.95),
    tail_levels: tuple[float, ...] = (0.05, 0.95),
    temperature: float = 0.05,
) -> tf.Tensor:
    """Axis-wise calibration loss using central coverage and tail mass."""
    r_real = tf.cast(tf.stop_gradient(r_real), tf.float32)
    r_gen = tf.cast(r_gen, tf.float32)
    temp = tf.maximum(tf.cast(temperature, tf.float32), tf.constant(1e-4, tf.float32))

    losses = []

    abs_real = tf.abs(r_real)
    abs_gen = tf.abs(r_gen)
    for level in coverage_levels:
        thr = _quantile_axis0(abs_real, float(level))
        pred_cov = tf.reduce_mean(tf.sigmoid((thr - abs_gen) / temp), axis=0)
        target = tf.fill(tf.shape(pred_cov), tf.cast(level, tf.float32))
        losses.append(tf.reduce_mean(tf.square(pred_cov - target)))

    for level in tail_levels:
        q = float(level)
        if q <= 0.5:
            thr = _quantile_axis0(r_real, q)
            pred_tail = tf.reduce_mean(tf.sigmoid((thr - r_gen) / temp), axis=0)
            target = tf.fill(tf.shape(pred_tail), tf.cast(q, tf.float32))
        else:
            thr = _quantile_axis0(r_real, q)
            pred_tail = tf.reduce_mean(tf.sigmoid((r_gen - thr) / temp), axis=0)
            target = tf.fill(tf.shape(pred_tail), tf.cast(1.0 - q, tf.float32))
        losses.append(tf.reduce_mean(tf.square(pred_tail - target)))

    if not losses:
        return tf.constant(0.0, dtype=tf.float32)
    return tf.add_n(losses) / tf.cast(len(losses), tf.float32)


def _log_psd_1d(x: tf.Tensor) -> tf.Tensor:
    """Return log-PSD for a 1-D signal represented along the batch axis."""
    x = tf.cast(x, tf.float32)
    n = tf.shape(x)[0]
    x = x - tf.reduce_mean(x)
    window = tf.signal.hann_window(n, periodic=True, dtype=tf.float32)
    spec = tf.signal.rfft(x * window)
    power = tf.square(tf.abs(spec))
    return tf.math.log(power + 1e-6)


def spectral_psd_loss_tf(
    r_real: tf.Tensor,
    r_gen: tf.Tensor,
    min_batch: int = 64,
) -> tf.Tensor:
    """Batch-order PSD proxy; valid only when batch order preserves time."""
    n = tf.shape(r_real)[0]

    def _compute() -> tf.Tensor:
        real_i = _log_psd_1d(r_real[:, 0])
        real_q = _log_psd_1d(r_real[:, 1])
        gen_i = _log_psd_1d(r_gen[:, 0])
        gen_q = _log_psd_1d(r_gen[:, 1])
        return 0.5 * (
            tf.reduce_mean(tf.square(gen_i - tf.stop_gradient(real_i)))
            + tf.reduce_mean(tf.square(gen_q - tf.stop_gradient(real_q)))
        )

    return tf.cond(n >= int(min_batch), _compute, lambda: tf.constant(0.0, tf.float32))


# ======================================================================
# Keras layer (used inside training graph — wraps the above functions)
# ======================================================================
@tf.keras.utils.register_keras_serializable(package="VLC")
class CondPriorVAELoss(layers.Layer):
    """Heteroscedastic Gaussian NLL + KL(q‖p) with β-annealing, free-bits,
    and optional auxiliary MMD² loss term.

    Inputs (call):
        (y_true, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p)
        or, when lambda_mmd > 0:
        (y_true, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p, x_center)

    The layer adds the total loss via ``self.add_loss`` and tracks
    ``recon_loss`` / ``kl_loss`` (and ``mmd_loss`` when active) as Keras metrics.
    """

    def __init__(
        self,
        beta: float = 1.0,
        free_bits: float = 0.0,
        lambda_mmd: float = 0.0,
        lambda_axis: float = 0.0,
        lambda_psd: float = 0.0,
        lambda_coverage: float = 0.0,
        lambda_kurt: float = 0.0,
        axis_std_weight: float = 1.0,
        axis_skew_weight: float = 0.25,
        axis_kurt_weight: float = 0.10,
        coverage_levels: tuple[float, ...] = (0.50, 0.80, 0.95),
        tail_levels: tuple[float, ...] = (0.05, 0.95),
        coverage_temperature: float = 0.05,
        mmd_mode: str = "mean_residual",
        decoder_distribution: str = "gaussian",
        flow_family: str = "coupling_2d",
        mdn_components: int = 1,
        mmd_bandwidth: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.beta_init = float(beta)
        self.free_bits = float(free_bits)
        self.lambda_mmd = float(lambda_mmd)
        self.lambda_axis = float(lambda_axis)
        self.lambda_psd = float(lambda_psd)
        self.lambda_coverage = float(lambda_coverage)
        self.lambda_kurt = float(lambda_kurt)
        self.axis_std_weight = float(axis_std_weight)
        self.axis_skew_weight = float(axis_skew_weight)
        self.axis_kurt_weight = float(axis_kurt_weight)
        self.coverage_levels = tuple(float(x) for x in coverage_levels)
        self.tail_levels = tuple(float(x) for x in tail_levels)
        self.coverage_temperature = float(coverage_temperature)
        self.mmd_mode = _resolve_mmd_mode(mmd_mode)
        self.decoder_distribution = _resolve_decoder_distribution(decoder_distribution)
        self.flow_family = _resolve_flow_family(flow_family)
        self.mdn_components = int(mdn_components)
        self.mmd_bandwidth = mmd_bandwidth
        self.beta = tf.Variable(
            self.beta_init, trainable=False, dtype=tf.float32, name="beta",
        )
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        if self.lambda_mmd > 0.0:
            self.mmd_loss_tracker = tf.keras.metrics.Mean(name="mmd_loss")
        if self.lambda_axis > 0.0:
            self.axis_loss_tracker = tf.keras.metrics.Mean(name="axis_loss")
        if self.lambda_psd > 0.0:
            self.psd_loss_tracker = tf.keras.metrics.Mean(name="psd_loss")
        if self.lambda_coverage > 0.0:
            self.coverage_loss_tracker = tf.keras.metrics.Mean(name="coverage_loss")
        if self.lambda_kurt > 0.0:
            self.kurt_loss_tracker = tf.keras.metrics.Mean(name="kurt_loss")

    def call(self, inputs):
        if len(inputs) == 7:
            y_true, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p, x_center = inputs
        else:
            y_true, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p = inputs
            x_center = None

        if self.decoder_distribution == "mdn":
            logits, comp_mean, comp_log_var = _unpack_mdn_params(
                out_params, self.mdn_components
            )
            y_mean = _mdn_expected_mean(logits, comp_mean)
            recon = mdn_reconstruction_loss(y_true, logits, comp_mean, comp_log_var)
            y_sample_cache = None

            def _ensure_sample():
                nonlocal y_sample_cache
                if y_sample_cache is None:
                    y_sample_cache = _sample_mdn(logits, comp_mean, comp_log_var)
                return y_sample_cache
        elif self.decoder_distribution == "flow":
            y_mean = _flow_deterministic_point(
                out_params, flow_family=self.flow_family
            )
            recon = flow_reconstruction_loss(
                y_true,
                out_params,
                flow_family=self.flow_family,
            )
            y_sample_cache = None

            def _ensure_sample():
                nonlocal y_sample_cache
                if y_sample_cache is None:
                    y_sample_cache = _sample_flow(
                        out_params, flow_family=self.flow_family
                    )
                return y_sample_cache
        else:
            y_mean, y_log_var = _unpack_gaussian_params(out_params)
            recon = reconstruction_loss(y_true, y_mean, y_log_var)
            y_sample_cache = None

            def _ensure_sample():
                nonlocal y_sample_cache
                if y_sample_cache is None:
                    y_sample_cache = _sample_heteroscedastic(y_mean, y_log_var)
                return y_sample_cache

        kl_per_sample = kl_divergence(
            z_mean_q, z_log_var_q, z_mean_p, z_log_var_p,
        )
        kl_fb = kl_with_freebits(kl_per_sample, self.free_bits)
        kl = tf.reduce_mean(kl_fb)

        total = compute_total_loss(recon, kl, self.beta)

        if self.lambda_mmd > 0.0 and x_center is not None:
            r_real = tf.stop_gradient(y_true - x_center)
            if self.mmd_mode == "sampled_residual":
                r_gen = _ensure_sample() - x_center
            else:
                r_gen = y_mean - x_center
            mmd2 = mmd2_tf(r_real, r_gen, n_sub=512, bandwidth=self.mmd_bandwidth)
            self.mmd_loss_tracker.update_state(mmd2)
            total = total + self.lambda_mmd * mmd2

        if (
            self.lambda_axis > 0.0
            or self.lambda_psd > 0.0
            or self.lambda_coverage > 0.0
            or self.lambda_kurt > 0.0
        ) and x_center is not None:
            r_real = tf.stop_gradient(y_true - x_center)
            r_gen_sample = _ensure_sample() - x_center

            if self.lambda_axis > 0.0:
                axis_loss = axis_moment_loss_tf(
                    r_real,
                    r_gen_sample,
                    std_weight=self.axis_std_weight,
                    skew_weight=self.axis_skew_weight,
                    kurt_weight=self.axis_kurt_weight,
                )
                self.axis_loss_tracker.update_state(axis_loss)
                total = total + self.lambda_axis * axis_loss

            if self.lambda_psd > 0.0:
                psd_loss = spectral_psd_loss_tf(r_real, r_gen_sample)
                self.psd_loss_tracker.update_state(psd_loss)
                total = total + self.lambda_psd * psd_loss

            if self.lambda_coverage > 0.0:
                coverage_loss = axis_coverage_tail_loss_tf(
                    r_real,
                    r_gen_sample,
                    coverage_levels=self.coverage_levels,
                    tail_levels=self.tail_levels,
                    temperature=self.coverage_temperature,
                )
                self.coverage_loss_tracker.update_state(coverage_loss)
                total = total + self.lambda_coverage * coverage_loss

            if self.lambda_kurt > 0.0:
                kurt_loss = kurt_only_loss_tf(r_real, r_gen_sample)
                self.kurt_loss_tracker.update_state(kurt_loss)
                total = total + self.lambda_kurt * kurt_loss

        self.add_loss(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(tf.reduce_mean(kl_per_sample))
        return y_mean

    @property
    def metrics(self):
        m = [self.recon_loss_tracker, self.kl_loss_tracker]
        if self.lambda_mmd > 0.0:
            m.append(self.mmd_loss_tracker)
        if self.lambda_axis > 0.0:
            m.append(self.axis_loss_tracker)
        if self.lambda_psd > 0.0:
            m.append(self.psd_loss_tracker)
        if self.lambda_coverage > 0.0:
            m.append(self.coverage_loss_tracker)
        if self.lambda_kurt > 0.0:
            m.append(self.kurt_loss_tracker)
        return m

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "beta": self.beta_init,
            "free_bits": self.free_bits,
            "lambda_mmd": self.lambda_mmd,
            "lambda_axis": self.lambda_axis,
            "lambda_psd": self.lambda_psd,
            "lambda_coverage": self.lambda_coverage,
            "lambda_kurt": self.lambda_kurt,
            "axis_std_weight": self.axis_std_weight,
            "axis_skew_weight": self.axis_skew_weight,
            "axis_kurt_weight": self.axis_kurt_weight,
            "coverage_levels": list(self.coverage_levels),
            "tail_levels": list(self.tail_levels),
            "coverage_temperature": self.coverage_temperature,
            "mmd_mode": self.mmd_mode,
            "decoder_distribution": self.decoder_distribution,
            "flow_family": self.flow_family,
            "mdn_components": self.mdn_components,
            "mmd_bandwidth": self.mmd_bandwidth,
        })
        return cfg


@tf.keras.utils.register_keras_serializable(package="VLC")
class CondPriorDeltaVAELoss(layers.Layer):
    """Explicit residual-target loss for ``delta_residual`` point-wise models.

    The decoder outputs residual parameters ``(Δ_mean, Δ_log_var)`` while the
    training target remains the received signal ``y_true``. This layer converts
    to the residual target ``Δ_true = y_true - x_true`` internally, optimises
    that heteroscedastic NLL, and returns ``y_mean = x_true + Δ_mean`` so the
    external model contract remains unchanged.

    Inputs (call):
        (y_true, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p, x_true)
    """

    def __init__(
        self,
        beta: float = 1.0,
        free_bits: float = 0.0,
        lambda_mmd: float = 0.0,
        lambda_axis: float = 0.0,
        lambda_psd: float = 0.0,
        lambda_coverage: float = 0.0,
        coverage_levels: tuple[float, ...] = (0.50, 0.80, 0.95),
        tail_levels: tuple[float, ...] = (0.05, 0.95),
        coverage_temperature: float = 0.05,
        mmd_mode: str = "mean_residual",
        decoder_distribution: str = "gaussian",
        mmd_bandwidth: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.beta_init = float(beta)
        self.free_bits = float(free_bits)
        self.lambda_mmd = float(lambda_mmd)
        self.lambda_axis = float(lambda_axis)
        self.lambda_psd = float(lambda_psd)
        self.lambda_coverage = float(lambda_coverage)
        self.coverage_levels = tuple(float(x) for x in coverage_levels)
        self.tail_levels = tuple(float(x) for x in tail_levels)
        self.coverage_temperature = float(coverage_temperature)
        self.mmd_mode = _resolve_mmd_mode(mmd_mode)
        self.decoder_distribution = _resolve_decoder_distribution(decoder_distribution)
        if self.decoder_distribution != "gaussian":
            raise ValueError(
                "CondPriorDeltaVAELoss currently supports only "
                "decoder_distribution='gaussian'."
            )
        self.mmd_bandwidth = mmd_bandwidth
        self.beta = tf.Variable(
            self.beta_init, trainable=False, dtype=tf.float32, name="beta",
        )
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        if self.lambda_mmd > 0.0:
            self.mmd_loss_tracker = tf.keras.metrics.Mean(name="mmd_loss")
        if self.lambda_axis > 0.0:
            self.axis_loss_tracker = tf.keras.metrics.Mean(name="axis_loss")
        if self.lambda_psd > 0.0:
            self.psd_loss_tracker = tf.keras.metrics.Mean(name="psd_loss")
        if self.lambda_coverage > 0.0:
            self.coverage_loss_tracker = tf.keras.metrics.Mean(name="coverage_loss")

    def call(self, inputs):
        if len(inputs) < 7:
            raise ValueError(
                "CondPriorDeltaVAELoss expects "
                "(y_true, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p, x_true)."
            )

        y_true, out_params, z_mean_q, z_log_var_q, z_mean_p, z_log_var_p, x_true = inputs[:7]

        delta_true = y_true - x_true
        delta_mean = out_params[:, :2]
        delta_log_var = out_params[:, 2:]

        recon = reconstruction_loss(delta_true, delta_mean, delta_log_var)

        kl_per_sample = kl_divergence(
            z_mean_q, z_log_var_q, z_mean_p, z_log_var_p,
        )
        kl_fb = kl_with_freebits(kl_per_sample, self.free_bits)
        kl = tf.reduce_mean(kl_fb)

        total = compute_total_loss(recon, kl, self.beta)

        delta_sample_cache = None

        def _ensure_sample():
            nonlocal delta_sample_cache
            if delta_sample_cache is None:
                delta_sample_cache = _sample_heteroscedastic(delta_mean, delta_log_var)
            return delta_sample_cache

        if self.lambda_mmd > 0.0:
            if self.mmd_mode == "sampled_residual":
                delta_gen = _ensure_sample()
            else:
                delta_gen = delta_mean
            mmd2 = mmd2_tf(
                tf.stop_gradient(delta_true),
                delta_gen,
                n_sub=512,
                bandwidth=self.mmd_bandwidth,
            )
            self.mmd_loss_tracker.update_state(mmd2)
            total = total + self.lambda_mmd * mmd2

        if self.lambda_axis > 0.0:
            axis_loss = axis_moment_loss_tf(
                tf.stop_gradient(delta_true), _ensure_sample()
            )
            self.axis_loss_tracker.update_state(axis_loss)
            total = total + self.lambda_axis * axis_loss

        if self.lambda_psd > 0.0:
            psd_loss = spectral_psd_loss_tf(
                tf.stop_gradient(delta_true), _ensure_sample()
            )
            self.psd_loss_tracker.update_state(psd_loss)
            total = total + self.lambda_psd * psd_loss

        if self.lambda_coverage > 0.0:
            coverage_loss = axis_coverage_tail_loss_tf(
                tf.stop_gradient(delta_true),
                _ensure_sample(),
                coverage_levels=self.coverage_levels,
                tail_levels=self.tail_levels,
                temperature=self.coverage_temperature,
            )
            self.coverage_loss_tracker.update_state(coverage_loss)
            total = total + self.lambda_coverage * coverage_loss

        self.add_loss(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(tf.reduce_mean(kl_per_sample))
        return x_true + delta_mean

    @property
    def metrics(self):
        m = [self.recon_loss_tracker, self.kl_loss_tracker]
        if self.lambda_mmd > 0.0:
            m.append(self.mmd_loss_tracker)
        if self.lambda_axis > 0.0:
            m.append(self.axis_loss_tracker)
        if self.lambda_psd > 0.0:
            m.append(self.psd_loss_tracker)
        if self.lambda_coverage > 0.0:
            m.append(self.coverage_loss_tracker)
        return m

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "beta": self.beta_init,
            "free_bits": self.free_bits,
            "lambda_mmd": self.lambda_mmd,
            "lambda_axis": self.lambda_axis,
            "lambda_psd": self.lambda_psd,
            "lambda_coverage": self.lambda_coverage,
            "coverage_levels": list(self.coverage_levels),
            "tail_levels": list(self.tail_levels),
            "coverage_temperature": self.coverage_temperature,
            "mmd_mode": self.mmd_mode,
            "decoder_distribution": self.decoder_distribution,
            "mmd_bandwidth": self.mmd_bandwidth,
        })
        return cfg


@tf.keras.utils.register_keras_serializable(package="VLC")
class StdNormalHeteroscedasticVAELoss(layers.Layer):
    """Heteroscedastic Gaussian NLL + KL(q‖N(0,I)) with β-annealing.

    Inputs (call):
        (y_true, out_params, z_mean_q, z_log_var_q, *ignored)

    Any extra tensors are accepted and ignored. This lets the training graph
    keep compatibility-only submodels (for example ``prior_net`` in the
    legacy-2025 variant) connected without changing the actual loss formula.
    """

    def __init__(self, beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta_init = float(beta)
        self.beta = tf.Variable(
            self.beta_init, trainable=False, dtype=tf.float32, name="beta",
        )
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        if len(inputs) < 4:
            raise ValueError(
                "StdNormalHeteroscedasticVAELoss expects at least "
                "(y_true, out_params, z_mean_q, z_log_var_q)."
            )
        y_true, out_params, z_mean_q, z_log_var_q = inputs[:4]

        y_mean = out_params[:, :2]
        y_log_var = out_params[:, 2:]

        recon = reconstruction_loss(y_true, y_mean, y_log_var)
        kl_per_sample = kl_to_standard_normal(z_mean_q, z_log_var_q)
        kl = tf.reduce_mean(kl_per_sample)

        total = compute_total_loss(recon, kl, self.beta)

        self.add_loss(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)
        return y_mean

    @property
    def metrics(self):
        return [self.recon_loss_tracker, self.kl_loss_tracker]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"beta": self.beta_init})
        return cfg
