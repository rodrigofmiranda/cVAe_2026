# -*- coding: utf-8 -*-
"""Canonical cVAE grid definition and override filtering."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional


def _cfg(**kwargs: Any) -> Dict[str, Any]:
    cfg = dict(
        activation="leaky_relu",
        kl_anneal_epochs=80,
        batch_size=16384,
        lr=3e-4,
        dropout=0.0,
        free_bits=0.10,
        arch_variant="concat",
        mmd_mode="mean_residual",
    )
    cfg.update(kwargs)
    return cfg


def _tag_beta(beta: float) -> str:
    return f"{beta:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _tag_lr(lr: float) -> str:
    return f"{lr:.7f}".rstrip("0").rstrip(".").replace(".", "p")


def _tag_layers(layer_sizes: List[int]) -> str:
    return "-".join(str(x) for x in layer_sizes)


def build_default_grid() -> List[Dict[str, Any]]:
    """Return the canonical cVAE grid without any runtime filtering."""
    grid: List[Dict[str, Any]] = []

    grid += [
        dict(
            group="G0_ref",
            tag=f"G0_lat4_b{_tag_beta(0.003)}_fb0p10_lr{_tag_lr(3e-4)}_L{_tag_layers([128,256,512])}",
            cfg=_cfg(layer_sizes=[128, 256, 512], latent_dim=4, beta=0.003),
        ),
        dict(
            group="G0_ref",
            tag=f"G0_lat4_b{_tag_beta(0.001)}_fb0p10_lr{_tag_lr(3e-4)}_L{_tag_layers([128,256,512])}",
            cfg=_cfg(layer_sizes=[128, 256, 512], latent_dim=4, beta=0.001),
        ),
    ]

    for latent_dim in [4, 6, 8]:
        for beta in [0.001, 0.002, 0.003]:
            for dropout in [0.0, 0.05]:
                grid.append(
                    dict(
                        group="G1_core",
                        tag=(
                            f"G1_lat{latent_dim}_b{_tag_beta(beta)}_fb0p10_"
                            f"do{str(dropout).replace('.','p')}_lr{_tag_lr(3e-4)}_"
                            f"L{_tag_layers([128,256,512])}"
                        ),
                        cfg=_cfg(
                            layer_sizes=[128, 256, 512],
                            latent_dim=latent_dim,
                            beta=beta,
                            dropout=dropout,
                        ),
                    )
                )

    for latent_dim in [4, 6]:
        for beta in [0.001, 0.002]:
            for free_bits in [0.0, 0.05, 0.20]:
                grid.append(
                    dict(
                        group="G2_freebits",
                        tag=(
                            f"G2_lat{latent_dim}_b{_tag_beta(beta)}_"
                            f"fb{str(free_bits).replace('.','p')}_lr{_tag_lr(3e-4)}_"
                            f"L{_tag_layers([128,256,512])}"
                        ),
                        cfg=_cfg(
                            layer_sizes=[128, 256, 512],
                            latent_dim=latent_dim,
                            beta=beta,
                            free_bits=free_bits,
                        ),
                    )
                )

    for latent_dim, beta, lr, batch_size, anneal_epochs in [
        (6, 0.002, 2e-4, 16384, 80),
        (6, 0.002, 2e-4, 8192, 80),
        (6, 0.002, 3e-4, 8192, 80),
        (6, 0.001, 2e-4, 16384, 100),
        (6, 0.001, 3e-4, 16384, 100),
        (8, 0.002, 2e-4, 16384, 100),
        (8, 0.002, 3e-4, 8192, 100),
        (4, 0.002, 2e-4, 16384, 60),
        (4, 0.001, 2e-4, 8192, 60),
        (8, 0.003, 2e-4, 16384, 80),
    ]:
        grid.append(
            dict(
                group="G3_opt",
                tag=(
                    f"G3_lat{latent_dim}_b{_tag_beta(beta)}_fb0p10_lr{_tag_lr(lr)}_"
                    f"bs{batch_size}_anneal{anneal_epochs}_L{_tag_layers([128,256,512])}"
                ),
                cfg=_cfg(
                    layer_sizes=[128, 256, 512],
                    latent_dim=latent_dim,
                    beta=beta,
                    lr=lr,
                    batch_size=batch_size,
                    kl_anneal_epochs=anneal_epochs,
                ),
            )
        )

    for latent_dim in [4, 6]:
        for beta in [0.010, 0.030, 0.100]:
            grid.append(
                dict(
                    group="G4_beta_latent_sweep",
                    tag=f"G4_lat{latent_dim}_b{_tag_beta(beta)}_fb0p10_lr{_tag_lr(3e-4)}_L{_tag_layers([128,256,512])}",
                    cfg=_cfg(layer_sizes=[128, 256, 512], latent_dim=latent_dim, beta=beta),
                )
            )

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in grid:
        if item["tag"] in seen:
            continue
        seen.add(item["tag"])
        deduped.append(item)
    return deduped


def _preset_exploratory_small(grid: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a compact exploratory subset for expensive full-data runs."""
    keep_tags = [
        "G0_lat4_b0p003_fb0p10_lr0p0003_L128-256-512",
        "G0_lat4_b0p001_fb0p10_lr0p0003_L128-256-512",
        "G1_lat6_b0p002_fb0p10_do0p0_lr0p0003_L128-256-512",
        "G1_lat6_b0p002_fb0p10_do0p05_lr0p0003_L128-256-512",
        "G2_lat4_b0p001_fb0p0_lr0p0003_L128-256-512",
        "G2_lat4_b0p001_fb0p2_lr0p0003_L128-256-512",
        "G3_lat6_b0p002_fb0p10_lr0p0002_bs16384_anneal80_L128-256-512",
        "G3_lat8_b0p002_fb0p10_lr0p0002_bs16384_anneal100_L128-256-512",
    ]
    by_tag = {item["tag"]: item for item in grid}
    return [by_tag[tag] for tag in keep_tags if tag in by_tag]


def _residual_candidates() -> List[Dict[str, Any]]:
    """Return the compact residual-architecture candidates."""
    residual_cfg = dict(layer_sizes=[128, 256, 512], latent_dim=4, arch_variant="channel_residual")
    return [
        dict(
            group="R0_residual",
            tag=f"R0res_lat4_b{_tag_beta(0.003)}_fb0p10_lr{_tag_lr(3e-4)}_L{_tag_layers([128,256,512])}",
            cfg=_cfg(beta=0.003, free_bits=0.10, **residual_cfg),
        ),
        dict(
            group="R0_residual",
            tag=f"R0res_lat4_b{_tag_beta(0.001)}_fb0p10_lr{_tag_lr(3e-4)}_L{_tag_layers([128,256,512])}",
            cfg=_cfg(beta=0.001, free_bits=0.10, **residual_cfg),
        ),
        dict(
            group="R1_residual",
            tag=f"R1res_lat4_b{_tag_beta(0.001)}_fb0p0_lr{_tag_lr(3e-4)}_L{_tag_layers([128,256,512])}",
            cfg=_cfg(beta=0.001, free_bits=0.0, **residual_cfg),
        ),
        dict(
            group="R1_residual",
            tag=f"R1res_lat4_b{_tag_beta(0.002)}_fb0p0_lr{_tag_lr(3e-4)}_L{_tag_layers([128,256,512])}",
            cfg=_cfg(beta=0.002, free_bits=0.0, **residual_cfg),
        ),
    ]


def _preset_residual_small(grid: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the dedicated residual-architecture exploratory subset."""
    merged = list(grid) + _residual_candidates()
    keep_tags = [
        "R0res_lat4_b0p003_fb0p10_lr0p0003_L128-256-512",
        "R0res_lat4_b0p001_fb0p10_lr0p0003_L128-256-512",
        "R1res_lat4_b0p001_fb0p0_lr0p0003_L128-256-512",
        "R1res_lat4_b0p002_fb0p0_lr0p0003_L128-256-512",
    ]
    by_tag = {item["tag"]: item for item in merged}
    return [by_tag[tag] for tag in keep_tags if tag in by_tag]


def _delta_residual_candidates() -> List[Dict[str, Any]]:
    """Return explicit residual-target candidates for the simplified hypothesis."""
    base_cfg = dict(
        layer_sizes=[128, 256, 512],
        latent_dim=4,
        arch_variant="delta_residual",
    )
    return [
        dict(
            group="D0_delta_smoke",
            tag=f"D0delta_lat4_b{_tag_beta(0.001)}_fb0p10_lr{_tag_lr(3e-4)}_L{_tag_layers([128,256,512])}",
            cfg=_cfg(beta=0.001, free_bits=0.10, **base_cfg),
        ),
        dict(
            group="D1_delta_small",
            tag=f"D1delta_lat4_b{_tag_beta(0.003)}_fb0p10_lr{_tag_lr(3e-4)}_L{_tag_layers([128,256,512])}",
            cfg=_cfg(beta=0.003, free_bits=0.10, **base_cfg),
        ),
        dict(
            group="D1_delta_small",
            tag=f"D1delta_lat4_b{_tag_beta(0.001)}_fb0p10_lr{_tag_lr(3e-4)}_L{_tag_layers([128,256,512])}",
            cfg=_cfg(beta=0.001, free_bits=0.10, **base_cfg),
        ),
        dict(
            group="D1_delta_small",
            tag=f"D1delta_lat4_b{_tag_beta(0.001)}_fb0p0_lr{_tag_lr(3e-4)}_L{_tag_layers([128,256,512])}",
            cfg=_cfg(beta=0.001, free_bits=0.0, **base_cfg),
        ),
        dict(
            group="D1_delta_small",
            tag=f"D1delta_lat4_b{_tag_beta(0.002)}_fb0p0_lr{_tag_lr(3e-4)}_L{_tag_layers([128,256,512])}",
            cfg=_cfg(beta=0.002, free_bits=0.0, **base_cfg),
        ),
    ]


def _preset_delta_residual_smoke() -> List[Dict[str, Any]]:
    """Single-item smoke preset for the explicit residual-target variant."""
    return [item for item in _delta_residual_candidates() if item["group"] == "D0_delta_smoke"]


def _preset_delta_residual_small() -> List[Dict[str, Any]]:
    """4-config exploratory preset for the explicit residual-target variant."""
    return [item for item in _delta_residual_candidates() if item["group"] == "D1_delta_small"]


def _preset_delta_residual_refine() -> List[Dict[str, Any]]:
    """Refinement sweep around the best explicit residual-target region.

    Anchored on the current winner:
      arch_variant=delta_residual
      layer_sizes=[128,256,512]
      free_bits=0.0
      lr=3e-4
      batch_size=16384
      kl_anneal_epochs=80

    Varies only:
      beta       in {0.0005, 0.001}
      latent_dim in {4, 6, 8}
    """
    base = dict(
        arch_variant="delta_residual",
        layer_sizes=[128, 256, 512],
        free_bits=0.0,
        lr=3e-4,
        batch_size=16384,
        kl_anneal_epochs=80,
    )
    grid: List[Dict[str, Any]] = []
    for latent_dim in [4, 6, 8]:
        for beta in [0.0005, 0.001]:
            tag = (
                f"D2delta_lat{latent_dim}_b{_tag_beta(beta)}_fb0p0_"
                f"lr{_tag_lr(3e-4)}_bs16384_anneal80_L{_tag_layers([128,256,512])}"
            )
            grid.append(
                dict(
                    group="D2_delta_refine",
                    tag=tag,
                    cfg=_cfg(
                        latent_dim=latent_dim,
                        beta=beta,
                        **base,
                    ),
                )
            )
    return grid


def _preset_delta_residual_local() -> List[Dict[str, Any]]:
    """Local refinement around the best current delta_residual reference.

    Anchored on the current scientific winner from ``delta_residual_small``:
      latent_dim=4
      beta=0.001
      free_bits=0.0
      layer_sizes=[128,256,512]
      lr=3e-4
      kl_anneal_epochs=80

    Varies only:
      latent_dim in {3, 4, 5, 6}
      batch_size in {8192, 16384}
    """
    base = dict(
        arch_variant="delta_residual",
        layer_sizes=[128, 256, 512],
        beta=0.001,
        free_bits=0.0,
        lr=3e-4,
        kl_anneal_epochs=80,
    )
    grid: List[Dict[str, Any]] = []
    for latent_dim in [3, 4, 5, 6]:
        for batch_size in [8192, 16384]:
            tag = (
                f"D3delta_lat{latent_dim}_b{_tag_beta(0.001)}_fb0p0_"
                f"lr{_tag_lr(3e-4)}_bs{batch_size}_anneal80_"
                f"L{_tag_layers([128,256,512])}"
            )
            grid.append(
                dict(
                    group="D3_delta_local",
                    tag=tag,
                    cfg=_cfg(
                        latent_dim=latent_dim,
                        batch_size=batch_size,
                        **base,
                    ),
                )
            )
    return grid


def _preset_delta_residual_frontier() -> List[Dict[str, Any]]:
    """Frontier sweep around the residual-target trade-off zone.

    Built from the historical cross-reference to avoid exact repetition of the
    already tested `delta_residual` center while probing the boundary between:
      - better signal fidelity (`G1/G2`)
      - better residual-structure fidelity (`G3/G4/G5`)

    Fixed:
      layer_sizes=[128,256,512]
      lr=3e-4
      batch_size=16384
      kl_anneal_epochs=80

    Varies only:
      latent_dim in {4, 5, 6}
      beta       in {0.0007, 0.00085, 0.00115}
      free_bits  in {0.0, 0.02, 0.05}
    """
    base = dict(
        arch_variant="delta_residual",
        layer_sizes=[128, 256, 512],
        lr=3e-4,
        batch_size=16384,
        kl_anneal_epochs=80,
    )
    grid: List[Dict[str, Any]] = []
    for latent_dim in [4, 5, 6]:
        for beta in [0.0007, 0.00085, 0.00115]:
            for free_bits in [0.0, 0.02, 0.05]:
                tag = (
                    f"D4delta_lat{latent_dim}_b{_tag_beta(beta)}_"
                    f"fb{str(free_bits).replace('.','p')}_"
                    f"lr{_tag_lr(3e-4)}_bs16384_anneal80_"
                    f"L{_tag_layers([128,256,512])}"
                )
                grid.append(
                    dict(
                        group="D4_delta_frontier",
                        tag=tag,
                        cfg=_cfg(
                            latent_dim=latent_dim,
                            beta=beta,
                            free_bits=free_bits,
                            **base,
                        ),
                    )
                )
    return grid


def _legacy_2025_candidates() -> List[Dict[str, Any]]:
    """Return the dedicated legacy-2025 candidates."""
    base = dict(
        arch_variant="legacy_2025_zero_y",
        dropout=0.0,
        free_bits=0.0,
        activation="leaky_relu",
    )
    return [
        dict(
            group="L0_legacy2025_smoke",
            tag="L0legacy_lat4_b0p01_fb0p0_lr0p0003_bs1024_anneal3_L32-64",
            cfg=_cfg(
                layer_sizes=[32, 64],
                latent_dim=4,
                beta=0.01,
                lr=3e-4,
                batch_size=1024,
                kl_anneal_epochs=3,
                **base,
            ),
        ),
        dict(
            group="L1_legacy2025_ref",
            tag="L1legacy_lat16_b0p1_fb0p0_lr0p0001_bs4096_anneal50_L32-64-128-256",
            cfg=_cfg(
                layer_sizes=[32, 64, 128, 256],
                latent_dim=16,
                beta=0.1,
                lr=1e-4,
                batch_size=4096,
                kl_anneal_epochs=50,
                **base,
            ),
        ),
    ]


def _preset_legacy2025_smoke() -> List[Dict[str, Any]]:
    """Single-item smoke preset for the legacy 2025 zero-y variant."""
    return [item for item in _legacy_2025_candidates() if item["group"] == "L0_legacy2025_smoke"]


def _preset_legacy2025_ref() -> List[Dict[str, Any]]:
    """Reference preset matching the intended 2025-style benchmark config."""
    return [item for item in _legacy_2025_candidates() if item["group"] == "L1_legacy2025_ref"]


def _preset_legacy2025_batch_sweep() -> List[Dict[str, Any]]:
    """Batch-size sweep around the legacy 2025 reference configuration.

    Keeps the full legacy-ref architecture fixed and varies only ``batch_size``
    so throughput/stability can be compared fairly under the same protocol.
    Ordered from smallest to largest batch to make incremental escalation easy.
    """
    base = dict(
        arch_variant="legacy_2025_zero_y",
        dropout=0.0,
        free_bits=0.0,
        activation="leaky_relu",
        layer_sizes=[32, 64, 128, 256],
        latent_dim=16,
        beta=0.1,
        lr=1e-4,
        kl_anneal_epochs=50,
    )
    batch_sizes = [4096, 8192, 16384, 32768, 65536]
    return [
        dict(
            group="L2_legacy2025_batch_sweep",
            tag=(
                f"L2legacy_lat16_b0p1_fb0p0_lr0p0001_bs{batch_size}"
                "_anneal50_L32-64-128-256"
            ),
            cfg=_cfg(batch_size=batch_size, **base),
        )
        for batch_size in batch_sizes
    ]


def _preset_legacy2025_large() -> List[Dict[str, Any]]:
    """Larger exploratory sweep for the legacy 2025 zero-y variant.

    Intended for reduced-data scientific screening on the 4-current protocol:
    keep the data subset small enough to afford a wider hyperparameter sweep,
    while fixing the batch size to the accepted operational ceiling (8192).
    """
    base = dict(
        arch_variant="legacy_2025_zero_y",
        dropout=0.0,
        free_bits=0.0,
        activation="leaky_relu",
        batch_size=8192,
        lr=1e-4,
        kl_anneal_epochs=50,
    )
    layer_sets = [
        [32, 64, 128, 256],
        [64, 128, 256, 512],
    ]
    latent_dims = [8, 16, 24]
    betas = [0.03, 0.1]
    grid: List[Dict[str, Any]] = []
    for layers in layer_sets:
        for latent_dim in latent_dims:
            for beta in betas:
                tag = (
                    f"L3legacy_lat{latent_dim}_b{_tag_beta(beta)}_fb0p0_"
                    f"lr{_tag_lr(1e-4)}_bs8192_anneal50_L{_tag_layers(layers)}"
                )
                grid.append(
                    dict(
                        group="L3_legacy2025_large",
                        tag=tag,
                        cfg=_cfg(
                            layer_sizes=layers,
                            latent_dim=latent_dim,
                            beta=beta,
                            **base,
                        ),
                    )
                )
    return grid


def _seq_bigru_residual_candidates() -> List[Dict[str, Any]]:
    """seq_bigru_residual grid items: smoke (S0) and small-sweep (S1) configs.

    S0 — smoke-only
    ---------------
    Tiny model (h=16, L=[64,128]) with kl_anneal_epochs=3 for fast CI/end-to-end
    validation.  Not intended for scientific evaluation.

    S1 — exploratory small (seq_residual_small preset)
    ---------------------------------------------------
    Production-scale MLP (L=[128,256,512]) with production kl_anneal_epochs=80.
    Fully-crossed 2×2 design over seq_hidden_size × beta:
      seq_hidden_size: 64, 128   — tests BiGRU capacity as bottleneck
      beta:            0.001, 0.003  — covers the same reference range as G0/G1

    Fixed across all S1 configs: window_size=7, free_bits=0.10, latent_dim=4,
    batch_size=8192, lr=3e-4, seq_num_layers=1, seq_bidirectional=True.

    Protocol constraint (seq_bigru_residual only)
    ---------------------------------------------
    balanced_blocks data-reduction is INCOMPATIBLE with windowed architectures
    because non-contiguous block selection breaks temporal context.
    All seq runs must be launched with --no_data_reduction (or equivalent
    runtime override no_data_reduction=True).  This is enforced as a hard
    guard in src/training/pipeline.py.
    """

    # ------------------------------------------------------------------ #
    # Shared base for all seq configs                                      #
    # ------------------------------------------------------------------ #
    def _scfg_base(**kwargs):
        base = dict(
            activation="leaky_relu",
            lr=3e-4,
            dropout=0.0,
            arch_variant="seq_bigru_residual",
            window_size=7,
            window_stride=1,
            window_pad_mode="edge",
            seq_num_layers=1,
            seq_bidirectional=True,
        )
        base.update(kwargs)
        return base

    # ------------------------------------------------------------------ #
    # S0 — smoke (tiny model, fast epochs, CI only)                        #
    # ------------------------------------------------------------------ #
    s0 = [
        dict(
            group="S0_seq_smoke",
            tag="S0seq_W7_h16_lat4_b0p001_fb0p10_lr0p0003_L64-128",
            cfg=_scfg_base(
                layer_sizes=[64, 128],
                latent_dim=4,
                beta=0.001,
                free_bits=0.10,
                seq_hidden_size=16,
                kl_anneal_epochs=3,
                batch_size=8192,
            ),
        ),
    ]

    # ------------------------------------------------------------------ #
    # S1 — small exploratory sweep (production scale, 4 configs)           #
    # 2×2 fully-crossed: seq_hidden_size={64,128} × beta={0.001,0.003}    #
    # ------------------------------------------------------------------ #
    s1 = []
    for h in [64, 128]:
        for beta in [0.001, 0.003]:
            tag = (
                f"S1seq_W7_h{h}_lat4_b{_tag_beta(beta)}"
                f"_fb0p10_lr{_tag_lr(3e-4)}_L{_tag_layers([128, 256, 512])}"
            )
            s1.append(
                dict(
                    group="S1_seq_small",
                    tag=tag,
                    cfg=_scfg_base(
                        layer_sizes=[128, 256, 512],
                        latent_dim=4,
                        beta=beta,
                        free_bits=0.10,
                        seq_hidden_size=h,
                        kl_anneal_epochs=80,
                        batch_size=8192,
                    ),
                )
            )

    return s0 + s1


def _preset_seq_residual_smoke() -> List[Dict[str, Any]]:
    """Single-item smoke preset for seq_bigru_residual end-to-end validation."""
    all_seq = _seq_bigru_residual_candidates()
    return [item for item in all_seq if item["group"] == "S0_seq_smoke"]


def _preset_seq_residual_small() -> List[Dict[str, Any]]:
    """4-config exploratory preset for seq_bigru_residual (S1 group only).

    Fully-crossed 2×2 design: seq_hidden_size={64,128} × beta={0.001,0.003}.
    Uses production-scale MLP ([128,256,512]) and kl_anneal_epochs=80.

    Requires --no_data_reduction (balanced_blocks incompatible with windowing).
    """
    all_seq = _seq_bigru_residual_candidates()
    return [item for item in all_seq if item["group"] == "S1_seq_small"]


def _preset_seq_residual_mmd() -> List[Dict[str, Any]]:
    """MMD-augmented seq_bigru_residual sweep (Etapa C — G6 investigation).

    Tests lambda_mmd in {0.1, 0.5, 1.0} with the two best configs from
    seq_residual_small (h=64 beta=0.003 and h=64 beta=0.001).

    The MMD auxiliary loss adds λ·MMD²(residuals_real, residuals_gen) to
    the ELBO, directly optimising what G6 measures.

    Requires --no_data_reduction.
    """
    items = []
    for lam in [0.1, 0.5, 1.0]:
        for beta in [0.001, 0.003]:
            lam_tag = str(lam).replace(".", "p")
            beta_tag = _tag_beta(beta)
            items.append({
                "group": "S2_seq_mmd",
                "tag": f"S2seq_W7_h64_lat4_b{beta_tag}_lmmd{lam_tag}_fb0p10_lr0p0003_L128-256-512",
                "cfg": _cfg(
                    arch_variant="seq_bigru_residual",
                    layer_sizes=[128, 256, 512],
                    latent_dim=4,
                    beta=beta,
                    free_bits=0.1,
                    lr=3e-4,
                    batch_size=8192,
                    kl_anneal_epochs=80,
                    seq_hidden_size=64,
                    seq_num_layers=1,
                    seq_bidirectional=True,
                    window_size=7,
                    window_stride=1,
                    window_pad_mode="edge",
                    lambda_mmd=lam,
                ),
            })
    return items


def _preset_seq_residual_mmd_final() -> List[Dict[str, Any]]:
    """Single-grid validation run with the Etapa C champion (λ_mmd=1.0, β=0.001).

    Used to produce a clean summary_by_regime.csv with all six gates evaluated
    after the G5 recalibration (delta_jb_stat_rel < 0.20).
    """
    return [{
        "group": "S2_seq_mmd_final",
        "tag": "S2seq_W7_h64_lat4_b0p001_lmmd1p0_fb0p10_lr0p0003_L128-256-512",
        "cfg": _cfg(
            arch_variant="seq_bigru_residual",
            layer_sizes=[128, 256, 512],
            latent_dim=4,
            beta=0.001,
            free_bits=0.1,
            lr=3e-4,
            batch_size=8192,
            kl_anneal_epochs=80,
            seq_hidden_size=64,
            seq_num_layers=1,
            seq_bidirectional=True,
            window_size=7,
            window_stride=1,
            window_pad_mode="edge",
            lambda_mmd=1.0,
        ),
    }]


def _preset_seq_residual_nightly() -> List[Dict[str, Any]]:
    """Larger overnight sweep around the current seq_bigru_residual winner.

    Focuses only on the strongest family observed so far:
      - latent_dim fixed at 4 (active_dim_ratio has been healthy)
      - window_size fixed at 7
      - production MLP kept at [128,256,512]
      - batch_size fixed at 8192
      - free_bits fixed at 0.10

    Overnight variables:
      seq_hidden_size in {64, 96, 128}
      beta            in {0.001, 0.003}
      lambda_mmd      in {0.25, 0.5, 0.75, 1.0}

    Total: 3 × 2 × 4 = 24 runs.

    Requires --no_data_reduction.
    """
    grid: List[Dict[str, Any]] = []
    for seq_hidden_size in [64, 96, 128]:
        for beta in [0.001, 0.003]:
            beta_tag = _tag_beta(beta)
            for lam in [0.25, 0.5, 0.75, 1.0]:
                lam_tag = str(lam).replace(".", "p")
                grid.append(
                    dict(
                        group="S3_seq_nightly",
                        tag=(
                            f"S3seq_W7_h{seq_hidden_size}_lat4_b{beta_tag}_"
                            f"lmmd{lam_tag}_fb0p10_lr0p0003_L128-256-512"
                        ),
                        cfg=_cfg(
                            arch_variant="seq_bigru_residual",
                            layer_sizes=[128, 256, 512],
                            latent_dim=4,
                            beta=beta,
                            free_bits=0.10,
                            lr=3e-4,
                            batch_size=8192,
                            kl_anneal_epochs=80,
                            seq_hidden_size=seq_hidden_size,
                            seq_num_layers=1,
                            seq_bidirectional=True,
                            window_size=7,
                            window_stride=1,
                            window_pad_mode="edge",
                            lambda_mmd=lam,
                        ),
                    )
                )
    return grid


def _preset_seq_investigation_large() -> List[Dict[str, Any]]:
    """Focused follow-up sweep around the current multi-regime seq winner.

    Design goals:
      - keep the strongest point-wise anchor for calibration
      - expand temporal context to test the hard 0.8 m regimes
      - increase seq capacity moderately without exploding runtime
      - probe slightly stronger MMD regularization around the current winner
      - keep latent_dim/free_bits fixed because recent runs showed healthy
        active-dimension usage and no posterior-collapse signal

    Structure:
      - Block A: 12 seq runs
          window_size    in {7, 9, 11}
          seq_hidden     in {64, 96}
          lambda_mmd     in {1.0, 1.25}
          beta           fixed at 0.003
      - Block B: 4 seq runs
          window_size    fixed at 9
          seq_hidden     fixed at 96
          lambda_mmd     in {1.0, 1.25}
          beta           in {0.002, 0.004}
      - Block C: 1 point-wise anchor

    Total: 17 runs.

    Requires --no_data_reduction.
    """

    def _seq_cfg(*, beta: float, window_size: int, seq_hidden_size: int, lambda_mmd: float) -> Dict[str, Any]:
        return _cfg(
            arch_variant="seq_bigru_residual",
            layer_sizes=[128, 256, 512],
            latent_dim=4,
            beta=beta,
            free_bits=0.10,
            lr=3e-4,
            batch_size=8192,
            kl_anneal_epochs=80,
            window_size=window_size,
            window_stride=1,
            window_pad_mode="edge",
            seq_hidden_size=seq_hidden_size,
            seq_num_layers=1,
            seq_bidirectional=True,
            lambda_mmd=lambda_mmd,
        )

    grid: List[Dict[str, Any]] = []

    for window_size in [7, 9, 11]:
        for seq_hidden_size in [64, 96]:
            for lambda_mmd in [1.0, 1.25]:
                lam_tag = str(lambda_mmd).replace(".", "p")
                grid.append(
                    dict(
                        group="S4_seq_investigation",
                        tag=(
                            f"S4seq_W{window_size}_h{seq_hidden_size}_lat4_b0p003_"
                            f"lmmd{lam_tag}_fb0p10_lr0p0003_L128-256-512"
                        ),
                        cfg=_seq_cfg(
                            beta=0.003,
                            window_size=window_size,
                            seq_hidden_size=seq_hidden_size,
                            lambda_mmd=lambda_mmd,
                        ),
                    )
                )

    for beta in [0.002, 0.004]:
        beta_tag = _tag_beta(beta)
        for lambda_mmd in [1.0, 1.25]:
            lam_tag = str(lambda_mmd).replace(".", "p")
            grid.append(
                dict(
                    group="S4_seq_investigation",
                    tag=(
                        f"S4seq_W9_h96_lat4_b{beta_tag}_lmmd{lam_tag}_"
                        f"fb0p10_lr0p0003_L128-256-512"
                    ),
                    cfg=_seq_cfg(
                        beta=beta,
                        window_size=9,
                        seq_hidden_size=96,
                        lambda_mmd=lambda_mmd,
                    ),
                )
            )

    grid.append(
        dict(
            group="S4_seq_investigation",
            tag="COPT_lat6_b0p001_fb0p0_lr0p0001_bs16384_anneal120_L64-128-256",
            cfg=_cfg(
                arch_variant="delta_residual",
                layer_sizes=[64, 128, 256],
                latent_dim=6,
                beta=0.001,
                free_bits=0.0,
                lr=1e-4,
                batch_size=16384,
                kl_anneal_epochs=120,
            ),
        )
    )

    return grid


def _preset_seq_stability_mmd_focus() -> List[Dict[str, Any]]:
    """Focused recovery sweep around the strongest current seq champion.

    Rationale from the last two multi-regime runs:
      - ``W7_h64_beta0.003_lambda_mmd=1.25`` remains the best protocol-level
        reference even after the larger context/capacity sweep.
      - Increasing ``seq_hidden_size`` to 96 improved train-side ranking but
        did not improve the final protocol scoreboard.
      - The remaining blocker is still distribution fidelity at 0.8 m, while
        train diagnostics keep flagging instability.

    Hypothesis:
      - keep the best architecture family fixed (W7 / h64 / latent4)
      - test whether a lower initial learning rate stabilises the run
      - probe slightly stronger MMD pressure only under the lower-LR variants

    Total: 6 runs.

    Requires --no_data_reduction.
    """

    def _seq_cfg(*, lr: float, lambda_mmd: float) -> Dict[str, Any]:
        return _cfg(
            arch_variant="seq_bigru_residual",
            layer_sizes=[128, 256, 512],
            latent_dim=4,
            beta=0.003,
            free_bits=0.10,
            lr=lr,
            batch_size=8192,
            kl_anneal_epochs=80,
            window_size=7,
            window_stride=1,
            window_pad_mode="edge",
            seq_hidden_size=64,
            seq_num_layers=1,
            seq_bidirectional=True,
            lambda_mmd=lambda_mmd,
        )

    grid: List[Dict[str, Any]] = []
    candidates = [
        (3e-4, 1.25),
        (2e-4, 1.25),
        (1.5e-4, 1.25),
        (2e-4, 1.50),
        (1.5e-4, 1.50),
        (1.5e-4, 1.75),
    ]
    for lr, lambda_mmd in candidates:
        lam_tag = str(lambda_mmd).replace(".", "p")
        grid.append(
            dict(
                group="S5_seq_stability_mmd",
                tag=(
                    f"S5seq_W7_h64_lat4_b0p003_lmmd{lam_tag}_"
                    f"fb0p10_lr{_tag_lr(lr)}_L128-256-512"
                ),
                cfg=_seq_cfg(lr=lr, lambda_mmd=lambda_mmd),
            )
        )
    return grid


def _preset_seq_overnight_12h() -> List[Dict[str, Any]]:
    """12-hour overnight sweep around the strongest seq family.

    Historical rationale:
      - before ``exp_20260324_023558``, ``exp_20260322_193738`` was the best
        protocol-level seq result with ``W7_h64_beta0.003_lambda_mmd=1.25``.
      - ``exp_20260323_210309`` showed that simply increasing hidden size to
        96 does not improve the final protocol scoreboard.
      - this preset was built to:
          1. stabilise the winning W7/h64 family with lower initial LR
          2. probe stronger MMD pressure for the hard 0.8 m regimes
          3. keep only a small low-LR larger-context block as a hedge

    Current status:
      - the preset already produced the new protocol winner in
        ``exp_20260324_023558``.
      - use ``seq_finish_0p8m`` as the preferred next step when the goal is to
        close the last two failing 0.8 m regimes instead of running another
        broad overnight sweep.

    Structure:
      - Block A: 12 runs (core winner family)
          beta           fixed at 0.003
          window_size    fixed at 7
          seq_hidden     fixed at 64
          lr             in {3e-4, 2e-4, 1.5e-4}
          lambda_mmd     in {1.0, 1.25, 1.5, 1.75}
      - Block B: 12 runs (beta refinement around the winner)
          beta           in {0.001, 0.002, 0.004}
          window_size    fixed at 7
          seq_hidden     fixed at 64
          lr             in {2e-4, 1.5e-4}
          lambda_mmd     in {1.25, 1.5}
      - Block C: 4 runs (larger-context hedge, low-LR only)
          beta           fixed at 0.003
          window_size    fixed at 9
          seq_hidden     fixed at 96
          lr             in {2e-4, 1.5e-4}
          lambda_mmd     in {1.25, 1.5}

    Total: 28 runs.

    On the recent A6000-class setup, this is intended to land roughly in the
    10–12 hour range including final protocol evaluation.

    Requires --no_data_reduction.
    """

    def _seq_cfg(
        *,
        beta: float,
        window_size: int,
        seq_hidden_size: int,
        lr: float,
        lambda_mmd: float,
    ) -> Dict[str, Any]:
        return _cfg(
            arch_variant="seq_bigru_residual",
            layer_sizes=[128, 256, 512],
            latent_dim=4,
            beta=beta,
            free_bits=0.10,
            lr=lr,
            batch_size=8192,
            kl_anneal_epochs=80,
            window_size=window_size,
            window_stride=1,
            window_pad_mode="edge",
            seq_hidden_size=seq_hidden_size,
            seq_num_layers=1,
            seq_bidirectional=True,
            lambda_mmd=lambda_mmd,
        )

    grid: List[Dict[str, Any]] = []

    for lr in [3e-4, 2e-4, 1.5e-4]:
        for lambda_mmd in [1.0, 1.25, 1.5, 1.75]:
            lam_tag = str(lambda_mmd).replace(".", "p")
            grid.append(
                dict(
                    group="S6_seq_overnight",
                    tag=(
                        f"S6seq_W7_h64_lat4_b0p003_lmmd{lam_tag}_"
                        f"fb0p10_lr{_tag_lr(lr)}_L128-256-512"
                    ),
                    cfg=_seq_cfg(
                        beta=0.003,
                        window_size=7,
                        seq_hidden_size=64,
                        lr=lr,
                        lambda_mmd=lambda_mmd,
                    ),
                )
            )

    for beta in [0.001, 0.002, 0.004]:
        beta_tag = _tag_beta(beta)
        for lr in [2e-4, 1.5e-4]:
            for lambda_mmd in [1.25, 1.5]:
                lam_tag = str(lambda_mmd).replace(".", "p")
                grid.append(
                    dict(
                        group="S6_seq_overnight",
                        tag=(
                            f"S6seq_W7_h64_lat4_b{beta_tag}_lmmd{lam_tag}_"
                            f"fb0p10_lr{_tag_lr(lr)}_L128-256-512"
                        ),
                        cfg=_seq_cfg(
                            beta=beta,
                            window_size=7,
                            seq_hidden_size=64,
                            lr=lr,
                            lambda_mmd=lambda_mmd,
                        ),
                    )
                )

    for lr in [2e-4, 1.5e-4]:
        for lambda_mmd in [1.25, 1.5]:
            lam_tag = str(lambda_mmd).replace(".", "p")
            grid.append(
                dict(
                    group="S6_seq_overnight",
                    tag=(
                        f"S6seq_W9_h96_lat4_b0p003_lmmd{lam_tag}_"
                        f"fb0p10_lr{_tag_lr(lr)}_L128-256-512"
                    ),
                    cfg=_seq_cfg(
                        beta=0.003,
                        window_size=9,
                        seq_hidden_size=96,
                        lr=lr,
                        lambda_mmd=lambda_mmd,
                    ),
                )
            )

    return grid


def _preset_seq_replay_axis_diagnostics() -> List[Dict[str, Any]]:
    """Replay the strongest seq candidates under the new axis-wise diagnostics.

    Purpose:
      - regenerate runs with the new per-axis residual metrics/dashboard
      - compare the best protocol winner against the strongest nearby variants
      - keep the replay small enough to read candidate-by-candidate

    Selected candidates:
      - former best protocol winner (`exp_20260322_193738`)
      - best higher-capacity W7 variant (`exp_20260323_210309`)
      - best longer-context W9/h96 variant from the same comparison run
      - historical S2 reference that opened the MMD path

    Total: 4 runs.
    Requires --no_data_reduction.
    """
    return [
        dict(
            group="S7_seq_replay_axis",
            tag="S4seq_W7_h64_lat4_b0p003_lmmd1p25_fb0p10_lr0p0003_L128-256-512",
            cfg=_cfg(
                arch_variant="seq_bigru_residual",
                layer_sizes=[128, 256, 512],
                latent_dim=4,
                beta=0.003,
                free_bits=0.10,
                lr=3e-4,
                batch_size=8192,
                kl_anneal_epochs=80,
                window_size=7,
                window_stride=1,
                window_pad_mode="edge",
                seq_hidden_size=64,
                seq_num_layers=1,
                seq_bidirectional=True,
                lambda_mmd=1.25,
            ),
        ),
        dict(
            group="S7_seq_replay_axis",
            tag="S4seq_W7_h96_lat4_b0p003_lmmd1p25_fb0p10_lr0p0003_L128-256-512",
            cfg=_cfg(
                arch_variant="seq_bigru_residual",
                layer_sizes=[128, 256, 512],
                latent_dim=4,
                beta=0.003,
                free_bits=0.10,
                lr=3e-4,
                batch_size=8192,
                kl_anneal_epochs=80,
                window_size=7,
                window_stride=1,
                window_pad_mode="edge",
                seq_hidden_size=96,
                seq_num_layers=1,
                seq_bidirectional=True,
                lambda_mmd=1.25,
            ),
        ),
        dict(
            group="S7_seq_replay_axis",
            tag="S4seq_W9_h96_lat4_b0p003_lmmd1p25_fb0p10_lr0p0003_L128-256-512",
            cfg=_cfg(
                arch_variant="seq_bigru_residual",
                layer_sizes=[128, 256, 512],
                latent_dim=4,
                beta=0.003,
                free_bits=0.10,
                lr=3e-4,
                batch_size=8192,
                kl_anneal_epochs=80,
                window_size=9,
                window_stride=1,
                window_pad_mode="edge",
                seq_hidden_size=96,
                seq_num_layers=1,
                seq_bidirectional=True,
                lambda_mmd=1.25,
            ),
        ),
        dict(
            group="S7_seq_replay_axis",
            tag="S2seq_W7_h64_lat4_b0p001_lmmd1p0_fb0p10_lr0p0003_L128-256-512",
            cfg=_cfg(
                arch_variant="seq_bigru_residual",
                layer_sizes=[128, 256, 512],
                latent_dim=4,
                beta=0.001,
                free_bits=0.10,
                lr=3e-4,
                batch_size=8192,
                kl_anneal_epochs=80,
                window_size=7,
                window_stride=1,
                window_pad_mode="edge",
                seq_hidden_size=64,
                seq_num_layers=1,
                seq_bidirectional=True,
                lambda_mmd=1.0,
            ),
        ),
    ]


def _preset_seq_finish_0p8m() -> List[Dict[str, Any]]:
    """Focused finishing grid around the new 10/12 protocol winner.

    Evidence after ``exp_20260324_023558``:
      - the new winner is now
        ``S6seq_W7_h64_lat4_b0p003_lmmd1p75_fb0p10_lr0p0003_L128-256-512``
      - all 1.0 m and 1.5 m regimes pass
      - only ``0.8 m / 100 mA`` and ``0.8 m / 300 mA`` remain failing
      - h96 / W9 variants are no longer competitive enough to justify budget

    Hypothesis:
      - keep the winning W7/h64 family fixed
      - push MMD slightly beyond 1.75 to attack G6 at 0.8 m
      - use lower-LR / slightly higher-beta variants as stability hedges for G3/G5

    Total: 6 runs.
    Requires --no_data_reduction.
    """

    def _seq_cfg(*, beta: float, lr: float, lambda_mmd: float) -> Dict[str, Any]:
        return _cfg(
            arch_variant="seq_bigru_residual",
            layer_sizes=[128, 256, 512],
            latent_dim=4,
            beta=beta,
            free_bits=0.10,
            lr=lr,
            batch_size=8192,
            kl_anneal_epochs=80,
            window_size=7,
            window_stride=1,
            window_pad_mode="edge",
            seq_hidden_size=64,
            seq_num_layers=1,
            seq_bidirectional=True,
            lambda_mmd=lambda_mmd,
        )

    return [
        dict(
            group="S8_seq_finish_0p8m",
            tag="S8seq_W7_h64_lat4_b0p003_lmmd1p75_fb0p10_lr0p0003_L128-256-512",
            cfg=_seq_cfg(beta=0.003, lr=3e-4, lambda_mmd=1.75),
        ),
        dict(
            group="S8_seq_finish_0p8m",
            tag="S8seq_W7_h64_lat4_b0p003_lmmd2p0_fb0p10_lr0p0003_L128-256-512",
            cfg=_seq_cfg(beta=0.003, lr=3e-4, lambda_mmd=2.0),
        ),
        dict(
            group="S8_seq_finish_0p8m",
            tag="S8seq_W7_h64_lat4_b0p003_lmmd2p0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(beta=0.003, lr=2e-4, lambda_mmd=2.0),
        ),
        dict(
            group="S8_seq_finish_0p8m",
            tag="S8seq_W7_h64_lat4_b0p004_lmmd1p5_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(beta=0.004, lr=2e-4, lambda_mmd=1.5),
        ),
        dict(
            group="S8_seq_finish_0p8m",
            tag="S8seq_W7_h64_lat4_b0p004_lmmd1p75_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(beta=0.004, lr=2e-4, lambda_mmd=1.75),
        ),
        dict(
            group="S8_seq_finish_0p8m",
            tag="S8seq_W7_h64_lat4_b0p002_lmmd1p5_fb0p10_lr0p00015_L128-256-512",
            cfg=_seq_cfg(beta=0.002, lr=1.5e-4, lambda_mmd=1.5),
        ),
    ]


def _preset_seq_sampled_mmd_compare() -> List[Dict[str, Any]]:
    """Minimal causal comparison for sampled-residual MMD.

    Goal:
      - keep the best known seq family fixed
      - compare the historical ``mean_residual`` MMD objective against the new
        ``sampled_residual`` objective
      - include one low-LR hedge only after the objective changes

    Total: 3 runs.
    Requires --no_data_reduction.
    """

    def _seq_cfg(*, lr: float, mmd_mode: str) -> Dict[str, Any]:
        return _cfg(
            arch_variant="seq_bigru_residual",
            layer_sizes=[128, 256, 512],
            latent_dim=4,
            beta=0.003,
            free_bits=0.10,
            lr=lr,
            batch_size=8192,
            kl_anneal_epochs=80,
            window_size=7,
            window_stride=1,
            window_pad_mode="edge",
            seq_hidden_size=64,
            seq_num_layers=1,
            seq_bidirectional=True,
            lambda_mmd=1.75,
            mmd_mode=mmd_mode,
        )

    return [
        dict(
            group="S9_seq_sampled_mmd",
            tag="S9seq_W7_h64_lat4_b0p003_lmmd1p75_mmdmean_fb0p10_lr0p0003_L128-256-512",
            cfg=_seq_cfg(lr=3e-4, mmd_mode="mean_residual"),
        ),
        dict(
            group="S9_seq_sampled_mmd",
            tag="S9seq_W7_h64_lat4_b0p003_lmmd1p75_mmdsample_fb0p10_lr0p0003_L128-256-512",
            cfg=_seq_cfg(lr=3e-4, mmd_mode="sampled_residual"),
        ),
        dict(
            group="S9_seq_sampled_mmd",
            tag="S9seq_W7_h64_lat4_b0p003_lmmd1p75_mmdsample_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lr=2e-4, mmd_mode="sampled_residual"),
        ),
    ]


def _preset_seq_hybrid_loss_smoke() -> List[Dict[str, Any]]:
    """Single-candidate smoke for the Gaussian hybrid objective."""

    return [
        dict(
            group="S10_seq_hybrid_smoke",
            tag="S10seq_W7_h64_lat4_b0p003_lmmd1p75_axis0p1_psd0p02_noshuf_fb0p10_lr0p0003_L128-256-512",
            cfg=_cfg(
                arch_variant="seq_bigru_residual",
                layer_sizes=[128, 256, 512],
                latent_dim=4,
                beta=0.003,
                free_bits=0.10,
                lr=3e-4,
                batch_size=4096,
                kl_anneal_epochs=80,
                window_size=7,
                window_stride=1,
                window_pad_mode="edge",
                seq_hidden_size=64,
                seq_num_layers=1,
                seq_bidirectional=True,
                lambda_mmd=1.75,
                mmd_mode="mean_residual",
                lambda_axis=0.10,
                lambda_psd=0.02,
                decoder_distribution="gaussian",
                mdn_components=1,
                shuffle_train_batches=False,
            ),
        ),
    ]


def _preset_seq_mdn_smoke() -> List[Dict[str, Any]]:
    """Single-candidate smoke for the seq MDN decoder with hybrid diagnostics."""

    return [
        dict(
            group="S11_seq_mdn_smoke",
            tag="S11seq_W7_h64_lat4_mdn3_b0p003_axis0p1_psd0p02_noshuf_fb0p10_lr0p0003_L128-256-512",
            cfg=_cfg(
                arch_variant="seq_bigru_residual",
                layer_sizes=[128, 256, 512],
                latent_dim=4,
                beta=0.003,
                free_bits=0.10,
                lr=3e-4,
                batch_size=4096,
                kl_anneal_epochs=80,
                window_size=7,
                window_stride=1,
                window_pad_mode="edge",
                seq_hidden_size=64,
                seq_num_layers=1,
                seq_bidirectional=True,
                lambda_mmd=0.0,
                mmd_mode="mean_residual",
                lambda_axis=0.10,
                lambda_psd=0.02,
                decoder_distribution="mdn",
                mdn_components=3,
                shuffle_train_batches=False,
            ),
        ),
    ]


def _preset_seq_mdn_proof() -> List[Dict[str, Any]]:
    """Focused proof run for the seq MDN line on the full protocol."""

    def _seq_cfg(*, mdn_components: int) -> Dict[str, Any]:
        return _cfg(
            arch_variant="seq_bigru_residual",
            layer_sizes=[128, 256, 512],
            latent_dim=4,
            beta=0.003,
            free_bits=0.10,
            lr=3e-4,
            batch_size=4096,
            kl_anneal_epochs=80,
            window_size=7,
            window_stride=1,
            window_pad_mode="edge",
            seq_hidden_size=64,
            seq_num_layers=1,
            seq_bidirectional=True,
            lambda_mmd=0.0,
            mmd_mode="mean_residual",
            lambda_axis=0.10,
            lambda_psd=0.02,
            decoder_distribution="mdn",
            mdn_components=mdn_components,
            shuffle_train_batches=False,
        )

    return [
        dict(
            group="S12_seq_mdn_proof",
            tag="S12seq_W7_h64_lat4_mdn3_b0p003_axis0p1_psd0p02_noshuf_fb0p10_lr0p0003_L128-256-512",
            cfg=_seq_cfg(mdn_components=3),
        ),
        dict(
            group="S12_seq_mdn_proof",
            tag="S12seq_W7_h64_lat4_mdn5_b0p003_axis0p1_psd0p02_noshuf_fb0p10_lr0p0003_L128-256-512",
            cfg=_seq_cfg(mdn_components=5),
        ),
    ]


def _preset_seq_mdn_conservative_proof() -> List[Dict[str, Any]]:
    """Conservative MDN retry after the variance-inflation failure.

    Design choices:
      - keep only ``mdn3`` to reduce mixture instability
      - drop PSD loss entirely
      - reduce axis loss by 10x
      - lower LR to 2e-4
      - keep one lightly anchored candidate with small MMD
    """

    def _seq_cfg(*, lambda_mmd: float) -> Dict[str, Any]:
        return _cfg(
            arch_variant="seq_bigru_residual",
            layer_sizes=[128, 256, 512],
            latent_dim=4,
            beta=0.003,
            free_bits=0.10,
            lr=2e-4,
            batch_size=4096,
            kl_anneal_epochs=80,
            window_size=7,
            window_stride=1,
            window_pad_mode="edge",
            seq_hidden_size=64,
            seq_num_layers=1,
            seq_bidirectional=True,
            lambda_mmd=lambda_mmd,
            mmd_mode="mean_residual",
            lambda_axis=0.01,
            lambda_psd=0.0,
            decoder_distribution="mdn",
            mdn_components=3,
            shuffle_train_batches=True,
        )

    return [
        dict(
            group="S13_seq_mdn_conservative",
            tag="S13seq_W7_h64_lat4_mdn3_b0p003_axis0p01_psd0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_mmd=0.0),
        ),
        dict(
            group="S13_seq_mdn_conservative",
            tag="S13seq_W7_h64_lat4_mdn3_b0p003_lmmd0p25_axis0p01_psd0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_mmd=0.25),
        ),
    ]


def _preset_seq_mdn_exploratory_quick() -> List[Dict[str, Any]]:
    """Exploratory quick grid around the stable conservative MDN line.

    This preset intentionally keeps the full 12-regime protocol but is meant to
    be run with sample caps in the CLI. The search is centered on the current
    best stable MDN configuration:

      - ``mdn3``
      - ``beta=0.003``
      - ``lr=2e-4``
      - ``lambda_axis=0.01``
      - ``lambda_mmd=0.25``

    Knobs explored:
      - stronger MMD anchor for G6 (`0.50`)
      - slightly stronger axis shaping (`0.02`)
      - lower beta (`0.002`) to free more latent capacity
      - lower LR (`1.5e-4`) for stability
      - slightly richer mixture (`mdn4`) without jumping to mdn5
    """

    def _seq_cfg(
        *,
        mdn_components: int = 3,
        beta: float = 0.003,
        lr: float = 2e-4,
        lambda_axis: float = 0.01,
        lambda_mmd: float = 0.25,
    ) -> Dict[str, Any]:
        return _cfg(
            arch_variant="seq_bigru_residual",
            layer_sizes=[128, 256, 512],
            latent_dim=4,
            beta=beta,
            free_bits=0.10,
            lr=lr,
            batch_size=4096,
            kl_anneal_epochs=80,
            window_size=7,
            window_stride=1,
            window_pad_mode="edge",
            seq_hidden_size=64,
            seq_num_layers=1,
            seq_bidirectional=True,
            lambda_mmd=lambda_mmd,
            mmd_mode="mean_residual",
            lambda_axis=lambda_axis,
            lambda_psd=0.0,
            decoder_distribution="mdn",
            mdn_components=mdn_components,
            shuffle_train_batches=True,
        )

    return [
        dict(
            group="S14_seq_mdn_explore",
            tag="S14seq_W7_h64_lat4_mdn3_b0p003_lmmd0p25_axis0p01_psd0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(),
        ),
        dict(
            group="S14_seq_mdn_explore",
            tag="S14seq_W7_h64_lat4_mdn3_b0p003_lmmd0p5_axis0p01_psd0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_mmd=0.50),
        ),
        dict(
            group="S14_seq_mdn_explore",
            tag="S14seq_W7_h64_lat4_mdn3_b0p003_lmmd0p25_axis0p02_psd0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_axis=0.02),
        ),
        dict(
            group="S14_seq_mdn_explore",
            tag="S14seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_psd0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(beta=0.002),
        ),
        dict(
            group="S14_seq_mdn_explore",
            tag="S14seq_W7_h64_lat4_mdn3_b0p003_lmmd0p25_axis0p01_psd0_fb0p10_lr0p00015_L128-256-512",
            cfg=_seq_cfg(lr=1.5e-4),
        ),
        dict(
            group="S14_seq_mdn_explore",
            tag="S14seq_W7_h64_lat4_mdn4_b0p003_lmmd0p25_axis0p01_psd0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(mdn_components=4),
        ),
    ]


def _preset_seq_mdn_g5_exploratory_quick() -> List[Dict[str, Any]]:
    """Focused MDN quick sweep targeting the remaining G5 failures.

    Anchored on the current best quick MDN candidate:

      - ``mdn3``
      - ``beta=0.002``
      - ``free_bits=0.10``
      - ``lr=2e-4``
      - ``lambda_axis=0.01``
      - ``lambda_mmd=0.25``

    Diagnostics from ``exp_20260325_230938`` indicate:
      - all remaining failures are now G5-only
      - G6 already passes all regimes
      - the mismatch is concentrated in near-range marginal shape, especially
        the JB-relative error on ``I`` at ``0.8 m``

    So this sweep only explores shape-sensitive knobs:
      - lower ``beta`` to free a bit more latent capacity
      - reduce ``free_bits`` to avoid over-regularizing the posterior
      - slightly relax axis shaping (`0.005`) since `0.02` was too strong
      - try an intermediate MMD weight (`0.35`) without reopening the full MMD sweep
      - test ``mdn2`` as a lighter mixture that may reduce tail overfitting
    """

    def _seq_cfg(
        *,
        mdn_components: int = 3,
        beta: float = 0.002,
        free_bits: float = 0.10,
        lr: float = 2e-4,
        lambda_axis: float = 0.01,
        lambda_mmd: float = 0.25,
    ) -> Dict[str, Any]:
        return _cfg(
            arch_variant="seq_bigru_residual",
            layer_sizes=[128, 256, 512],
            latent_dim=4,
            beta=beta,
            free_bits=free_bits,
            lr=lr,
            batch_size=4096,
            kl_anneal_epochs=80,
            window_size=7,
            window_stride=1,
            window_pad_mode="edge",
            seq_hidden_size=64,
            seq_num_layers=1,
            seq_bidirectional=True,
            lambda_mmd=lambda_mmd,
            mmd_mode="mean_residual",
            lambda_axis=lambda_axis,
            lambda_psd=0.0,
            decoder_distribution="mdn",
            mdn_components=mdn_components,
            shuffle_train_batches=True,
        )

    return [
        dict(
            group="S15_seq_mdn_g5",
            tag="S15seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_psd0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(),
        ),
        dict(
            group="S15_seq_mdn_g5",
            tag="S15seq_W7_h64_lat4_mdn3_b0p0015_lmmd0p25_axis0p01_psd0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(beta=0.0015),
        ),
        dict(
            group="S15_seq_mdn_g5",
            tag="S15seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_psd0_fb0p05_lr0p0002_L128-256-512",
            cfg=_seq_cfg(free_bits=0.05),
        ),
        dict(
            group="S15_seq_mdn_g5",
            tag="S15seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p005_psd0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_axis=0.005),
        ),
        dict(
            group="S15_seq_mdn_g5",
            tag="S15seq_W7_h64_lat4_mdn3_b0p002_lmmd0p35_axis0p01_psd0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_mmd=0.35),
        ),
        dict(
            group="S15_seq_mdn_g5",
            tag="S15seq_W7_h64_lat4_mdn2_b0p002_lmmd0p25_axis0p01_psd0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(mdn_components=2),
        ),
    ]


def _preset_seq_mdn_g5_broader_quick() -> List[Dict[str, Any]]:
    """Broader MDN quick sweep after the first targeted G5 retry.

    What the last two runs established:
      - ``exp_20260325_230938`` is still the best MDN quick anchor
      - lowering ``beta`` to ``0.0015`` improved train-side fit and some G5
        behavior, but weakened G6 on near-range regimes
      - increasing axis pressure or changing mixture count was not the right
        next move

    This broader sweep therefore focuses on the interaction:
      - ``beta`` in {0.0020, 0.0018, 0.0015}
      - ``lambda_mmd`` in {0.25, 0.35, 0.50}
      - ``lambda_axis`` in {0.01, 0.005} only for the lower-beta line

    The goal is explicit:
      - preserve the S14/S15 stability on 1.0 m and 1.5 m
      - recover the all-regime G6 pass rate of ``exp_20260325_230938``
      - keep pushing the remaining G5 failures at ``0.8 m``
    """

    def _seq_cfg(
        *,
        beta: float,
        lambda_mmd: float,
        lambda_axis: float = 0.01,
    ) -> Dict[str, Any]:
        return _cfg(
            arch_variant="seq_bigru_residual",
            layer_sizes=[128, 256, 512],
            latent_dim=4,
            beta=beta,
            free_bits=0.10,
            lr=2e-4,
            batch_size=4096,
            kl_anneal_epochs=80,
            window_size=7,
            window_stride=1,
            window_pad_mode="edge",
            seq_hidden_size=64,
            seq_num_layers=1,
            seq_bidirectional=True,
            lambda_mmd=lambda_mmd,
            mmd_mode="mean_residual",
            lambda_axis=lambda_axis,
            lambda_psd=0.0,
            decoder_distribution="mdn",
            mdn_components=3,
            shuffle_train_batches=True,
        )

    return [
        dict(
            group="S16_seq_mdn_g5_broad",
            tag="S16seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_psd0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(beta=0.0020, lambda_mmd=0.25, lambda_axis=0.01),
        ),
        dict(
            group="S16_seq_mdn_g5_broad",
            tag="S16seq_W7_h64_lat4_mdn3_b0p0018_lmmd0p25_axis0p01_psd0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(beta=0.0018, lambda_mmd=0.25, lambda_axis=0.01),
        ),
        dict(
            group="S16_seq_mdn_g5_broad",
            tag="S16seq_W7_h64_lat4_mdn3_b0p0015_lmmd0p25_axis0p01_psd0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(beta=0.0015, lambda_mmd=0.25, lambda_axis=0.01),
        ),
        dict(
            group="S16_seq_mdn_g5_broad",
            tag="S16seq_W7_h64_lat4_mdn3_b0p002_lmmd0p35_axis0p01_psd0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(beta=0.0020, lambda_mmd=0.35, lambda_axis=0.01),
        ),
        dict(
            group="S16_seq_mdn_g5_broad",
            tag="S16seq_W7_h64_lat4_mdn3_b0p002_lmmd0p5_axis0p01_psd0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(beta=0.0020, lambda_mmd=0.50, lambda_axis=0.01),
        ),
        dict(
            group="S16_seq_mdn_g5_broad",
            tag="S16seq_W7_h64_lat4_mdn3_b0p0018_lmmd0p35_axis0p01_psd0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(beta=0.0018, lambda_mmd=0.35, lambda_axis=0.01),
        ),
        dict(
            group="S16_seq_mdn_g5_broad",
            tag="S16seq_W7_h64_lat4_mdn3_b0p0018_lmmd0p5_axis0p01_psd0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(beta=0.0018, lambda_mmd=0.50, lambda_axis=0.01),
        ),
        dict(
            group="S16_seq_mdn_g5_broad",
            tag="S16seq_W7_h64_lat4_mdn3_b0p0015_lmmd0p35_axis0p01_psd0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(beta=0.0015, lambda_mmd=0.35, lambda_axis=0.01),
        ),
        dict(
            group="S16_seq_mdn_g5_broad",
            tag="S16seq_W7_h64_lat4_mdn3_b0p0015_lmmd0p5_axis0p01_psd0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(beta=0.0015, lambda_mmd=0.50, lambda_axis=0.01),
        ),
        dict(
            group="S16_seq_mdn_g5_broad",
            tag="S16seq_W7_h64_lat4_mdn3_b0p0015_lmmd0p35_axis0p005_psd0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(beta=0.0015, lambda_mmd=0.35, lambda_axis=0.005),
        ),
        dict(
            group="S16_seq_mdn_g5_broad",
            tag="S16seq_W7_h64_lat4_mdn3_b0p0015_lmmd0p5_axis0p005_psd0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(beta=0.0015, lambda_mmd=0.50, lambda_axis=0.005),
        ),
    ]


def _preset_best_compare_large() -> List[Dict[str, Any]]:
    """Comparative protocol-first grid using the strongest current candidates.

    This preset is intentionally mixed-family:
      - ``delta_residual`` anchors from the best point-wise residual experiments
      - ``seq_bigru_residual`` anchors from the MMD-augmented sequential line

    It is designed for protocol-first comparison, not broad discovery:
      - reuse the best protocol-tested delta candidates
      - include the strongest non-MMD seq anchor
      - include the full 2×3 MMD seq block around the all-gates-passed final eval
      - include two promising delta candidates imported from the capacity/optim
        sweep that was run outside the protocol path

    Because seq_bigru_residual requires contiguous context, this preset should
    be executed with ``--no_data_reduction``.
    """
    grid: List[Dict[str, Any]] = []

    # Strongest current point-wise residual anchors.
    delta_candidates = [
        dict(
            group="C5_best_compare",
            tag="D1delta_lat4_b0p001_fb0p0_lr0p0003_L128-256-512",
            cfg=_cfg(
                arch_variant="delta_residual",
                layer_sizes=[128, 256, 512],
                latent_dim=4,
                beta=0.001,
                free_bits=0.0,
                lr=3e-4,
                batch_size=16384,
                kl_anneal_epochs=80,
            ),
        ),
        dict(
            group="C5_best_compare",
            tag="D3delta_lat5_b0p001_fb0p0_lr0p0003_bs16384_anneal80_L128-256-512",
            cfg=_cfg(
                arch_variant="delta_residual",
                layer_sizes=[128, 256, 512],
                latent_dim=5,
                beta=0.001,
                free_bits=0.0,
                lr=3e-4,
                batch_size=16384,
                kl_anneal_epochs=80,
            ),
        ),
        dict(
            group="C5_best_compare",
            tag="COPT_lat6_b0p001_fb0p0_lr0p0001_bs16384_anneal120_L64-128-256",
            cfg=_cfg(
                arch_variant="delta_residual",
                layer_sizes=[64, 128, 256],
                latent_dim=6,
                beta=0.001,
                free_bits=0.0,
                lr=1e-4,
                batch_size=16384,
                kl_anneal_epochs=120,
            ),
        ),
        dict(
            group="C5_best_compare",
            tag="COPT_lat4_b0p001_fb0p0_lr0p0002_bs16384_anneal40_L256-256-256",
            cfg=_cfg(
                arch_variant="delta_residual",
                layer_sizes=[256, 256, 256],
                latent_dim=4,
                beta=0.001,
                free_bits=0.0,
                lr=2e-4,
                batch_size=16384,
                kl_anneal_epochs=40,
            ),
        ),
    ]
    grid.extend(delta_candidates)

    def _seq_cfg(*, beta: float, seq_hidden_size: int, lambda_mmd: float = 0.0) -> Dict[str, Any]:
        return _cfg(
            arch_variant="seq_bigru_residual",
            layer_sizes=[128, 256, 512],
            latent_dim=4,
            beta=beta,
            free_bits=0.10,
            lr=3e-4,
            batch_size=8192,
            kl_anneal_epochs=80,
            window_size=7,
            window_stride=1,
            window_pad_mode="edge",
            seq_hidden_size=seq_hidden_size,
            seq_num_layers=1,
            seq_bidirectional=True,
            lambda_mmd=lambda_mmd,
        )

    # Best seq anchors before and after MMD augmentation.
    seq_candidates = [
        dict(
            group="C5_best_compare",
            tag="S1seq_W7_h64_lat4_b0p001_fb0p10_lr0p0003_L128-256-512",
            cfg=_seq_cfg(beta=0.001, seq_hidden_size=64, lambda_mmd=0.0),
        ),
        dict(
            group="C5_best_compare",
            tag="S1seq_W7_h64_lat4_b0p003_fb0p10_lr0p0003_L128-256-512",
            cfg=_seq_cfg(beta=0.003, seq_hidden_size=64, lambda_mmd=0.0),
        ),
    ]
    for beta in [0.001, 0.003]:
        beta_tag = _tag_beta(beta)
        for lam in [0.1, 0.5, 1.0]:
            lam_tag = str(lam).replace(".", "p")
            seq_candidates.append(
                dict(
                    group="C5_best_compare",
                    tag=f"S2seq_W7_h64_lat4_b{beta_tag}_lmmd{lam_tag}_fb0p10_lr0p0003_L128-256-512",
                    cfg=_seq_cfg(beta=beta, seq_hidden_size=64, lambda_mmd=lam),
                )
            )
    grid.extend(seq_candidates)
    return grid


def _preset_delta_residual_fast() -> List[Dict[str, Any]]:
    """Single-config delta_residual reference with kl_anneal_epochs=20."""
    return [
        dict(
            group="D3_delta_fast",
            tag="D3delta_lat6_b0p001_fb0p0_ann20_L128-256-512",
            cfg=_cfg(
                arch_variant="delta_residual",
                layer_sizes=[128, 256, 512],
                latent_dim=6,
                beta=0.001,
                free_bits=0.0,
                kl_anneal_epochs=20,
                batch_size=16384,
                lr=3e-4,
            ),
        )
    ]


def select_grid(
    overrides: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return the canonical grid filtered by runtime overrides."""
    ov = dict(overrides or {})
    grid = build_default_grid()
    n_original = len(grid)

    preset = ov.get("grid_preset")
    if preset is not None:
        preset_name = str(preset).strip().lower()
        if preset_name == "exploratory_small":
            grid = _preset_exploratory_small(grid)
        elif preset_name == "residual_small":
            grid = _preset_residual_small(grid)
        elif preset_name == "delta_residual_smoke":
            grid = _preset_delta_residual_smoke()
        elif preset_name == "delta_residual_small":
            grid = _preset_delta_residual_small()
        elif preset_name == "delta_residual_refine":
            grid = _preset_delta_residual_refine()
        elif preset_name == "delta_residual_local":
            grid = _preset_delta_residual_local()
        elif preset_name == "delta_residual_frontier":
            grid = _preset_delta_residual_frontier()
        elif preset_name == "legacy2025_smoke":
            grid = _preset_legacy2025_smoke()
        elif preset_name == "legacy2025_ref":
            grid = _preset_legacy2025_ref()
        elif preset_name == "legacy2025_batch_sweep":
            grid = _preset_legacy2025_batch_sweep()
        elif preset_name == "legacy2025_large":
            grid = _preset_legacy2025_large()
        elif preset_name == "seq_residual_smoke":
            grid = _preset_seq_residual_smoke()
        elif preset_name == "seq_residual_small":
            grid = _preset_seq_residual_small()
        elif preset_name == "seq_residual_mmd":
            grid = _preset_seq_residual_mmd()
        elif preset_name == "seq_residual_mmd_final":
            grid = _preset_seq_residual_mmd_final()
        elif preset_name == "seq_residual_nightly":
            grid = _preset_seq_residual_nightly()
        elif preset_name == "seq_investigation_large":
            grid = _preset_seq_investigation_large()
        elif preset_name == "seq_stability_mmd_focus":
            grid = _preset_seq_stability_mmd_focus()
        elif preset_name == "seq_overnight_12h":
            grid = _preset_seq_overnight_12h()
        elif preset_name == "seq_replay_axis_diagnostics":
            grid = _preset_seq_replay_axis_diagnostics()
        elif preset_name == "seq_finish_0p8m":
            grid = _preset_seq_finish_0p8m()
        elif preset_name == "seq_sampled_mmd_compare":
            grid = _preset_seq_sampled_mmd_compare()
        elif preset_name == "seq_hybrid_loss_smoke":
            grid = _preset_seq_hybrid_loss_smoke()
        elif preset_name == "seq_mdn_smoke":
            grid = _preset_seq_mdn_smoke()
        elif preset_name == "seq_mdn_proof":
            grid = _preset_seq_mdn_proof()
        elif preset_name == "seq_mdn_conservative_proof":
            grid = _preset_seq_mdn_conservative_proof()
        elif preset_name == "seq_mdn_exploratory_quick":
            grid = _preset_seq_mdn_exploratory_quick()
        elif preset_name == "seq_mdn_g5_exploratory_quick":
            grid = _preset_seq_mdn_g5_exploratory_quick()
        elif preset_name == "seq_mdn_g5_broader_quick":
            grid = _preset_seq_mdn_g5_broader_quick()
        elif preset_name == "best_compare_large":
            grid = _preset_best_compare_large()
        elif preset_name == "delta_residual_fast":
            grid = _preset_delta_residual_fast()
        else:
            raise ValueError(f"Unknown grid_preset={preset!r}")

    if ov.get("grid_group") is not None:
        grid = [g for g in grid if re.search(str(ov["grid_group"]), g.get("group", ""))]
    if ov.get("grid_tag") is not None:
        grid = [g for g in grid if re.search(str(ov["grid_tag"]), g.get("tag", ""))]
    if ov.get("max_grids") is not None:
        grid = grid[: max(1, int(ov["max_grids"]))]

    if len(grid) != n_original:
        preview = ", ".join(g["tag"] for g in grid[:5])
        if len(grid) > 5:
            preview += f" … (+{len(grid) - 5} more)"
        print(f"⚡ Grid filtered {n_original} → {len(grid)} | [{preview}]")

    print(f"📊 GRID TOTAL (enxuto) = {len(grid)} runs")
    return grid
