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


def _preset_seq_mdn_regime_weight_quick() -> List[Dict[str, Any]]:
    """Compact MDN recovery grid using regime-aware weighted resampling.

    Anchored on the strongest MDN quick result from ``exp_20260325_230938``:

      - ``mdn3``
      - ``beta=0.002``
      - ``lambda_mmd=0.25``
      - ``lambda_axis=0.01``
      - ``lr=2e-4``

    The remaining failures are concentrated in:

      - ``dist_0p8m__curr_100mA``
      - ``dist_0p8m__curr_300mA``
      - ``dist_0p8m__curr_500mA``

    Instead of reopening a broad hyperparameter sweep, keep the anchor fixed
    and change only the training sampling distribution:

      - control: no regime weighting
      - G5 focus A: moderate emphasis on the hard 0.8m regimes
      - G5 focus B: stronger emphasis hedge
    """

    def _seq_cfg(
        *,
        regime_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        cfg = _cfg(
            arch_variant="seq_bigru_residual",
            layer_sizes=[128, 256, 512],
            latent_dim=4,
            beta=0.002,
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
            lambda_mmd=0.25,
            mmd_mode="mean_residual",
            lambda_axis=0.01,
            lambda_psd=0.0,
            decoder_distribution="mdn",
            mdn_components=3,
            shuffle_train_batches=True,
        )
        if regime_weights:
            cfg["train_regime_resample_weights"] = dict(regime_weights)
        return cfg

    return [
        dict(
            group="S17_seq_mdn_regime_weight",
            tag="S17seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_rw0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(),
        ),
        dict(
            group="S17_seq_mdn_regime_weight",
            tag="S17seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_rwg5a_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(
                regime_weights={
                    "dist_0p8m__curr_100ma": 2.5,
                    "dist_0p8m__curr_300ma": 2.5,
                    "dist_0p8m__curr_500ma": 1.75,
                }
            ),
        ),
        dict(
            group="S17_seq_mdn_regime_weight",
            tag="S17seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_rwg5b_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(
                regime_weights={
                    "dist_0p8m__curr_100ma": 3.5,
                    "dist_0p8m__curr_300ma": 3.5,
                    "dist_0p8m__curr_500ma": 2.0,
                }
            ),
        ),
    ]


def _preset_seq_mdn_g5_shape_quick() -> List[Dict[str, Any]]:
    """Quick MDN sweep focused on the remaining G5 failures at 0.8m.

    Anchor:

      - best MDN run ``exp_20260325_230938``
      - ``mdn3``
      - ``beta=0.002``
      - ``lambda_mmd=0.25``
      - ``lambda_axis=0.01``
      - ``lr=2e-4``

    Diagnosis from that run:

      - G6 already passes for all 12 regimes
      - the remaining failures are only G5 on:
        ``0.8m/100mA``, ``0.8m/300mA``, ``0.8m/500mA``
      - variance is already close enough; the mismatch is more about
        skew/kurtosis and marginal tail shape

    So this preset changes only the internal shape of ``axis_loss``:

      - reduce the std term weight
      - increase skew / kurt pressure
      - keep MMD fixed so we do not reopen the G6 problem
    """

    def _seq_cfg(
        *,
        lambda_axis: float = 0.01,
        axis_std_weight: float = 1.0,
        axis_skew_weight: float = 0.25,
        axis_kurt_weight: float = 0.10,
    ) -> Dict[str, Any]:
        return _cfg(
            arch_variant="seq_bigru_residual",
            layer_sizes=[128, 256, 512],
            latent_dim=4,
            beta=0.002,
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
            lambda_mmd=0.25,
            mmd_mode="mean_residual",
            lambda_axis=lambda_axis,
            lambda_psd=0.0,
            axis_std_weight=axis_std_weight,
            axis_skew_weight=axis_skew_weight,
            axis_kurt_weight=axis_kurt_weight,
            decoder_distribution="mdn",
            mdn_components=3,
            shuffle_train_batches=True,
        )

    return [
        dict(
            group="S18_seq_mdn_g5_shape",
            tag="S18seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_std1_sk0p25_ku0p10_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(),
        ),
        dict(
            group="S18_seq_mdn_g5_shape",
            tag="S18seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_std0p5_sk0p5_ku0p20_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(
                axis_std_weight=0.50,
                axis_skew_weight=0.50,
                axis_kurt_weight=0.20,
            ),
        ),
        dict(
            group="S18_seq_mdn_g5_shape",
            tag="S18seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_std0p25_sk0p75_ku0p35_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(
                axis_std_weight=0.25,
                axis_skew_weight=0.75,
                axis_kurt_weight=0.35,
            ),
        ),
        dict(
            group="S18_seq_mdn_g5_shape",
            tag="S18seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p015_std0p25_sk0p75_ku0p35_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(
                lambda_axis=0.015,
                axis_std_weight=0.25,
                axis_skew_weight=0.75,
                axis_kurt_weight=0.35,
            ),
        ),
        dict(
            group="S18_seq_mdn_g5_shape",
            tag="S18seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_std0p25_sk0p5_ku0p50_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(
                axis_std_weight=0.25,
                axis_skew_weight=0.50,
                axis_kurt_weight=0.50,
            ),
        ),
    ]


def _preset_seq_mdn_structure_quick() -> List[Dict[str, Any]]:
    """Quick structural MDN sweep around the best stable MDN anchor.

    Why this preset exists:

      - the local loss-space around the best MDN has already been explored
        fairly aggressively
      - the remaining gap is narrow (`0.8 m`, mostly `G5`)
      - structural MDN capacity has barely been varied so far

    Keep fixed:

      - ``decoder_distribution="mdn"``
      - ``mdn_components=3``
      - ``beta=0.002``
      - ``lambda_mmd=0.25``
      - ``lambda_axis=0.01``
      - ``lr=2e-4``
      - ``free_bits=0.10``

    Explore only:

      - ``latent_dim`` in {4, 6, 8}
      - ``seq_hidden_size`` in {64, 96}
      - ``window_size`` in {7, 11}

    The goal is to test whether the best MDN line was limited more by
    representational capacity than by regularization.
    """

    def _seq_cfg(
        *,
        latent_dim: int = 4,
        seq_hidden_size: int = 64,
        window_size: int = 7,
    ) -> Dict[str, Any]:
        return _cfg(
            arch_variant="seq_bigru_residual",
            layer_sizes=[128, 256, 512],
            latent_dim=latent_dim,
            beta=0.002,
            free_bits=0.10,
            lr=2e-4,
            batch_size=4096,
            kl_anneal_epochs=80,
            window_size=window_size,
            window_stride=1,
            window_pad_mode="edge",
            seq_hidden_size=seq_hidden_size,
            seq_num_layers=1,
            seq_bidirectional=True,
            lambda_mmd=0.25,
            mmd_mode="mean_residual",
            lambda_axis=0.01,
            lambda_psd=0.0,
            decoder_distribution="mdn",
            mdn_components=3,
            shuffle_train_batches=True,
        )

    return [
        dict(
            group="S19_seq_mdn_structure",
            tag="S19seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(),
        ),
        dict(
            group="S19_seq_mdn_structure",
            tag="S19seq_W7_h64_lat6_mdn3_b0p002_lmmd0p25_axis0p01_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(latent_dim=6),
        ),
        dict(
            group="S19_seq_mdn_structure",
            tag="S19seq_W7_h64_lat8_mdn3_b0p002_lmmd0p25_axis0p01_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(latent_dim=8),
        ),
        dict(
            group="S19_seq_mdn_structure",
            tag="S19seq_W7_h96_lat4_mdn3_b0p002_lmmd0p25_axis0p01_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(seq_hidden_size=96),
        ),
        dict(
            group="S19_seq_mdn_structure",
            tag="S19seq_W11_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(window_size=11),
        ),
        dict(
            group="S19_seq_mdn_structure",
            tag="S19seq_W11_h96_lat6_mdn3_b0p002_lmmd0p25_axis0p01_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(latent_dim=6, seq_hidden_size=96, window_size=11),
        ),
    ]


def _preset_seq_mdn_v2_quick() -> List[Dict[str, Any]]:
    """Quick-first MDN v2 sweep with coverage/tail loss and mini reanalysis.

    Anchor:

      - best MDN so far: ``exp_20260325_230938``
      - ``mdn3``
      - ``beta=0.002``
      - ``lambda_mmd=0.25``
      - ``lambda_axis=0.01``
      - ``lr=2e-4``

    New ingredients:

      - ``lambda_coverage`` for marginal calibration pressure
      - ``mini_protocol_v1`` as the grid ranking criterion
      - mini protocol reanalysis over all 12 regimes, once per candidate

    Keep the sweep intentionally small and local:

      - anchor without coverage loss, but already under the new ranking
      - two coverage strengths
      - one sharper temperature hedge
    """

    analysis_quick_overrides = {
        "mini_reanalysis_enabled": True,
        "mini_reanalysis_scope": "all12",
        "mini_reanalysis_max_samples_per_regime": 4096,
        "grid_ranking_mode": "mini_protocol_v1",
    }

    def _seq_cfg(
        *,
        lambda_coverage: float,
        coverage_temperature: float = 0.05,
    ) -> Dict[str, Any]:
        return _cfg(
            arch_variant="seq_bigru_residual",
            layer_sizes=[128, 256, 512],
            latent_dim=4,
            beta=0.002,
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
            lambda_mmd=0.25,
            mmd_mode="mean_residual",
            lambda_axis=0.01,
            lambda_psd=0.0,
            lambda_coverage=lambda_coverage,
            coverage_levels=[0.50, 0.80, 0.95],
            tail_levels=[0.05, 0.95],
            coverage_temperature=coverage_temperature,
            decoder_distribution="mdn",
            mdn_components=3,
            shuffle_train_batches=True,
        )

    return [
        dict(
            group="S20_seq_mdn_v2",
            tag="S20seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_coverage=0.0),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S20_seq_mdn_v2",
            tag="S20seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p02_t0p05_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_coverage=0.02, coverage_temperature=0.05),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S20_seq_mdn_v2",
            tag="S20seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p05_t0p05_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_coverage=0.05, coverage_temperature=0.05),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S20_seq_mdn_v2",
            tag="S20seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p05_t0p03_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_coverage=0.05, coverage_temperature=0.03),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
    ]


def _preset_seq_mdn_v2_perf_compare_quick() -> List[Dict[str, Any]]:
    """Throughput-focused compare against the current MDN v2 anchor.

    This preset is intentionally narrow:

      1. current anchor (`batch_size=4096`, `batch_infer=8192`, `seq_gru_unroll=True`)
      2. larger train/eval batches only
      3. larger train/eval batches + `seq_gru_unroll=False`

    The scientific objective is unchanged; this preset exists to compare
    throughput/fit behaviour across the conservative path and the faster GRU
    path. ``seq_gru_unroll=False`` remains opt-in because previous seq runs on
    newer stacks (notably RTX 5090/cuDNN 9) motivated the conservative default.
    """

    analysis_quick_base = {
        "mini_reanalysis_enabled": True,
        "mini_reanalysis_scope": "all12",
        "mini_reanalysis_max_samples_per_regime": 4096,
        "grid_ranking_mode": "mini_protocol_v1",
        "batch_infer": 8192,
    }
    analysis_quick_fast = {
        **analysis_quick_base,
        "batch_infer": 16384,
    }

    def _seq_cfg(
        *,
        batch_size: int,
        seq_gru_unroll: bool,
    ) -> Dict[str, Any]:
        return _cfg(
            arch_variant="seq_bigru_residual",
            layer_sizes=[128, 256, 512],
            latent_dim=4,
            beta=0.002,
            free_bits=0.10,
            lr=2e-4,
            batch_size=batch_size,
            kl_anneal_epochs=80,
            window_size=7,
            window_stride=1,
            window_pad_mode="edge",
            seq_hidden_size=64,
            seq_num_layers=1,
            seq_bidirectional=True,
            seq_gru_unroll=seq_gru_unroll,
            lambda_mmd=0.25,
            mmd_mode="mean_residual",
            lambda_axis=0.01,
            lambda_psd=0.0,
            lambda_coverage=0.0,
            coverage_levels=[0.50, 0.80, 0.95],
            tail_levels=[0.05, 0.95],
            coverage_temperature=0.05,
            decoder_distribution="mdn",
            mdn_components=3,
            shuffle_train_batches=True,
        )

    return [
        dict(
            group="S21_seq_mdn_v2_perf",
            tag="S21seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0_bs4096_bi8192_gruroll1_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(batch_size=4096, seq_gru_unroll=True),
            analysis_quick_overrides=analysis_quick_base,
        ),
        dict(
            group="S21_seq_mdn_v2_perf",
            tag="S21seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0_bs8192_bi16384_gruroll1_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(batch_size=8192, seq_gru_unroll=True),
            analysis_quick_overrides=analysis_quick_fast,
        ),
        dict(
            group="S21_seq_mdn_v2_perf",
            tag="S21seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(batch_size=8192, seq_gru_unroll=False),
            analysis_quick_overrides=analysis_quick_fast,
        ),
    ]


def _preset_seq_mdn_v2_fastbase_quick() -> List[Dict[str, Any]]:
    """Quick-first MDN v2 sweep on top of the faster validated seq baseline.

    Baseline chosen from ``seq_mdn_v2_perf_compare_quick``:

      - ``batch_size=8192``
      - ``batch_infer=16384``
      - ``seq_gru_unroll=False``

    The scientific sweep stays the same as ``seq_mdn_v2_quick``:

      - anchor without coverage loss
      - two coverage strengths
      - one sharper temperature hedge

    This keeps the hypothesis local while moving the whole MDN v2 line onto the
    faster operational path validated on the current A6000 stack.
    """

    analysis_quick_overrides = {
        "mini_reanalysis_enabled": True,
        "mini_reanalysis_scope": "all12",
        "mini_reanalysis_max_samples_per_regime": 4096,
        "grid_ranking_mode": "mini_protocol_v1",
        "batch_infer": 16384,
    }

    def _seq_cfg(
        *,
        lambda_coverage: float,
        coverage_temperature: float = 0.05,
    ) -> Dict[str, Any]:
        return _cfg(
            arch_variant="seq_bigru_residual",
            layer_sizes=[128, 256, 512],
            latent_dim=4,
            beta=0.002,
            free_bits=0.10,
            lr=2e-4,
            batch_size=8192,
            kl_anneal_epochs=80,
            window_size=7,
            window_stride=1,
            window_pad_mode="edge",
            seq_hidden_size=64,
            seq_num_layers=1,
            seq_bidirectional=True,
            seq_gru_unroll=False,
            lambda_mmd=0.25,
            mmd_mode="mean_residual",
            lambda_axis=0.01,
            lambda_psd=0.0,
            lambda_coverage=lambda_coverage,
            coverage_levels=[0.50, 0.80, 0.95],
            tail_levels=[0.05, 0.95],
            coverage_temperature=coverage_temperature,
            decoder_distribution="mdn",
            mdn_components=3,
            shuffle_train_batches=True,
        )

    return [
        dict(
            group="S22_seq_mdn_v2_fast",
            tag="S22seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_coverage=0.0),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S22_seq_mdn_v2_fast",
            tag="S22seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p02_t0p05_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_coverage=0.02, coverage_temperature=0.05),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S22_seq_mdn_v2_fast",
            tag="S22seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p05_t0p05_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_coverage=0.05, coverage_temperature=0.05),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S22_seq_mdn_v2_fast",
            tag="S22seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p05_t0p03_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_coverage=0.05, coverage_temperature=0.03),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
    ]


def _preset_seq_mdn_v2_g5_followup_quick() -> List[Dict[str, Any]]:
    """Local follow-up around the best fastbase+coverage MDN v2 candidate.

    Anchor chosen from ``seq_mdn_v2_fastbase_quick``:

      - ``lambda_coverage=0.05``
      - ``coverage_temperature=0.03``
      - ``batch_size=8192``
      - ``batch_infer=16384``
      - ``seq_gru_unroll=False``

    Goal:

      - preserve the `G6` recovery seen in the fastbase coverage run
      - probe a narrow neighborhood that may recover one more `G5` regime

    Local knobs only:

      - slightly lighter / stronger coverage
      - slightly softer temperature
      - one small `lambda_axis` hedge
    """

    analysis_quick_overrides = {
        "mini_reanalysis_enabled": True,
        "mini_reanalysis_scope": "all12",
        "mini_reanalysis_max_samples_per_regime": 4096,
        "grid_ranking_mode": "mini_protocol_v1",
        "batch_infer": 16384,
    }

    def _seq_cfg(
        *,
        lambda_coverage: float,
        coverage_temperature: float,
        lambda_axis: float = 0.01,
    ) -> Dict[str, Any]:
        return _cfg(
            arch_variant="seq_bigru_residual",
            layer_sizes=[128, 256, 512],
            latent_dim=4,
            beta=0.002,
            free_bits=0.10,
            lr=2e-4,
            batch_size=8192,
            kl_anneal_epochs=80,
            window_size=7,
            window_stride=1,
            window_pad_mode="edge",
            seq_hidden_size=64,
            seq_num_layers=1,
            seq_bidirectional=True,
            seq_gru_unroll=False,
            lambda_mmd=0.25,
            mmd_mode="mean_residual",
            lambda_axis=lambda_axis,
            lambda_psd=0.0,
            lambda_coverage=lambda_coverage,
            coverage_levels=[0.50, 0.80, 0.95],
            tail_levels=[0.05, 0.95],
            coverage_temperature=coverage_temperature,
            decoder_distribution="mdn",
            mdn_components=3,
            shuffle_train_batches=True,
        )

    return [
        dict(
            group="S23_seq_mdn_v2_g5",
            tag="S23seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p05_t0p03_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_coverage=0.05, coverage_temperature=0.03, lambda_axis=0.01),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S23_seq_mdn_v2_g5",
            tag="S23seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p04_t0p03_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_coverage=0.04, coverage_temperature=0.03, lambda_axis=0.01),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S23_seq_mdn_v2_g5",
            tag="S23seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p06_t0p03_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_coverage=0.06, coverage_temperature=0.03, lambda_axis=0.01),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S23_seq_mdn_v2_g5",
            tag="S23seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p05_t0p04_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_coverage=0.05, coverage_temperature=0.04, lambda_axis=0.01),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S23_seq_mdn_v2_g5",
            tag="S23seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p0125_cov0p05_t0p03_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_coverage=0.05, coverage_temperature=0.03, lambda_axis=0.0125),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
    ]


def _preset_seq_mdn_v2_overnight_decision_quick() -> List[Dict[str, Any]]:
    """Overnight MDN v2 grid: local refinement plus exploratory probes.

    Purpose:

      - exploit the current best ``S23`` neighborhood
      - include a small structural/regularization exploration branch
      - decide whether the line still has headroom or is plateauing

    Layout:

      - local branch around the current winner ``cov=0.06 / t=0.03``
      - exploratory branch probing capacity and one stronger MMD hedge

    The whole preset stays on the faster seq baseline:

      - `batch_infer=16384`
      - `seq_gru_unroll=False`
    """

    analysis_quick_overrides = {
        "mini_reanalysis_enabled": True,
        "mini_reanalysis_scope": "all12",
        "mini_reanalysis_max_samples_per_regime": 4096,
        "grid_ranking_mode": "mini_protocol_v1",
        "batch_infer": 16384,
    }

    def _seq_cfg(
        *,
        lambda_coverage: float,
        coverage_temperature: float,
        lambda_axis: float = 0.01,
        lambda_mmd: float = 0.25,
        latent_dim: int = 4,
        seq_hidden_size: int = 64,
        window_size: int = 7,
        batch_size: int = 8192,
    ) -> Dict[str, Any]:
        return _cfg(
            arch_variant="seq_bigru_residual",
            layer_sizes=[128, 256, 512],
            latent_dim=latent_dim,
            beta=0.002,
            free_bits=0.10,
            lr=2e-4,
            batch_size=batch_size,
            kl_anneal_epochs=80,
            window_size=window_size,
            window_stride=1,
            window_pad_mode="edge",
            seq_hidden_size=seq_hidden_size,
            seq_num_layers=1,
            seq_bidirectional=True,
            seq_gru_unroll=False,
            lambda_mmd=lambda_mmd,
            mmd_mode="mean_residual",
            lambda_axis=lambda_axis,
            lambda_psd=0.0,
            lambda_coverage=lambda_coverage,
            coverage_levels=[0.50, 0.80, 0.95],
            tail_levels=[0.05, 0.95],
            coverage_temperature=coverage_temperature,
            decoder_distribution="mdn",
            mdn_components=3,
            shuffle_train_batches=True,
        )

    return [
        # Local branch around the S23 champion.
        dict(
            group="S24_seq_mdn_v2_overnight",
            tag="S24seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p06_t0p03_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_coverage=0.06, coverage_temperature=0.03),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S24_seq_mdn_v2_overnight",
            tag="S24seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p07_t0p03_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_coverage=0.07, coverage_temperature=0.03),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S24_seq_mdn_v2_overnight",
            tag="S24seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p06_t0p025_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_coverage=0.06, coverage_temperature=0.025),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S24_seq_mdn_v2_overnight",
            tag="S24seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p06_t0p035_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_coverage=0.06, coverage_temperature=0.035),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # Exploratory branch to decide whether the line still has headroom.
        dict(
            group="S24_seq_mdn_v2_overnight",
            tag="S24seq_W7_h64_lat6_mdn3_b0p002_lmmd0p25_axis0p01_cov0p06_t0p03_bs6144_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(
                lambda_coverage=0.06,
                coverage_temperature=0.03,
                latent_dim=6,
                batch_size=6144,
            ),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S24_seq_mdn_v2_overnight",
            tag="S24seq_W7_h96_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p06_t0p03_bs6144_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(
                lambda_coverage=0.06,
                coverage_temperature=0.03,
                seq_hidden_size=96,
                batch_size=6144,
            ),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S24_seq_mdn_v2_overnight",
            tag="S24seq_W11_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p06_t0p03_bs6144_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(
                lambda_coverage=0.06,
                coverage_temperature=0.03,
                window_size=11,
                batch_size=6144,
            ),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S24_seq_mdn_v2_overnight",
            tag="S24seq_W7_h64_lat4_mdn3_b0p002_lmmd0p30_axis0p01_cov0p06_t0p03_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(
                lambda_coverage=0.06,
                coverage_temperature=0.03,
                lambda_mmd=0.30,
            ),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S24_seq_mdn_v2_overnight",
            tag="S24seq_W7_h96_lat6_mdn3_b0p002_lmmd0p25_axis0p01_cov0p06_t0p03_bs6144_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(
                lambda_coverage=0.06,
                coverage_temperature=0.03,
                latent_dim=6,
                seq_hidden_size=96,
                batch_size=6144,
            ),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
    ]


def _preset_seq_mdn_v2_overnight_5090safe_quick() -> List[Dict[str, Any]]:
    """5090-safe overnight MDN v2 grid.

    This preset keeps the scientific intent of
    ``seq_mdn_v2_overnight_decision_quick`` but encodes the runtime guardrail
    learned from the RTX 5090 stack:

      - keep `seq_gru_unroll=False` only on the already validated `W7 / h64`
        branch
      - force `seq_gru_unroll=True` on structural probes such as `h96` or
        `W11`

    That separates scientific failures from cuDNN runtime failures on the
    5090 machine.
    """

    analysis_quick_overrides = {
        "mini_reanalysis_enabled": True,
        "mini_reanalysis_scope": "all12",
        "mini_reanalysis_max_samples_per_regime": 4096,
        "grid_ranking_mode": "mini_protocol_v1",
        "batch_infer": 16384,
    }

    def _seq_cfg(
        *,
        lambda_coverage: float,
        coverage_temperature: float,
        lambda_axis: float = 0.01,
        lambda_mmd: float = 0.25,
        latent_dim: int = 4,
        seq_hidden_size: int = 64,
        window_size: int = 7,
        batch_size: int = 8192,
        seq_gru_unroll: bool = False,
    ) -> Dict[str, Any]:
        return _cfg(
            arch_variant="seq_bigru_residual",
            layer_sizes=[128, 256, 512],
            latent_dim=latent_dim,
            beta=0.002,
            free_bits=0.10,
            lr=2e-4,
            batch_size=batch_size,
            kl_anneal_epochs=80,
            window_size=window_size,
            window_stride=1,
            window_pad_mode="edge",
            seq_hidden_size=seq_hidden_size,
            seq_num_layers=1,
            seq_bidirectional=True,
            seq_gru_unroll=seq_gru_unroll,
            lambda_mmd=lambda_mmd,
            mmd_mode="mean_residual",
            lambda_axis=lambda_axis,
            lambda_psd=0.0,
            lambda_coverage=lambda_coverage,
            coverage_levels=[0.50, 0.80, 0.95],
            tail_levels=[0.05, 0.95],
            coverage_temperature=coverage_temperature,
            decoder_distribution="mdn",
            mdn_components=3,
            shuffle_train_batches=True,
        )

    return [
        # Local branch: keep the already validated 5090 fast path.
        dict(
            group="S25_seq_mdn_v2_overnight_5090safe",
            tag="S25seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p06_t0p03_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_coverage=0.06, coverage_temperature=0.03, seq_gru_unroll=False),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S25_seq_mdn_v2_overnight_5090safe",
            tag="S25seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p07_t0p03_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_coverage=0.07, coverage_temperature=0.03, seq_gru_unroll=False),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S25_seq_mdn_v2_overnight_5090safe",
            tag="S25seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p06_t0p025_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_coverage=0.06, coverage_temperature=0.025, seq_gru_unroll=False),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S25_seq_mdn_v2_overnight_5090safe",
            tag="S25seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p06_t0p035_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(lambda_coverage=0.06, coverage_temperature=0.035, seq_gru_unroll=False),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S25_seq_mdn_v2_overnight_5090safe",
            tag="S25seq_W7_h64_lat4_mdn3_b0p002_lmmd0p30_axis0p01_cov0p06_t0p03_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(
                lambda_coverage=0.06,
                coverage_temperature=0.03,
                lambda_mmd=0.30,
                seq_gru_unroll=False,
            ),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # Structural probes: force the conservative GRU path on the 5090 stack.
        dict(
            group="S25_seq_mdn_v2_overnight_5090safe",
            tag="S25seq_W7_h64_lat6_mdn3_b0p002_lmmd0p25_axis0p01_cov0p06_t0p03_bs6144_bi16384_gruroll1_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(
                lambda_coverage=0.06,
                coverage_temperature=0.03,
                latent_dim=6,
                batch_size=6144,
                seq_gru_unroll=True,
            ),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S25_seq_mdn_v2_overnight_5090safe",
            tag="S25seq_W7_h96_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p06_t0p03_bs6144_bi16384_gruroll1_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(
                lambda_coverage=0.06,
                coverage_temperature=0.03,
                seq_hidden_size=96,
                batch_size=6144,
                seq_gru_unroll=True,
            ),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S25_seq_mdn_v2_overnight_5090safe",
            tag="S25seq_W11_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p06_t0p03_bs6144_bi16384_gruroll1_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(
                lambda_coverage=0.06,
                coverage_temperature=0.03,
                window_size=11,
                batch_size=6144,
                seq_gru_unroll=True,
            ),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S25_seq_mdn_v2_overnight_5090safe",
            tag="S25seq_W7_h96_lat6_mdn3_b0p002_lmmd0p25_axis0p01_cov0p06_t0p03_bs6144_bi16384_gruroll1_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(
                lambda_coverage=0.06,
                coverage_temperature=0.03,
                latent_dim=6,
                seq_hidden_size=96,
                batch_size=6144,
                seq_gru_unroll=True,
            ),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
    ]


def _preset_seq_mdn_v2_ceiling_probe_quick() -> List[Dict[str, Any]]:
    """Ceiling probe: test untouched hyperparameter axes for MDN v2.

    The best MDN v2 result so far is 9/12 (S25, W7/h64/lat6/mdn3).
    Three regimes fail at 0.8 m:
      - 300 mA: G3 (cov_rel_var = 0.226, threshold < 0.20)
      - 500 mA: G5 (delta_jb_stat_rel = 0.574, threshold < 0.20)
      - 700 mA: G5 (delta_jb_stat_rel = 0.468, threshold < 0.20)

    This grid probes axes that remained fixed throughout the MDN v2 line:
      A) mdn_components = 5  (was always 3)
      B) deeper decoder      (was always [128,256,512])
      C) latent_dim = 8      (was max 6)
      D) higher beta          (was always 0.002)
      E) higher lambda_axis   (was always 0.01)

    Each candidate changes exactly one axis from the champion baseline
    so the effect is attributable.
    """
    analysis_quick_overrides = {
        "train_regime_diagnostics_enabled": False,
        "mini_reanalysis_enabled": True,
        "mini_reanalysis_scope": "all12",
        "mini_reanalysis_max_samples_per_regime": 4096,
        "grid_ranking_mode": "mini_protocol_v1",
        "batch_infer": 16384,
    }

    _base = dict(
        arch_variant="seq_bigru_residual",
        layer_sizes=[128, 256, 512],
        latent_dim=6,
        beta=0.002,
        free_bits=0.10,
        lr=2e-4,
        batch_size=6144,
        kl_anneal_epochs=80,
        window_size=7,
        window_stride=1,
        window_pad_mode="edge",
        seq_hidden_size=64,
        seq_num_layers=1,
        seq_bidirectional=True,
        seq_gru_unroll=True,
        lambda_mmd=0.25,
        mmd_mode="mean_residual",
        lambda_axis=0.01,
        lambda_psd=0.0,
        lambda_coverage=0.06,
        coverage_levels=[0.50, 0.80, 0.95],
        tail_levels=[0.05, 0.95],
        coverage_temperature=0.03,
        decoder_distribution="mdn",
        mdn_components=3,
        shuffle_train_batches=True,
    )

    def _variant(overrides: Dict[str, Any]) -> Dict[str, Any]:
        cfg = dict(_base)
        cfg.update(overrides)
        return _cfg(**cfg)

    return [
        # 0 — Champion control (exact S25 winner for direct comparison)
        dict(
            group="S26_seq_mdn_v2_ceiling_probe",
            tag="S26seq_control_W7_h64_lat6_mdn3_cov0p06_t0p03_L128-256-512",
            cfg=_variant({}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # A — MDN 5 components (more mixture capacity for residual shape)
        dict(
            group="S26_seq_mdn_v2_ceiling_probe",
            tag="S26seq_mdn5_W7_h64_lat6_cov0p06_t0p03_L128-256-512",
            cfg=_variant({"mdn_components": 5}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # B — Deeper decoder (extra capacity for covariance fit)
        dict(
            group="S26_seq_mdn_v2_ceiling_probe",
            tag="S26seq_decdeep_W7_h64_lat6_mdn3_cov0p06_t0p03_L128-256-512-256",
            cfg=_variant({"layer_sizes": [128, 256, 512, 256]}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # C — Latent dim 8 (more latent capacity)
        dict(
            group="S26_seq_mdn_v2_ceiling_probe",
            tag="S26seq_lat8_W7_h64_mdn3_cov0p06_t0p03_L128-256-512",
            cfg=_variant({"latent_dim": 8}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # D — Higher beta (stronger KL regularization → more gaussian latent)
        dict(
            group="S26_seq_mdn_v2_ceiling_probe",
            tag="S26seq_beta5e3_W7_h64_lat6_mdn3_cov0p06_t0p03_L128-256-512",
            cfg=_variant({"beta": 0.005}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # E — Higher lambda_axis (direct shape regularization)
        dict(
            group="S26_seq_mdn_v2_ceiling_probe",
            tag="S26seq_axis5e2_W7_h64_lat6_mdn3_cov0p06_t0p03_L128-256-512",
            cfg=_variant({"lambda_axis": 0.05}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
    ]


def _preset_seq_mdn_v2_ceiling_full() -> List[Dict[str, Any]]:
    """Phase 1b: retrain top ceiling-probe candidates with maximum data + full eval.

    Advances the three most informative candidates from the quick ceiling probe:
      - control (S25 clone) to measure stochastic variance
      - lat8    (latent_dim=8, best quick-probe result: 5/12)
      - mdn5    (mdn_components=5, second-best variant: 4/12)

    Intended to run WITHOUT --max_samples_per_exp (use all ~8.6 M train samples)
    and with full protocol evaluation (no quick overrides).
    """
    analysis_quick_overrides = {
        "train_regime_diagnostics_enabled": False,
        "mini_reanalysis_enabled": True,
        "mini_reanalysis_scope": "all12",
        "mini_reanalysis_max_samples_per_regime": 4096,
        "grid_ranking_mode": "mini_protocol_v1",
        "batch_infer": 16384,
    }

    _base = dict(
        arch_variant="seq_bigru_residual",
        layer_sizes=[128, 256, 512],
        latent_dim=6,
        beta=0.002,
        free_bits=0.10,
        lr=2e-4,
        batch_size=6144,
        kl_anneal_epochs=80,
        window_size=7,
        window_stride=1,
        window_pad_mode="edge",
        seq_hidden_size=64,
        seq_num_layers=1,
        seq_bidirectional=True,
        seq_gru_unroll=True,
        lambda_mmd=0.25,
        mmd_mode="mean_residual",
        lambda_axis=0.01,
        lambda_psd=0.0,
        lambda_coverage=0.06,
        coverage_levels=[0.50, 0.80, 0.95],
        tail_levels=[0.05, 0.95],
        coverage_temperature=0.03,
        decoder_distribution="mdn",
        mdn_components=3,
        shuffle_train_batches=True,
    )

    def _variant(overrides: Dict[str, Any]) -> Dict[str, Any]:
        cfg = dict(_base)
        cfg.update(overrides)
        return _cfg(**cfg)

    return [
        # 0 — Control (exact S25 baseline for variance estimation)
        dict(
            group="S26_seq_mdn_v2_ceiling_full",
            tag="S26full_control_W7_h64_lat6_mdn3_cov0p06_t0p03_L128-256-512",
            cfg=_variant({}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # 1 — Latent dim 8 (best quick-probe candidate: 5/12)
        dict(
            group="S26_seq_mdn_v2_ceiling_full",
            tag="S26full_lat8_W7_h64_mdn3_cov0p06_t0p03_L128-256-512",
            cfg=_variant({"latent_dim": 8}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # 2 — MDN 5 components (second-best quick-probe variant: 4/12)
        dict(
            group="S26_seq_mdn_v2_ceiling_full",
            tag="S26full_mdn5_W7_h64_lat6_cov0p06_t0p03_L128-256-512",
            cfg=_variant({"mdn_components": 5}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
    ]


def _preset_seq_coverage_tail_sweep() -> List[Dict[str, Any]]:
    """Coverage/tail calibration sweep — MDN v2 lat8 base.

    Systematic under-dispersion observed across ALL regimes: the cVAE
    consistently produces lighter tails and a broader/flatter peak than
    the real distribution (Δcov95 ≈ -0.17 universal). The current
    lambda_coverage=0.06 gradient is too weak to compete with recon loss.

    This grid directly targets tail calibration by varying:
      A) lambda_coverage: 0.06 (control) → 0.15 → 0.25 → 0.40
      B) tail_levels: [0.05,0.95] → [0.01,0.99]  (extreme tails)
      C) coverage_temperature: 0.03 → 0.01  (harder/sharper loss)

    Base config: lat8 MDN v2 champion (best full-protocol result: 8/12).
    Run WITHOUT --max_samples_per_exp (full 8.6M) + full stat eval.
    """
    analysis_quick_overrides = {
        "train_regime_diagnostics_enabled": False,
        "mini_reanalysis_enabled": True,
        "mini_reanalysis_scope": "all12",
        "mini_reanalysis_max_samples_per_regime": 4096,
        "grid_ranking_mode": "mini_protocol_v1",
        "batch_infer": 16384,
    }

    _base = dict(
        arch_variant="seq_bigru_residual",
        layer_sizes=[128, 256, 512],
        latent_dim=8,
        beta=0.002,
        free_bits=0.10,
        lr=2e-4,
        batch_size=6144,
        kl_anneal_epochs=80,
        window_size=7,
        window_stride=1,
        window_pad_mode="edge",
        seq_hidden_size=64,
        seq_num_layers=1,
        seq_bidirectional=True,
        seq_gru_unroll=True,
        lambda_mmd=0.25,
        mmd_mode="mean_residual",
        lambda_axis=0.01,
        lambda_psd=0.0,
        lambda_coverage=0.06,
        coverage_levels=[0.50, 0.80, 0.95],
        tail_levels=[0.05, 0.95],
        coverage_temperature=0.03,
        decoder_distribution="mdn",
        mdn_components=3,
        shuffle_train_batches=True,
    )

    def _variant(overrides):
        cfg = dict(_base)
        cfg.update(overrides)
        return _cfg(**cfg)

    return [
        # 0 — Control (lat8 baseline, coverage as-is)
        dict(
            group="S27_coverage_tail_sweep",
            tag="S27cov_ctrl_lc0p06_tail95_t0p03",
            cfg=_variant({}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # A1 — Stronger coverage (2.5x)
        dict(
            group="S27_coverage_tail_sweep",
            tag="S27cov_lc0p15_tail95_t0p03",
            cfg=_variant({"lambda_coverage": 0.15}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # A2 — Stronger coverage (4x)
        dict(
            group="S27_coverage_tail_sweep",
            tag="S27cov_lc0p25_tail95_t0p03",
            cfg=_variant({"lambda_coverage": 0.25}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # A3 — Strongest coverage (6.5x)
        dict(
            group="S27_coverage_tail_sweep",
            tag="S27cov_lc0p40_tail95_t0p03",
            cfg=_variant({"lambda_coverage": 0.40}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # B — Extreme tail levels (1st/99th percentile) + strong coverage
        dict(
            group="S27_coverage_tail_sweep",
            tag="S27cov_lc0p25_tail99_t0p03",
            cfg=_variant({"lambda_coverage": 0.25, "tail_levels": [0.01, 0.99]}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # C — Extreme tails + hard temperature (sharper loss)
        dict(
            group="S27_coverage_tail_sweep",
            tag="S27cov_lc0p25_tail99_t0p01",
            cfg=_variant({"lambda_coverage": 0.25, "tail_levels": [0.01, 0.99],
                          "coverage_temperature": 0.01}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
    ]


def _preset_seq_kurt_loss_sweep() -> List[Dict[str, Any]]:
    """S28 — Explicit kurtosis loss sweep on top of S27 champion.

    S27 champion (lc=0.25, tail95, t=0.03) reached 10/12 full protocol.
    Remaining failures: 0.8m/100mA and 0.8m/300mA fail only G5 (JBrel=3.59
    and 0.36 respectively) due to leptokurtic shape mismatch — the model
    produces smoother bell-shaped output vs the real peaky/heavy-tailed
    residual distribution.

    lambda_axis already exists but uses weighted std+skew+kurt (axis_kurt_weight
    =0.10) and was 0.01 in the S27 champion — kurtosis signal is diluted.
    lambda_kurt isolates the 4th-moment MSE so the gradient budget is fully
    dedicated to matching kurtosis.

    Sweep lambda_kurt ∈ {0.05, 0.10, 0.20} on top of the S27 A2 winner base.
    CTRL = exact S27 A2 winner (lambda_kurt=0.0) for a clean before/after.
    """
    analysis_quick_overrides = {
        "train_regime_diagnostics_enabled": False,
        "mini_reanalysis_enabled": True,
        "mini_reanalysis_scope": "all12",
        "mini_reanalysis_max_samples_per_regime": 4096,
        "grid_ranking_mode": "mini_protocol_v1",
        "batch_infer": 16384,
    }

    _base = dict(
        arch_variant="seq_bigru_residual",
        layer_sizes=[128, 256, 512],
        latent_dim=8,
        beta=0.002,
        free_bits=0.10,
        lr=2e-4,
        batch_size=6144,
        kl_anneal_epochs=80,
        window_size=7,
        window_stride=1,
        window_pad_mode="edge",
        seq_hidden_size=64,
        seq_num_layers=1,
        seq_bidirectional=True,
        seq_gru_unroll=True,
        lambda_mmd=0.25,
        mmd_mode="mean_residual",
        lambda_axis=0.01,
        lambda_psd=0.0,
        lambda_coverage=0.25,
        coverage_levels=[0.50, 0.80, 0.95],
        tail_levels=[0.05, 0.95],
        coverage_temperature=0.03,
        lambda_kurt=0.0,
        decoder_distribution="mdn",
        mdn_components=3,
        shuffle_train_batches=True,
    )

    def _variant(overrides):
        cfg = dict(_base)
        cfg.update(overrides)
        return _cfg(**cfg)

    return [
        # CTRL — exact S27 A2 winner (no kurtosis loss)
        dict(
            group="S28_kurt_loss_sweep",
            tag="S28kurt_ctrl_lk0p0",
            cfg=_variant({}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # A1 — Mild kurtosis pressure
        dict(
            group="S28_kurt_loss_sweep",
            tag="S28kurt_lk0p05",
            cfg=_variant({"lambda_kurt": 0.05}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # A2 — Moderate kurtosis pressure
        dict(
            group="S28_kurt_loss_sweep",
            tag="S28kurt_lk0p10",
            cfg=_variant({"lambda_kurt": 0.10}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # A3 — Strong kurtosis pressure
        dict(
            group="S28_kurt_loss_sweep",
            tag="S28kurt_lk0p20",
            cfg=_variant({"lambda_kurt": 0.20}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
    ]


def _preset_seq_mdn_components_sweep() -> List[Dict[str, Any]]:
    """S29 — MDN components sweep on S27 A2 champion base.

    S28 showed that lambda_kurt (explicit kurtosis pressure) does not fix G5
    at 0.8m/100mA and 300mA — the JBrel failure is rooted in the MDN's limited
    capacity to reproduce leptokurtic peak+tail structure with only 3 components.

    The ceiling probe (S26) tested mdn_components=5 globally but on the older
    base (no lambda_coverage, 100k/exp quick run) — it showed marginal gain vs
    lat8 with 3 components. Now that lambda_coverage=0.25 reshapes the MDN
    component placement, more components may unlock the shape capacity needed
    for 0.8m low-current regimes.

    This sweep fixes the S27 A2 base and varies mdn_components:
      CTRL: mdn_components=3 (exact S27 A2 winner — measures training variance)
      A1:   mdn_components=5
      A2:   mdn_components=8

    Full data (8.6M), same eval flags as S27/S28.
    """
    analysis_quick_overrides = {
        "train_regime_diagnostics_enabled": False,
        "mini_reanalysis_enabled": True,
        "mini_reanalysis_scope": "all12",
        "mini_reanalysis_max_samples_per_regime": 4096,
        "grid_ranking_mode": "mini_protocol_v1",
        "batch_infer": 16384,
    }

    _base = dict(
        arch_variant="seq_bigru_residual",
        layer_sizes=[128, 256, 512],
        latent_dim=8,
        beta=0.002,
        free_bits=0.10,
        lr=2e-4,
        batch_size=6144,
        kl_anneal_epochs=80,
        window_size=7,
        window_stride=1,
        window_pad_mode="edge",
        seq_hidden_size=64,
        seq_num_layers=1,
        seq_bidirectional=True,
        seq_gru_unroll=True,
        lambda_mmd=0.25,
        mmd_mode="mean_residual",
        lambda_axis=0.01,
        lambda_psd=0.0,
        lambda_coverage=0.25,
        coverage_levels=[0.50, 0.80, 0.95],
        tail_levels=[0.05, 0.95],
        coverage_temperature=0.03,
        lambda_kurt=0.0,
        decoder_distribution="mdn",
        mdn_components=3,
        shuffle_train_batches=True,
    )

    def _variant(overrides):
        cfg = dict(_base)
        cfg.update(overrides)
        return _cfg(**cfg)

    return [
        # CTRL — exact S27 A2 winner (mdn=3, second seed for variance estimate)
        dict(
            group="S29_mdn_components_sweep",
            tag="S29mdn_ctrl_k3",
            cfg=_variant({}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # A1 — 5 MDN components
        dict(
            group="S29_mdn_components_sweep",
            tag="S29mdn_k5",
            cfg=_variant({"mdn_components": 5}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # A2 — 8 MDN components
        dict(
            group="S29_mdn_components_sweep",
            tag="S29mdn_k8",
            cfg=_variant({"mdn_components": 8}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
    ]


def _preset_seq_cond_embed_sweep() -> List[Dict[str, Any]]:
    """S30 — Decoder regime conditioning embedding sweep on S27 A2 base.

    S29 showed that increasing MDN components (k=5, 8) does not fix G5 at
    0.8m/300mA. The root cause hypothesis is that the decoder receives d and c
    as raw scalars (2 / (latent+4) of its input), whereas prior and encoder
    broadcast d,c into a BiGRU time path — giving the encoder/prior a far
    richer nonlinear representation of the regime.

    This sweep adds a small 2-layer MLP (cond_embed_dim units) that embeds
    (d, c) before concatenation with z and x_center in the decoder:
      CTRL: cond_embed_dim=0 (exact S27 A2 winner — measures training variance)
      A1:   cond_embed_dim=16
      A2:   cond_embed_dim=32

    All runs: mdn_components=3, S27-A2 base config, full data (8.6M).
    """
    analysis_quick_overrides = {
        "train_regime_diagnostics_enabled": False,
        "mini_reanalysis_enabled": True,
        "mini_reanalysis_scope": "all12",
        "mini_reanalysis_max_samples_per_regime": 4096,
        "grid_ranking_mode": "mini_protocol_v1",
        "batch_infer": 16384,
    }

    _base = dict(
        arch_variant="seq_bigru_residual",
        layer_sizes=[128, 256, 512],
        latent_dim=8,
        beta=0.002,
        free_bits=0.10,
        lr=2e-4,
        batch_size=6144,
        kl_anneal_epochs=80,
        window_size=7,
        window_stride=1,
        window_pad_mode="edge",
        seq_hidden_size=64,
        seq_num_layers=1,
        seq_bidirectional=True,
        seq_gru_unroll=True,
        lambda_mmd=0.25,
        mmd_mode="mean_residual",
        lambda_axis=0.01,
        lambda_psd=0.0,
        lambda_coverage=0.25,
        coverage_levels=[0.50, 0.80, 0.95],
        tail_levels=[0.05, 0.95],
        coverage_temperature=0.03,
        lambda_kurt=0.0,
        decoder_distribution="mdn",
        mdn_components=3,
        cond_embed_dim=0,
        shuffle_train_batches=True,
    )

    def _variant(overrides):
        cfg = dict(_base)
        cfg.update(overrides)
        return _cfg(**cfg)

    return [
        # CTRL — exact S27 A2 winner (embed=0, third seed for variance estimate)
        dict(
            group="S30_cond_embed_sweep",
            tag="S30ce_ctrl_embed0",
            cfg=_variant({}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # A1 — 16-unit conditioning embedding
        dict(
            group="S30_cond_embed_sweep",
            tag="S30ce_embed16",
            cfg=_variant({"cond_embed_dim": 16}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # A2 — 32-unit conditioning embedding
        dict(
            group="S30_cond_embed_sweep",
            tag="S30ce_embed32",
            cfg=_variant({"cond_embed_dim": 32}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
    ]


def _preset_seq_cond_embed_tuned_sweep() -> List[Dict[str, Any]]:
    """S31 — Decoder conditioning embedding: tuned LR, larger embed, residual skip.

    S30 identified cond_embed_dim as the right direction (δJBrel 2990→0.375 for
    embed16, 5 vs 8 G5 failures for embed32) but convergence was poor:
    - embed16 never triggered early stopping (still improving at epoch 500)
    - embed32 stopped at epoch 98 (undertrained, active_dims=7)

    This sweep addresses convergence with four targeted variants:
      CTRL: embed=16, lr=4e-4  (2× higher LR to accelerate embed16)
      A1:   embed=32, lr=4e-4  (higher LR for embed32)
      A2:   embed=64, lr=2e-4  (larger capacity, standard LR)
      A3:   embed=32, lr=4e-4, residual=True  (raw (d,c) + embed skip connection)

    All runs: mdn_components=3, S27-A2 base, full data (8.6M).
    patience=80 (vs default 60) to give embed variants enough exploration time.
    """
    analysis_quick_overrides = {
        "train_regime_diagnostics_enabled": False,
        "mini_reanalysis_enabled": True,
        "mini_reanalysis_scope": "all12",
        "mini_reanalysis_max_samples_per_regime": 4096,
        "grid_ranking_mode": "mini_protocol_v1",
        "batch_infer": 16384,
    }

    _base = dict(
        arch_variant="seq_bigru_residual",
        layer_sizes=[128, 256, 512],
        latent_dim=8,
        beta=0.002,
        free_bits=0.10,
        batch_size=6144,
        kl_anneal_epochs=80,
        window_size=7,
        window_stride=1,
        window_pad_mode="edge",
        seq_hidden_size=64,
        seq_num_layers=1,
        seq_bidirectional=True,
        seq_gru_unroll=True,
        lambda_mmd=0.25,
        mmd_mode="mean_residual",
        lambda_axis=0.01,
        lambda_psd=0.0,
        lambda_coverage=0.25,
        coverage_levels=[0.50, 0.80, 0.95],
        tail_levels=[0.05, 0.95],
        coverage_temperature=0.03,
        lambda_kurt=0.0,
        decoder_distribution="mdn",
        mdn_components=3,
        cond_embed_residual=False,
        shuffle_train_batches=True,
        patience=80,
    )

    def _variant(overrides):
        cfg = dict(_base)
        cfg.update(overrides)
        return _cfg(**cfg)

    return [
        # CTRL — embed=16 with higher LR (embed16 was still converging at epoch 500)
        dict(
            group="S31_cond_embed_tuned",
            tag="S31ce_e16_lr4e4",
            cfg=_variant({"cond_embed_dim": 16, "lr": 4e-4}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # A1 — embed=32 with higher LR (embed32 was undertrained at epoch 98)
        dict(
            group="S31_cond_embed_tuned",
            tag="S31ce_e32_lr4e4",
            cfg=_variant({"cond_embed_dim": 32, "lr": 4e-4}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # A2 — embed=64 with standard LR (larger capacity)
        dict(
            group="S31_cond_embed_tuned",
            tag="S31ce_e64_lr2e4",
            cfg=_variant({"cond_embed_dim": 64, "lr": 2e-4}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # A3 — embed=32 + residual skip: raw (d,c) preserved alongside embedding
        dict(
            group="S31_cond_embed_tuned",
            tag="S31ce_e32_resid_lr4e4",
            cfg=_variant({"cond_embed_dim": 32, "lr": 4e-4, "cond_embed_residual": True}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
    ]


def _preset_seq_cond_embed_large_sweep() -> List[Dict[str, Any]]:
    """S32 — Large decoder conditioning sweep: 6 variants covering unexplored axes.

    S30/S31 findings:
    - cond_embed_dim is the right direction: δJBrel 2990→0.375 for embed16/64
    - LR=4e-4 too high (crashes embed32 at epoch 11, embed16 at epoch 40)
    - embed=32 undertrained (S30, epoch 98, active_dims=7) had 5 G5 failures —
      best ever; 7D effective latent may be key
    - embed=64 lr=2e-4 converges well but still 8 G5 failures (gate too tight)
    - residual at lr=4e-4 stabilises training but sacrifices shape matching
    - MDN+embed and deeper embed are unexplored

    Six variants on S27-A2 base (patience=80):
      A: embed=64, lr=3e-4       — intermediate LR for e64
      B: embed=32 residual, lr=2e-4  — residual at safe LR (unexplored)
      C: embed=64 residual, lr=2e-4  — largest residual at safe LR (unexplored)
      D: embed=32, 3-layer MLP, lr=2e-4  — deeper embedding
      E: embed=64 + mdn=5, lr=2e-4  — best shape-match embed + more MDN
      F: embed=32, latent_dim=7, lr=2e-4  — replicate S30 7D effective latent
    """
    analysis_quick_overrides = {
        "train_regime_diagnostics_enabled": False,
        "mini_reanalysis_enabled": True,
        "mini_reanalysis_scope": "all12",
        "mini_reanalysis_max_samples_per_regime": 4096,
        "grid_ranking_mode": "mini_protocol_v1",
        "batch_infer": 16384,
    }

    _base = dict(
        arch_variant="seq_bigru_residual",
        layer_sizes=[128, 256, 512],
        latent_dim=8,
        beta=0.002,
        free_bits=0.10,
        lr=2e-4,
        batch_size=6144,
        kl_anneal_epochs=80,
        window_size=7,
        window_stride=1,
        window_pad_mode="edge",
        seq_hidden_size=64,
        seq_num_layers=1,
        seq_bidirectional=True,
        seq_gru_unroll=True,
        lambda_mmd=0.25,
        mmd_mode="mean_residual",
        lambda_axis=0.01,
        lambda_psd=0.0,
        lambda_coverage=0.25,
        coverage_levels=[0.50, 0.80, 0.95],
        tail_levels=[0.05, 0.95],
        coverage_temperature=0.03,
        lambda_kurt=0.0,
        decoder_distribution="mdn",
        mdn_components=3,
        cond_embed_dim=0,
        cond_embed_layers=2,
        cond_embed_residual=False,
        shuffle_train_batches=True,
        patience=80,
    )

    def _variant(overrides):
        cfg = dict(_base)
        cfg.update(overrides)
        return _cfg(**cfg)

    return [
        # A — embed=64, lr=3e-4 (intermediate LR; lr=2e-4 converged ok, lr=4e-4 crashes)
        dict(
            group="S32_cond_embed_large",
            tag="S32A_e64_lr3e4",
            cfg=_variant({"cond_embed_dim": 64, "lr": 3e-4}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # B — embed=32 + residual, lr=2e-4 (residual at safe LR, unexplored)
        dict(
            group="S32_cond_embed_large",
            tag="S32B_e32_resid_lr2e4",
            cfg=_variant({"cond_embed_dim": 32, "cond_embed_residual": True}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # C — embed=64 + residual, lr=2e-4 (largest residual at safe LR)
        dict(
            group="S32_cond_embed_large",
            tag="S32C_e64_resid_lr2e4",
            cfg=_variant({"cond_embed_dim": 64, "cond_embed_residual": True}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # D — embed=32, 3-layer MLP, lr=2e-4 (deeper embedding)
        dict(
            group="S32_cond_embed_large",
            tag="S32D_e32_3layer_lr2e4",
            cfg=_variant({"cond_embed_dim": 32, "cond_embed_layers": 3}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # E — embed=64 + mdn=5, lr=2e-4 (combine best shape-match embed with more MDN)
        dict(
            group="S32_cond_embed_large",
            tag="S32E_e64_mdn5_lr2e4",
            cfg=_variant({"cond_embed_dim": 64, "mdn_components": 5}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # F — embed=32, latent_dim=7, lr=2e-4 (S30 embed32 had active_dims=7 and 5 G5 fails)
        dict(
            group="S32_cond_embed_large",
            tag="S32F_e32_lat7_lr2e4",
            cfg=_variant({"cond_embed_dim": 32, "latent_dim": 7}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
    ]


def _preset_seq_mdn_v2_0p8m_isolation() -> List[Dict[str, Any]]:
    """Diagnostic: MDN v2 lat8 trained ONLY on 0.8m data.

    Tests the hypothesis that 0.8m gate failures come from inter-distance
    conflict in the global model. If a model trained exclusively on 0.8m
    passes gates that the global model fails, the problem is conditioning,
    not architecture capacity.
    """
    analysis_quick_overrides = {
        "train_regime_diagnostics_enabled": False,
        "mini_reanalysis_enabled": True,
        "mini_reanalysis_scope": "all12",
        "mini_reanalysis_max_samples_per_regime": 4096,
        "grid_ranking_mode": "mini_protocol_v1",
        "batch_infer": 16384,
    }

    _base = dict(
        arch_variant="seq_bigru_residual",
        layer_sizes=[128, 256, 512],
        latent_dim=8,
        beta=0.002,
        free_bits=0.10,
        lr=2e-4,
        batch_size=6144,
        kl_anneal_epochs=80,
        window_size=7,
        window_stride=1,
        window_pad_mode="edge",
        seq_hidden_size=64,
        seq_num_layers=1,
        seq_bidirectional=True,
        seq_gru_unroll=True,
        lambda_mmd=0.25,
        mmd_mode="mean_residual",
        lambda_axis=0.01,
        lambda_psd=0.0,
        lambda_coverage=0.06,
        coverage_levels=[0.50, 0.80, 0.95],
        tail_levels=[0.05, 0.95],
        coverage_temperature=0.03,
        decoder_distribution="mdn",
        mdn_components=3,
        shuffle_train_batches=True,
    )

    return [
        dict(
            group="S26_seq_mdn_v2_0p8m_isolation",
            tag="S26iso08_lat8_W7_h64_mdn3_cov0p06_t0p03_L128-256-512",
            cfg=_cfg(**_base),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
    ]


def _preset_seq_gaussian_reference_full() -> List[Dict[str, Any]]:
    """Phase 2: Gaussian decoder reference for MDN v2 ceiling comparison.

    Same architecture and hyperparameters as the MDN v2 lat8 champion,
    but with decoder_distribution='gaussian' instead of 'mdn'.
    Also includes the S25 baseline config (lat6) with gaussian decoder.

    Intended to run WITHOUT --max_samples_per_exp (all ~8.6 M train samples)
    and with full protocol evaluation for a direct comparison.
    """
    analysis_quick_overrides = {
        "train_regime_diagnostics_enabled": False,
        "mini_reanalysis_enabled": True,
        "mini_reanalysis_scope": "all12",
        "mini_reanalysis_max_samples_per_regime": 4096,
        "grid_ranking_mode": "mini_protocol_v1",
        "batch_infer": 16384,
    }

    _base = dict(
        arch_variant="seq_bigru_residual",
        layer_sizes=[128, 256, 512],
        latent_dim=6,
        beta=0.002,
        free_bits=0.10,
        lr=2e-4,
        batch_size=6144,
        kl_anneal_epochs=80,
        window_size=7,
        window_stride=1,
        window_pad_mode="edge",
        seq_hidden_size=64,
        seq_num_layers=1,
        seq_bidirectional=True,
        seq_gru_unroll=True,
        lambda_mmd=0.25,
        mmd_mode="mean_residual",
        lambda_axis=0.01,
        lambda_psd=0.0,
        lambda_coverage=0.06,
        coverage_levels=[0.50, 0.80, 0.95],
        tail_levels=[0.05, 0.95],
        coverage_temperature=0.03,
        decoder_distribution="gaussian",
        shuffle_train_batches=True,
    )

    def _variant(overrides: Dict[str, Any]) -> Dict[str, Any]:
        cfg = dict(_base)
        cfg.update(overrides)
        return _cfg(**cfg)

    return [
        # 0 — Gaussian baseline (lat6, same as S25 but gaussian decoder)
        dict(
            group="S26_seq_gaussian_reference",
            tag="S26gauss_lat6_W7_h64_cov0p06_t0p03_L128-256-512",
            cfg=_variant({}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        # 1 — Gaussian with lat8 (match MDN v2 champion architecture)
        dict(
            group="S26_seq_gaussian_reference",
            tag="S26gauss_lat8_W7_h64_cov0p06_t0p03_L128-256-512",
            cfg=_variant({"latent_dim": 8}),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
    ]


def _preset_seq_mdn_v2_a600_tail_explore_quick() -> List[Dict[str, Any]]:
    """A600-only overnight grid focused on tail calibration headroom.

    This preset complements the 5090-safe overnight:

      - keep the fast A600 path with `seq_gru_unroll=False`
      - open a dedicated sweep over `tail_levels`
      - include two structural probes that are intentionally left on the faster
        GRU path because this stack has already tolerated them

    Scientific intent:

      - test whether the remaining `G5` gap is more sensitive to tail shaping
        than to another round of plain coverage-temperature tuning
      - probe whether a small capacity increase helps once tails are tightened
    """

    analysis_quick_overrides = {
        "mini_reanalysis_enabled": True,
        "mini_reanalysis_scope": "all12",
        "mini_reanalysis_max_samples_per_regime": 4096,
        "grid_ranking_mode": "mini_protocol_v1",
        "batch_infer": 16384,
    }

    def _seq_cfg(
        *,
        lambda_coverage: float,
        coverage_temperature: float,
        tail_levels: List[float],
        lambda_axis: float = 0.01,
        latent_dim: int = 4,
        seq_hidden_size: int = 64,
        batch_size: int = 8192,
    ) -> Dict[str, Any]:
        return _cfg(
            arch_variant="seq_bigru_residual",
            layer_sizes=[128, 256, 512],
            latent_dim=latent_dim,
            beta=0.002,
            free_bits=0.10,
            lr=2e-4,
            batch_size=batch_size,
            kl_anneal_epochs=80,
            window_size=7,
            window_stride=1,
            window_pad_mode="edge",
            seq_hidden_size=seq_hidden_size,
            seq_num_layers=1,
            seq_bidirectional=True,
            seq_gru_unroll=False,
            lambda_mmd=0.25,
            mmd_mode="mean_residual",
            lambda_axis=lambda_axis,
            lambda_psd=0.0,
            lambda_coverage=lambda_coverage,
            coverage_levels=[0.50, 0.80, 0.95],
            tail_levels=tail_levels,
            coverage_temperature=coverage_temperature,
            decoder_distribution="mdn",
            mdn_components=3,
            shuffle_train_batches=True,
        )

    return [
        dict(
            group="S26_seq_mdn_v2_a600_tail",
            tag="S26seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p06_tail05-95_t0p03_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(
                lambda_coverage=0.06,
                coverage_temperature=0.03,
                tail_levels=[0.05, 0.95],
            ),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S26_seq_mdn_v2_a600_tail",
            tag="S26seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p06_tail02-98_t0p03_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(
                lambda_coverage=0.06,
                coverage_temperature=0.03,
                tail_levels=[0.02, 0.98],
            ),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S26_seq_mdn_v2_a600_tail",
            tag="S26seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p06_tail01-99_t0p03_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(
                lambda_coverage=0.06,
                coverage_temperature=0.03,
                tail_levels=[0.01, 0.99],
            ),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S26_seq_mdn_v2_a600_tail",
            tag="S26seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p07_tail02-98_t0p03_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(
                lambda_coverage=0.07,
                coverage_temperature=0.03,
                tail_levels=[0.02, 0.98],
            ),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S26_seq_mdn_v2_a600_tail",
            tag="S26seq_W7_h64_lat4_mdn3_b0p002_lmmd0p25_axis0p0125_cov0p06_tail02-98_t0p03_bs8192_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(
                lambda_coverage=0.06,
                coverage_temperature=0.03,
                tail_levels=[0.02, 0.98],
                lambda_axis=0.0125,
            ),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S26_seq_mdn_v2_a600_tail",
            tag="S26seq_W7_h64_lat6_mdn3_b0p002_lmmd0p25_axis0p01_cov0p06_tail02-98_t0p03_bs6144_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(
                lambda_coverage=0.06,
                coverage_temperature=0.03,
                tail_levels=[0.02, 0.98],
                latent_dim=6,
                batch_size=6144,
            ),
            analysis_quick_overrides=analysis_quick_overrides,
        ),
        dict(
            group="S26_seq_mdn_v2_a600_tail",
            tag="S26seq_W7_h96_lat4_mdn3_b0p002_lmmd0p25_axis0p01_cov0p06_tail02-98_t0p03_bs6144_bi16384_gruroll0_fb0p10_lr0p0002_L128-256-512",
            cfg=_seq_cfg(
                lambda_coverage=0.06,
                coverage_temperature=0.03,
                tail_levels=[0.02, 0.98],
                seq_hidden_size=96,
                batch_size=6144,
            ),
            analysis_quick_overrides=analysis_quick_overrides,
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


def _preset_delta_residual_adv_smoke() -> List[Dict[str, Any]]:
    """Single-item smoke preset for the adversarial residual-target variant."""
    base_cfg = dict(
        layer_sizes=[128, 256, 512],
        latent_dim=4,
        arch_variant="delta_residual_adv",
        lambda_adv=0.05,
        disc_layer_sizes=(128, 128),
    )
    return [
        dict(
            group="A0_adv_smoke",
            tag=f"A0adv_lat4_b{_tag_beta(0.001)}_fb0p10_lr{_tag_lr(3e-4)}_L{_tag_layers([128,256,512])}",
            cfg=_cfg(beta=0.001, free_bits=0.10, **base_cfg),
        ),
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
        elif preset_name == "seq_mdn_regime_weight_quick":
            grid = _preset_seq_mdn_regime_weight_quick()
        elif preset_name == "seq_mdn_g5_shape_quick":
            grid = _preset_seq_mdn_g5_shape_quick()
        elif preset_name == "seq_mdn_structure_quick":
            grid = _preset_seq_mdn_structure_quick()
        elif preset_name == "seq_mdn_v2_quick":
            grid = _preset_seq_mdn_v2_quick()
        elif preset_name == "seq_mdn_v2_perf_compare_quick":
            grid = _preset_seq_mdn_v2_perf_compare_quick()
        elif preset_name == "seq_mdn_v2_fastbase_quick":
            grid = _preset_seq_mdn_v2_fastbase_quick()
        elif preset_name == "seq_mdn_v2_g5_followup_quick":
            grid = _preset_seq_mdn_v2_g5_followup_quick()
        elif preset_name == "seq_mdn_v2_overnight_decision_quick":
            grid = _preset_seq_mdn_v2_overnight_decision_quick()
        elif preset_name == "seq_mdn_v2_overnight_5090safe_quick":
            grid = _preset_seq_mdn_v2_overnight_5090safe_quick()
        elif preset_name == "seq_mdn_v2_ceiling_probe_quick":
            grid = _preset_seq_mdn_v2_ceiling_probe_quick()
        elif preset_name == "seq_mdn_v2_ceiling_full":
            grid = _preset_seq_mdn_v2_ceiling_full()
        elif preset_name == "seq_coverage_tail_sweep":
            grid = _preset_seq_coverage_tail_sweep()
        elif preset_name == "seq_kurt_loss_sweep":
            grid = _preset_seq_kurt_loss_sweep()
        elif preset_name == "seq_mdn_components_sweep":
            grid = _preset_seq_mdn_components_sweep()
        elif preset_name == "seq_cond_embed_sweep":
            grid = _preset_seq_cond_embed_sweep()
        elif preset_name == "seq_cond_embed_tuned_sweep":
            grid = _preset_seq_cond_embed_tuned_sweep()
        elif preset_name == "seq_cond_embed_large_sweep":
            grid = _preset_seq_cond_embed_large_sweep()
        elif preset_name == "seq_mdn_v2_0p8m_isolation":
            grid = _preset_seq_mdn_v2_0p8m_isolation()
        elif preset_name == "seq_gaussian_reference_full":
            grid = _preset_seq_gaussian_reference_full()
        elif preset_name == "seq_mdn_v2_a600_tail_explore_quick":
            grid = _preset_seq_mdn_v2_a600_tail_explore_quick()
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
