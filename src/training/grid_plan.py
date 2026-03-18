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
