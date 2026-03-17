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
