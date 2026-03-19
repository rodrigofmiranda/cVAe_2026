#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""Cross-reference historical experiments with grid-search configs.

Outputs:
- tested_grid_catalog.csv          all tested grid rows with canonical signature
- champion_runs_crossref.csv       per-experiment champion joined with gate status
- config_signature_rollup.csv      aggregated view by exact config signature
- cross_reference_summary.md       short human-readable summary

By default writes under ``outputs/analysis/grid_history``.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


GATE_COLS = [f"gate_g{i}" for i in range(1, 7)]
CONFIG_COLS = [
    "arch_variant",
    "activation",
    "dropout",
    "layer_sizes",
    "latent_dim",
    "beta",
    "free_bits",
    "lr",
    "batch_size",
    "kl_anneal_epochs",
    "seq_hidden_size",
    "window_size",
    "seq_num_layers",
    "seq_bidirectional",
    "window_stride",
    "window_pad_mode",
    "lambda_mmd",
]


def _norm_scalar(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.12g}"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _norm_layers(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    if isinstance(v, (list, tuple)):
        return ",".join(_norm_scalar(x) for x in v)
    text = str(v).strip()
    if not text:
        return ""
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return ",".join(_norm_scalar(x) for x in parsed)
    except Exception:
        pass
    return text.replace(" ", "")


def _infer_arch_variant(row: Dict[str, Any]) -> str:
    explicit = row.get("arch_variant")
    if explicit is not None and not (isinstance(explicit, float) and math.isnan(explicit)):
        text = str(explicit).strip()
        if text:
            return text
    tag = str(row.get("tag", "") or "")
    group = str(row.get("group", "") or "")
    token = tag or group
    if token.startswith("R"):
        return "channel_residual"
    if token.startswith("S"):
        return "seq_bigru_residual"
    if token.startswith("L"):
        return "legacy_2025_zero_y"
    if token.startswith("D"):
        return "delta_residual"
    return "concat"


def _build_signature(row: Dict[str, Any]) -> str:
    parts: List[str] = []
    local = dict(row)
    local["arch_variant"] = _infer_arch_variant(local)
    for key in CONFIG_COLS:
        if key == "layer_sizes":
            value = _norm_layers(local.get(key))
        else:
            value = _norm_scalar(local.get(key))
        if value != "":
            parts.append(f"{key}={value}")
    return "|".join(parts)


def _truthy(v: Any) -> Optional[bool]:
    if pd.isna(v):
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    text = str(v).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _summary_csv_for(exp_dir: Path) -> Optional[Path]:
    preferred = exp_dir / "tables" / "summary_by_regime_recomputed.csv"
    if preferred.exists():
        return preferred
    fallback = exp_dir / "tables" / "summary_by_regime.csv"
    if fallback.exists():
        return fallback
    return None


def _load_champion_summary(exp_dir: Path) -> Optional[Dict[str, Any]]:
    p = _summary_csv_for(exp_dir)
    if p is None:
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None

    best_tags = [t for t in df.get("best_grid_tag", pd.Series([], dtype=object)).dropna().astype(str).unique() if t]
    best_tag = best_tags[0] if len(best_tags) == 1 else ""

    out: Dict[str, Any] = {
        "run": exp_dir.name,
        "summary_csv": str(p),
        "n_regimes": int(len(df)),
        "best_grid_tag": best_tag,
        "validation_status_counts": json.dumps(
            {str(k): int(v) for k, v in df["validation_status"].fillna("partial").value_counts(dropna=False).items()},
            ensure_ascii=False,
        ),
    }
    for gate in GATE_COLS:
        values = [_truthy(v) for v in df[gate]] if gate in df.columns else []
        valid = [v for v in values if v is not None]
        out[f"{gate}_pass_count"] = int(sum(1 for v in valid if v is True))
        out[f"{gate}_fail_count"] = int(sum(1 for v in valid if v is False))
        out[f"{gate}_all_regimes"] = bool(valid and all(v is True for v in valid))
    out["gates_all_regimes_true"] = int(sum(1 for gate in GATE_COLS if out.get(f"{gate}_all_regimes") is True))
    out["any_pass_status"] = bool((df["validation_status"] == "pass").any())
    out["any_partial_status"] = bool((df["validation_status"] == "partial").any())
    out["all_fail_status"] = bool((df["validation_status"] == "fail").all())
    return out


def _load_grid_catalog(exp_dir: Path) -> pd.DataFrame:
    p = exp_dir / "global_model" / "tables" / "gridsearch_results.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if df.empty:
        return df
    df = df.copy()
    df["run"] = exp_dir.name
    df["arch_variant"] = df.apply(lambda r: _infer_arch_variant(r.to_dict()), axis=1)
    df["config_signature"] = df.apply(lambda r: _build_signature(r.to_dict()), axis=1)
    return df


def _make_markdown(
    champion_df: pd.DataFrame,
    rollup_df: pd.DataFrame,
) -> str:
    lines: List[str] = []
    matched = champion_df.copy()
    if "config_signature" in matched.columns:
        matched = matched[matched["config_signature"].fillna("").astype(str) != ""].copy()
    lines.append("# Grid History Cross Reference")
    lines.append("")
    lines.append(f"- champion runs with gate summary: {len(champion_df)}")
    lines.append(f"- champion runs matched to grid config: {len(matched)}")
    lines.append(f"- unique exact config signatures: {len(rollup_df)}")
    lines.append("")

    if not matched.empty:
        lines.append("## Champion Runs")
        lines.append("")
        cols = ["run", "best_grid_tag", "gates_all_regimes_true"] + [f"{g}_all_regimes" for g in GATE_COLS]
        subset = matched[cols].sort_values(["gates_all_regimes_true", "run"], ascending=[False, True]).head(20)
        lines.append(_df_to_markdown(subset))
        lines.append("")

    repeated = rollup_df[rollup_df["n_trials"] > 1].copy()
    if not repeated.empty:
        lines.append("## Repeated Exact Signatures")
        lines.append("")
        subset = repeated[["arch_variant", "config_signature", "n_trials", "n_champion_runs", "best_score_v2", "max_gates_all_regimes_true"]]
        lines.append(_df_to_markdown(subset.sort_values(["n_trials", "best_score_v2"], ascending=[False, True]).head(30)))
        lines.append("")

    promising = rollup_df[(rollup_df["n_champion_runs"] > 0)].copy()
    if not promising.empty:
        lines.append("## Champion Signature Rollup")
        lines.append("")
        subset = promising[["arch_variant", "config_signature", "n_champion_runs", "best_score_v2", "max_gates_all_regimes_true", "champion_runs"]]
        lines.append(_df_to_markdown(subset.sort_values(["max_gates_all_regimes_true", "best_score_v2"], ascending=[False, True]).head(30)))
        lines.append("")

    return "\n".join(lines) + "\n"


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    cols = [str(c) for c in df.columns]
    rows: List[List[str]] = []
    for _, row in df.iterrows():
        cells: List[str] = []
        for c in df.columns:
            v = row[c]
            if pd.isna(v):
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.6g}")
            else:
                cells.append(str(v))
        rows.append(cells)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([header, sep] + body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outputs-root",
        default=str(ROOT / "outputs"),
        help="Root directory containing exp_* experiment folders.",
    )
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "outputs" / "analysis" / "grid_history"),
        help="Directory for generated cross-reference artifacts.",
    )
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog_parts: List[pd.DataFrame] = []
    champion_rows: List[Dict[str, Any]] = []

    for exp_dir in sorted(outputs_root.glob("exp_*")):
        catalog = _load_grid_catalog(exp_dir)
        if not catalog.empty:
            catalog_parts.append(catalog)
        champion = _load_champion_summary(exp_dir)
        if champion is not None:
            champion_rows.append(champion)

    catalog_df = pd.concat(catalog_parts, ignore_index=True) if catalog_parts else pd.DataFrame()
    champion_df = pd.DataFrame(champion_rows)

    if not catalog_df.empty:
        keep_cols = ["run", "rank", "grid_id", "group", "tag", "config_signature"] + [c for c in CONFIG_COLS if c in catalog_df.columns] + [c for c in [
            "score_v2", "status", "best_epoch", "best_val_loss", "evm_real_%", "evm_pred_%", "delta_evm_%", "snr_real_db", "snr_pred_db", "delta_snr_db",
            "delta_mean_l2", "delta_cov_fro", "delta_skew_l2", "delta_kurt_l2", "delta_psd_l2", "train_time_s"
        ] if c in catalog_df.columns]
        tested_catalog = catalog_df.loc[:, [c for c in keep_cols if c in catalog_df.columns]].copy()
    else:
        tested_catalog = pd.DataFrame()

    if not champion_df.empty and not catalog_df.empty:
        champion_join = champion_df.merge(
            catalog_df.drop_duplicates(subset=["run", "tag"]),
            left_on=["run", "best_grid_tag"],
            right_on=["run", "tag"],
            how="left",
            suffixes=("", "_grid"),
        )
    else:
        champion_join = champion_df.copy()

    if not champion_join.empty and "config_signature" in champion_join.columns:
        grouped = champion_join[champion_join["config_signature"].fillna("").astype(str) != ""].groupby("config_signature", dropna=False)
        roll_rows: List[Dict[str, Any]] = []
        for signature, sub in grouped:
            signature = "" if pd.isna(signature) else str(signature)
            if not signature:
                continue
            runs = sorted({str(v) for v in sub["run"].dropna().tolist()})
            row: Dict[str, Any] = {
                "config_signature": signature,
                "arch_variant": sub["arch_variant"].dropna().iloc[0] if "arch_variant" in sub.columns and sub["arch_variant"].notna().any() else "",
                "n_trials": int(len(catalog_df[catalog_df["config_signature"] == signature])) if signature else 0,
                "n_champion_runs": int(len(sub)),
                "champion_runs": ",".join(runs),
                "best_score_v2": float(pd.to_numeric(sub.get("score_v2"), errors="coerce").min()) if "score_v2" in sub.columns else float("nan"),
                "max_gates_all_regimes_true": int(pd.to_numeric(sub["gates_all_regimes_true"], errors="coerce").max()),
            }
            for gate in GATE_COLS:
                col = f"{gate}_all_regimes"
                if col in sub.columns:
                    vals = [_truthy(v) for v in sub[col]]
                    row[f"{gate}_all_any"] = any(v is True for v in vals)
                    row[f"{gate}_all_count"] = int(sum(1 for v in vals if v is True))
            roll_rows.append(row)
        rollup_df = pd.DataFrame(roll_rows)
    else:
        rollup_df = pd.DataFrame()

    tested_catalog.to_csv(out_dir / "tested_grid_catalog.csv", index=False)
    champion_join.to_csv(out_dir / "champion_runs_crossref.csv", index=False)
    rollup_df.to_csv(out_dir / "config_signature_rollup.csv", index=False)
    (out_dir / "cross_reference_summary.md").write_text(
        _make_markdown(champion_join, rollup_df),
        encoding="utf-8",
    )

    print(f"tested_grid_catalog.csv: {len(tested_catalog)} rows")
    print(f"champion_runs_crossref.csv: {len(champion_join)} rows")
    print(f"config_signature_rollup.csv: {len(rollup_df)} rows")
    print(f"out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
