from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

_ARTIFACT_FIELDS = (
    "run_dir",
    "metrics_path",
    "panel6_path",
    "overlay_path",
    "fingerprint_path",
    "dashboard_path",
)


def compact_16qam_run_tag(raw_tag: str | Path | None) -> str:
    """Collapse verbose historical 16QAM tags to a short canonical name."""
    if raw_tag is None:
        return "all_regimes"
    tag = Path(str(raw_tag).strip()).name
    if not tag:
        return "all_regimes"
    for prefix in ("eval_16qam_", "eval_"):
        if tag.startswith(prefix):
            tag = tag[len(prefix):]
            break
    return tag or "all_regimes"



def build_architecture_16qam_root(
    repo_root: str | Path,
    architecture_family: str,
    candidate_name: str,
    run_tag: str | Path | None,
) -> Path:
    repo = Path(repo_root).resolve()
    return (
        repo
        / "outputs"
        / "architectures"
        / str(architecture_family).strip()
        / str(candidate_name).strip()
        / "16qam"
        / compact_16qam_run_tag(run_tag)
    )



def build_crossline_16qam_root(repo_root: str | Path, run_tag: str | Path | None) -> Path:
    repo = Path(repo_root).resolve()
    return repo / "outputs" / "architectures" / "_crossline" / "16qam" / compact_16qam_run_tag(run_tag)



def build_local_16qam_root(model_parent_dir: str | Path, run_tag: str | Path | None) -> Path:
    return Path(model_parent_dir).resolve() / "16qam" / compact_16qam_run_tag(run_tag)



def relocate_regime_artifact_path(
    value: Any,
    *,
    regime_id: str,
    new_eval_root: str | Path,
) -> Any:
    if not value:
        return value

    text = str(value)
    regime = str(regime_id).strip()
    if not regime:
        return value

    root = Path(new_eval_root).resolve()
    marker = f"{regime}/"

    if text.endswith(regime):
        return str((root / regime).resolve())

    idx = text.rfind(marker)
    if idx >= 0:
        suffix = text[idx + len(marker) :]
        return str((root / regime / suffix).resolve())

    path = Path(text)
    if path.name == regime:
        return str((root / regime).resolve())

    return value



def relocate_manifest_payload(payload: dict[str, Any], new_eval_root: str | Path) -> dict[str, Any]:
    updated = deepcopy(payload)
    results = updated.get("results")
    if not isinstance(results, list):
        return updated

    for row in results:
        if not isinstance(row, dict):
            continue
        regime_id = str(row.get("regime_id", "")).strip()
        if not regime_id:
            continue
        for key in _ARTIFACT_FIELDS:
            row[key] = relocate_regime_artifact_path(
                row.get(key),
                regime_id=regime_id,
                new_eval_root=new_eval_root,
            )

    return updated



def relocate_manifest_rows(rows: list[dict[str, Any]], new_eval_root: str | Path) -> list[dict[str, Any]]:
    updated = deepcopy(rows)
    for row in updated:
        if not isinstance(row, dict):
            continue
        regime_id = str(row.get("regime_id", "")).strip()
        if not regime_id:
            continue
        for key in _ARTIFACT_FIELDS:
            row[key] = relocate_regime_artifact_path(
                row.get(key),
                regime_id=regime_id,
                new_eval_root=new_eval_root,
            )
    return updated
