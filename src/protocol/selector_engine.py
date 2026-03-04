# -*- coding: utf-8 -*-
"""
Selector engine — filter loaded experiments by physical operating point.

Pure selection logic, decoupled from protocol orchestration.  The only
dependency is NumPy (for array mean computation and sorting).

Commit 3V (Phase 2).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# Type alias for a loaded experiment tuple
# (X, Y, D, C, exp_path_str)  — same as returned by load_experiments_as_list
Experiment = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]


def select_experiments(
    inventory: List[Experiment],
    selector: Dict[str, float],
    dist_tol: float = 0.05,
    curr_tol: float = 25.0,
    *,
    label: str = "",
) -> List[Experiment]:
    """Filter *inventory* to experiments matching *selector* within tolerances.

    Parameters
    ----------
    inventory : list of (X, Y, D, C, path_str)
        All loaded experiments (full dataset).
    selector : dict
        Must contain ``distance_m`` and/or ``current_mA``.
        When a key is absent or ``None``, that axis is unconstrained.
    dist_tol : float
        Maximum allowed absolute difference in distance (metres).
    curr_tol : float
        Maximum allowed absolute difference in current (mA).
    label : str, optional
        Human-readable label for error messages (e.g. regime id).

    Returns
    -------
    list of (X, Y, D, C, path_str)
        Filtered and deterministically sorted by (distance, current, path).

    Raises
    ------
    RuntimeError
        If no experiments survive the filter.
    """
    target_dist: Optional[float] = None
    target_curr: Optional[float] = None

    if selector.get("distance_m") is not None:
        target_dist = float(selector["distance_m"])
    if selector.get("current_mA") is not None:
        target_curr = float(selector["current_mA"])

    # No targets → pass-through (no filtering)
    if target_dist is None and target_curr is None:
        return list(inventory)

    filtered: List[Experiment] = []
    for (X, Y, D, C, pth) in inventory:
        exp_dist = float(np.mean(D))
        exp_curr = float(np.mean(C))

        dist_ok = target_dist is None or abs(exp_dist - target_dist) <= dist_tol
        curr_ok = target_curr is None or abs(exp_curr - target_curr) <= curr_tol

        if dist_ok and curr_ok:
            filtered.append((X, Y, D, C, pth))

    # Deterministic sort: (distance, current, path)
    def _sort_key(t: Experiment):
        return (float(np.mean(t[2])), float(np.mean(t[3])), str(t[4]))

    filtered.sort(key=_sort_key)

    if len(filtered) == 0:
        avail = sorted(set(
            (float(np.mean(D)), float(np.mean(C)))
            for (_, _, D, C, _) in inventory
        ))
        tag = f" '{label}'" if label else ""
        raise RuntimeError(
            f"No experiments match selector{tag} "
            f"(target dist={target_dist}, curr={target_curr}, "
            f"tol_dist={dist_tol}, tol_curr={curr_tol}).  "
            f"Available (dist, curr) pairs: {avail}"
        )

    return filtered
