# -*- coding: utf-8 -*-
"""
src.evaluation.stat_tests — Statistical Fidelity Suite (SFS).

Two-sample tests and distance metrics for validating the digital twin
per (d, I) regime.

Sub-modules
-----------
mmd      – RBF-kernel MMD² + permutation p-value
energy   – Energy distance + permutation p-value
psd      – PSD L2 distance + bootstrap CI
fdr      – Benjamini–Hochberg FDR correction

Quick usage::

    from src.evaluation.stat_tests import mmd_rbf, energy_test, psd_distance
    from src.evaluation.stat_tests import benjamini_hochberg

Commit: refactor(etapaA1).
"""

from src.evaluation.stat_tests.mmd import mmd_rbf          # noqa: F401
from src.evaluation.stat_tests.energy import energy_test    # noqa: F401
from src.evaluation.stat_tests.psd import psd_distance      # noqa: F401
from src.evaluation.stat_tests.fdr import benjamini_hochberg # noqa: F401
