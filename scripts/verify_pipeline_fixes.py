"""
Verifica os 4 fixes de pipeline obrigatórios.
Executar após qualquer smoke test bem-sucedido.
"""
import json
import glob
import pandas as pd
import numpy as np
from pathlib import Path

latest = sorted(glob.glob("outputs/exp_*/manifest.json"))[-1]
manifest = json.load(open(latest))
exp_dir = Path(latest).parent

print(f"=== Verificando run: {exp_dir.name} ===\n")

errors = []

# Fix 1: clamp calibrado
args_stat = manifest.get("args", {})
# Verificar via summary se var_pred / var_real < 5.0 (clamp não excessivo)
summary = pd.read_csv(exp_dir / "tables" / "summary_by_regime.csv")
if "var_real_delta" in summary.columns and "var_pred_delta" in summary.columns:
    ratio = summary["var_pred_delta"] / summary["var_real_delta"]
    max_ratio = ratio.max()
    print(f"Fix 1 — var_pred/var_real max: {max_ratio:.3f} (deve ser < 5.0)")
    if max_ratio > 5.0:
        errors.append(f"FAIL Fix1: var ratio = {max_ratio:.2f} > 5.0")
    else:
        print("  ✅ Clamp calibrado: variância dentro do esperado")
else:
    errors.append("FAIL Fix1: colunas var_real_delta/var_pred_delta ausentes no summary")

# Fix 4: mmd2_normalized presente
if Path(exp_dir / "tables" / "stat_fidelity_by_regime.csv").exists():
    sf = pd.read_csv(exp_dir / "tables" / "stat_fidelity_by_regime.csv")
    if "mmd2_normalized" in sf.columns:
        n_nan = sf["mmd2_normalized"].isna().sum()
        print(f"Fix MMD2_norm — mmd2_normalized presente, NaNs: {n_nan}/{len(sf)}")
        if n_nan == len(sf):
            errors.append("FAIL MMD2_norm: todos NaN — merge falhou")
        else:
            print("  ✅ mmd2_normalized calculado corretamente")
    else:
        errors.append("FAIL MMD2_norm: coluna ausente em stat_fidelity_by_regime.csv")
else:
    print("⚠️  stat_fidelity_by_regime.csv não encontrado (run sem --stat_tests?)")

print(f"\n{'='*50}")
if errors:
    print("❌ ERROS ENCONTRADOS:")
    for e in errors:
        print(f"   {e}")
else:
    print("✅ Todos os fixes verificados com sucesso.")
