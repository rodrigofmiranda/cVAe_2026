# Dataset Conversion Report

## Configuration
- Anchor: **0.8 m / 100 mA** (change if needed)
- Global `factor_ref` = **4.33799**
- Experiments converted: **27**
- Regimes (dist × curr): 3 × 9

## cVAE Decoder Clamp Recommendation
Based on log(var) of the residual across all regimes:
- `log_var_I` range: **[-4.820, -1.689]**
- `log_var_Q` range: **[-4.823, -1.688]**
- **Recommended decoder clamp: [-5.82, -0.69]** (±1 nat margin)

## Summary Statistics
| Metric | min | mean | max |
|--------|-----|------|-----|
| EVM (%) | 16.060 | 43.261 | 74.616 |
| SNR (dB) | 2.543 | 8.395 | 15.885 |
| Kurtosis excess I | -1.096 | -0.723 | -0.015 |
| Kurtosis excess Q | -1.096 | -0.723 | -0.023 |
| log(var) I | -4.820 | -3.051 | -1.689 |
| log(var) Q | -4.823 | -3.051 | -1.688 |
| Skewness I | -0.003 | 0.000 | 0.005 |

## Figures

### Channel quality (EVM / SNR)
![EVM heatmap](heatmap_evm_pct.png)
![SNR heatmap](heatmap_snr_dB.png)
![EVM line](line_evm.png)
![SNR line](line_snr.png)

### Residual distribution
![Kurtosis I](heatmap_kurt_I.png)
![Kurtosis Q](heatmap_kurt_Q.png)
![Skewness I](heatmap_skew_I.png)

### cVAE decoder clamp calibration
![log(var) I](heatmap_log_var_I.png)
![log(var) Q](heatmap_log_var_Q.png)

### Received power
![Power vs current](line_p_rx.png)
