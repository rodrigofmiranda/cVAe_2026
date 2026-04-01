# Dataset Conversion Report

## Configuration
- Global `factor_ref` = **4.30989**
- Experiments converted: **27**
- Regimes (dist × curr): 3 × 9

## cVAE Decoder Clamp Recommendation
- `log_var_I` range: **[-4.835, -1.855]**
- `log_var_Q` range: **[-4.836, -1.854]**
- **Recommended decoder clamp: [-5.84, -0.85]** (±1 nat margin)

## Summary Statistics
| Metric | min | mean | max |
|--------|-----|------|-----|
| EVM (%) | 17.487 | 44.274 | 75.184 |
| SNR (dB) | 2.478 | 8.096 | 15.146 |
| Kurtosis excess I | -1.225 | -0.789 | -0.008 |
| Kurtosis excess Q | -1.226 | -0.790 | -0.011 |
| log(var) I | -4.835 | -3.166 | -1.855 |
| log(var) Q | -4.836 | -3.166 | -1.854 |
| Skewness I | -0.002 | -0.000 | 0.003 |

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
