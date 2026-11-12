import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# 1. Generate 30-day half-hourly commercial telemetry with ground-truth labels
np.random.seed(42)
N = 1440  # 30 days * 48 intervals/day
timestamps = pd.date_range("2026-10-01", periods=N, freq="30min")
t = np.arange(N)
hours = timestamps.hour + timestamps.minute / 60.0

# Diurnal harmonic load + weekday/weekend modulation
is_weekend = timestamps.weekday >= 5
diurnal = 32 * np.exp(-0.5 * ((hours - 13.5) / 3.2) ** 2) + 8 * np.sin(2 * np.pi * hours / 24.0)
base_load = np.where(is_weekend, 42 + 0.45 * diurnal, 48 + diurnal)
noise = np.random.normal(0, 2.2, size=N)
load_kw = base_load + noise

# Ground truth anomaly mask
y_true = np.zeros(N, dtype=int)

# Regime 1: Sustained multi-day nocturnal HVAC drift (Days 10-13, 00:00 - 04:30)
drift_mask = (timestamps.day.isin([10, 11, 12, 13])) & (timestamps.hour >= 0) & (timestamps.hour <= 4)
load_kw[drift_mask] += 19.5
y_true[drift_mask] = 1

# Regime 2: Transient compressor/equipment spikes
spike_idx = [180, 410, 625, 890, 1050, 1290, 1380]
load_kw[spike_idx] += np.array([34.0, 39.0, 31.0, 44.0, 36.0, 41.0, 33.0])
y_true[spike_idx] = 1

df = pd.DataFrame({"timestamp": timestamps, "load_kw": load_kw, "y_true": y_true})

# 2. Model A: Classical Rolling Z-Score (W = 48)
W = 48
roll_mean = df["load_kw"].rolling(W, min_periods=1).mean()
roll_std = df["load_kw"].rolling(W, min_periods=1).std().fillna(2.2)
df["z_classical"] = (df["load_kw"] - roll_mean) / (roll_std + 1e-6)
pred_z = (df["z_classical"].abs() > 2.5).astype(int)

# 3. Model B: Harmonic-Detrended Robust Rolling MAD Estimator
X_harm = np.column_stack([
    np.ones(N),
    np.sin(2 * np.pi * t / 48.0), np.cos(2 * np.pi * t / 48.0),
    np.sin(4 * np.pi * t / 48.0), np.cos(4 * np.pi * t / 48.0),
    is_weekend.astype(float)
])
beta, _, _, _ = np.linalg.lstsq(X_harm, df["load_kw"], rcond=None)
df["harmonic_baseline"] = X_harm @ beta
df["residual"] = df["load_kw"] - df["harmonic_baseline"]

roll_med = df["residual"].rolling(W, min_periods=1).median()
roll_mad = (df["residual"] - roll_med).abs().rolling(W, min_periods=1).median().fillna(1.5)
df["z_robust"] = (df["residual"] - roll_med) / (1.4826 * roll_mad + 1e-6)
pred_mad = (df["z_robust"].abs() > 2.8).astype(int)

# 4. Compute Comparative Metrics
for name, pred in [("Classical Z-Score", pred_z), ("Harmonic-MAD (Proposed)", pred_mad)]:
    p = precision_score(y_true, pred)
    r = recall_score(y_true, pred)
    f1 = f1_score(y_true, pred)
    print(f"{name} -> Precision: {p:.3f}, Recall: {r:.3f}, F1-Score: {f1:.3f}")

# 5. Publication-Grade Figures
plt.style.use("dark_background")

# Fig 1
fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
sub = df[(df["timestamp"] >= "2026-10-09") & (df["timestamp"] <= "2026-10-15")]
anom_sub = sub[sub["y_true"] == 1]

ax.plot(sub["timestamp"], sub["load_kw"], color="#38bdf8", lw=1.2, label="Observed Telemetry (kW)")
ax.plot(sub["timestamp"], sub["harmonic_baseline"], color="#22c55e", ls="--", lw=1.3, label="Harmonic Fourier Baseline")
ax.scatter(anom_sub["timestamp"], anom_sub["load_kw"], color="#f43f5e", s=32, zorder=5, label="Ground-Truth Anomalies")
ax.set_title("Diurnal Harmonic Decomposition vs. Nocturnal HVAC Drift (Days 10–13)", fontsize=11.5, fontweight="bold")
ax.set_ylabel("Power Demand (kW)")
ax.grid(True, alpha=0.2)
ax.legend(frameon=True, facecolor="#1e293b", edgecolor="none", fontsize=8.5)
plt.tight_layout()
plt.savefig("fig1_harmonic_baseline.png")
plt.close()

# Fig 2
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.2), dpi=300, sharex=True)
ax1.plot(sub["timestamp"], sub["z_classical"], color="#94a3b8", lw=1.1, label="Classical Rolling Z-Score")
ax1.axhline(2.5, color="#f43f5e", ls="--", lw=1.0, label="Threshold (±2.5σ)")
ax1.axhline(-2.5, color="#f43f5e", ls="--", lw=1.0)
ax1.set_ylabel("Classical Z")
ax1.set_title("Mitigating Statistical Masking: Classical Z-Score vs. Robust Harmonic-MAD", fontsize=11.5, fontweight="bold")
ax1.grid(True, alpha=0.2)
ax1.legend(loc="upper right", frameon=True, facecolor="#1e293b", edgecolor="none", fontsize=8)

ax2.plot(sub["timestamp"], sub["z_robust"], color="#fbbf24", lw=1.2, label="Harmonic-MAD Robust Score")
ax2.axhline(2.8, color="#f43f5e", ls="--", lw=1.0, label="Threshold (±2.8σ)")
ax2.axhline(-2.8, color="#f43f5e", ls="--", lw=1.0)
ax2.set_ylabel("Robust Z_MAD")
ax2.grid(True, alpha=0.2)
ax2.legend(loc="upper right", frameon=True, facecolor="#1e293b", edgecolor="none", fontsize=8)

plt.tight_layout()
plt.savefig("fig2_masking_comparison.png")
plt.close()