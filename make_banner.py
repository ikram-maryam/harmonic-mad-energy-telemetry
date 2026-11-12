import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use("dark_background")
fig = plt.figure(figsize=(12, 3.8), dpi=300)
fig.patch.set_facecolor("#0b1120")

# Left: 3D Surface of Diurnal Energy Telemetry & Anomaly Ridge
ax1 = fig.add_subplot(121, projection="3d")
ax1.set_facecolor("#0b1120")

days = np.linspace(1, 30, 60)
hours = np.linspace(0, 24, 60)
D, H = np.meshgrid(days, hours)
Z = 45 + 32 * np.exp(-0.5 * ((H - 13.5) / 3.2) ** 2) + 6 * np.sin(2 * np.pi * H / 24)
# Inject 3D nocturnal anomaly ridge
ridge = ((D >= 10) & (D <= 14) & (H >= 0.5) & (H <= 4.5))
Z[ridge] += 22.0

surf = ax1.plot_surface(D, H, Z, cmap="plasma", edgecolor="none", alpha=0.92, antialiased=True)
ax1.view_init(elev=28, azim=-55)
ax1.set_xlabel("Day", fontsize=7.5, labelpad=-2, color="#94a3b8")
ax1.set_ylabel("Hour", fontsize=7.5, labelpad=-2, color="#94a3b8")
ax1.set_zlabel("Load (kW)", fontsize=7.5, labelpad=-2, color="#94a3b8")
ax1.tick_params(labelsize=6, colors="#64748b")
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
ax1.grid(False)
ax1.set_title("3D Telemetry Manifold & Nocturnal Drift Ridge", fontsize=8.5, color="#38bdf8", pad=4)

# Right: Typography & Research Architecture Card
ax2 = fig.add_subplot(122)
ax2.set_facecolor("#0b1120")
ax2.axis("off")
ax2.text(0.02, 0.80, "HARMONIC-MAD TELEMETRY ENGINE", fontsize=13.5, fontweight="bold", color="#38bdf8")
ax2.text(0.02, 0.65, "Mitigating Statistical Masking in Sub-Hourly Smart Meter Data", fontsize=9.5, color="#e2e8f0")
ax2.text(0.02, 0.46, "• Fourier Harmonic Decomposition (K = 2)\n• Causal Rolling MAD Scale Estimator (50% Breakdown)\n• 5.1x F1-Score Gain Over Classical Z-Score (0.578 vs 0.113)", fontsize=8.5, color="#94a3b8", linespacing=1.6)
ax2.text(0.02, 0.14, "Maryam Ikram  |  BS Mathematics & Data Science (COMSATS Lahore)", fontsize=8, color="#2dd4bf", fontweight="bold")

plt.tight_layout()
plt.savefig("banner_3d.png", facecolor="#0b1120")
plt.close()
print("Saved banner_3d.png!")