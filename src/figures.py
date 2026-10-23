"""Figures for the README and LinkedIn (computed from saved test predictions and models)."""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIG_DIR, MODEL_DIR, PROCESSED, REPORTS
from features import gbm_frame, glm_frame

C_GLM, C_GBM = "#718096", "#2b6cb0"


def lorenz(y, rate, w):
    o = np.argsort(rate)
    w, y = np.asarray(w)[o], np.asarray(y)[o]
    return np.concatenate([[0], np.cumsum(w) / w.sum()]), np.concatenate([[0], np.cumsum(y) / y.sum()])


def main() -> None:
    fr = json.loads((REPORTS / "frequency_results.json").read_text(encoding="utf-8"))
    pr = json.loads((REPORTS / "pricing_results.json").read_text(encoding="utf-8"))
    tp = pd.read_parquet(REPORTS / "test_frequency_predictions.parquet")

    # 1. Lorenz curves, frequency
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for col, name, c in (("freq_glm", "Poisson GLM", C_GLM), ("freq_lgbm", "LightGBM (Poisson)", C_GBM)):
        x, y = lorenz(tp["ClaimNb"], tp[col], tp["Exposure"])
        ax.plot(x, y, color=c, lw=1.8, label=f"{name} (Gini {fr[name]['lorenz_gini']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Tarif uniforme")
    ax.set_xlabel("Part cumulée de l'exposition (du moins risqué au plus risqué)")
    ax.set_ylabel("Part cumulée des sinistres")
    ax.set_title("Courbes de Lorenz — fréquence (test)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "lorenz_frequency.png", dpi=160); plt.close(fig)

    # 2. Calibration by decile, frequency
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, c, m in (("Poisson GLM", C_GLM, "s"), ("LightGBM (Poisson)", C_GBM, "o")):
        cal = pd.DataFrame(fr["calibration_deciles"][name])
        ax.plot(cal["decile"], cal["predicted_rate"], marker=m, color=c, lw=1.2, label=f"{name} — prédit")
        ax.plot(cal["decile"], cal["actual_rate"], marker=m, color=c, lw=0, alpha=0.5, ms=8, label=f"{name} — observé")
    ax.set_xlabel("Décile de fréquence prédite")
    ax.set_ylabel("Sinistres par année-police")
    ax.set_title("Fréquence prédite vs observée par décile (test)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG_DIR / "calibration_frequency.png", dpi=160); plt.close(fig)

    # 3. Double lift chart
    dl = pd.DataFrame(pr["double_lift_capped"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dl["bucket"], dl["loss_ratio_a"], marker="s", color=C_GLM, label="Tarif GLM")
    ax.plot(dl["bucket"], dl["loss_ratio_b"], marker="o", color=C_GBM, label="Tarif LightGBM")
    ax.axhline(1, color="grey", ls="--", lw=1)
    ax.set_xlabel("Décile du ratio prime LightGBM / prime GLM")
    ax.set_ylabel("Sinistres / prime (loss ratio)")
    ax.set_title("Double lift chart (test, sinistres plafonnés)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "double_lift.png", dpi=160); plt.close(fig)

    # 4. Effect curves: GLM vs LightGBM marginal effect on frequency
    fm = joblib.load(MODEL_DIR / "frequency.joblib")
    base = glm_frame(pd.read_parquet(PROCESSED / "policies.parquet"))
    base = base[base.split == "test"].sample(20000, random_state=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, feat, grid, label in ((axes[0], "DrivAge", np.arange(18, 91, 2), "Âge du conducteur"),
                                  (axes[1], "BonusMalus", np.arange(50, 151, 5), "Bonus-malus")):
        g_glm, g_gbm = [], []
        for v in grid:
            d = base.copy(); d[feat] = v
            d = glm_frame(d.drop(columns=[c for c in d.columns if c.endswith("GLM") or c in ("logDensity", "logExposure")]))
            g_glm.append(fm["glm"].predict(d, offset=np.zeros(len(d))).mean())
            g_gbm.append(fm["lgbm"].predict(gbm_frame(d), num_iteration=fm["lgbm"].best_iteration).mean())
        ax.plot(grid, g_glm, color=C_GLM, lw=1.8, label="Poisson GLM")
        ax.plot(grid, g_gbm, color=C_GBM, lw=1.8, label="LightGBM")
        ax.set_xlabel(label); ax.set_ylabel("Fréquence moyenne prédite")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Effet marginal sur la fréquence (dépendance partielle, 20 000 polices de test)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "effects_frequency.png", dpi=160); plt.close(fig)
    print("Figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
