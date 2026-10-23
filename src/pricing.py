"""Pure premium = frequency x severity, compared GLM vs LightGBM frequency.

Standard actuarial split of the cost:
- attritional cost: claims capped at P99.5, modelled by frequency x severity and balanced on train so
  that total premium equals total capped claims (this also absorbs claims recorded without an amount);
- large-claim loading: the excess above the cap, spread proportionally (ratio of total to capped claims
  on train), because a handful of claims above 34k carry a quarter of the total cost.
Severity uses the model selected on validation (the portfolio average if no model beats it).
Evaluation on test: balance (loss ratios), Lorenz Gini on capped cost, double lift chart.
"""
import json

import joblib
import numpy as np
import pandas as pd

from config import MODEL_DIR, PROCESSED, REPORTS
from features import gbm_frame, glm_frame
from metrics import lorenz_gini


def severity(df, sev_m):
    sel = sev_m["selected"]
    if sel == "Gamma GLM":
        return sev_m["glm"].predict(df).values
    if sel == "LightGBM (Gamma)":
        return sev_m["lgbm"].predict(gbm_frame(df), num_iteration=sev_m["lgbm"].best_iteration)
    return np.full(len(df), sev_m["mean_sev"])


def pure_premiums(df, freq_m, sev_m):
    sev = severity(df, sev_m)
    glm_freq = freq_m["glm"].predict(df, offset=np.zeros(len(df))).values
    lgb_freq = freq_m["lgbm"].predict(gbm_frame(df), num_iteration=freq_m["lgbm"].best_iteration)
    return {"GLM tariff": glm_freq * sev, "LightGBM tariff": lgb_freq * sev}


def double_lift(actual, pp_a, pp_b, exposure, n=10):
    """Sort by pp_b / pp_a; in each decile compare actual cost with each model's premium."""
    df = pd.DataFrame({"actual": actual, "a": pp_a * exposure, "b": pp_b * exposure, "ratio": pp_b / pp_a})
    df["bucket"] = pd.qcut(df["ratio"].rank(method="first"), n, labels=False) + 1
    g = df.groupby("bucket")[["actual", "a", "b"]].sum()
    return pd.DataFrame({"loss_ratio_a": g["actual"] / g["a"], "loss_ratio_b": g["actual"] / g["b"]}).round(3).reset_index()


def main() -> None:
    df = glm_frame(pd.read_parquet(PROCESSED / "policies.parquet"))
    freq_m, sev_m = joblib.load(MODEL_DIR / "frequency.joblib"), joblib.load(MODEL_DIR / "severity.joblib")
    tr, te = df[df.split == "train"], df[df.split == "test"]
    large_loading = float(tr["amount"].sum() / tr["amount_capped"].sum())

    pp_tr, pp_te = pure_premiums(tr, freq_m, sev_m), pure_premiums(te, freq_m, sev_m)
    results = {"severity_used": sev_m["selected"], "large_claim_loading_from_train": round(large_loading, 4)}
    balanced = {}
    for name in pp_tr:
        factor = tr["amount_capped"].sum() / (pp_tr[name] * tr["Exposure"]).sum()
        balanced[name] = pp_te[name] * factor
        prem_attr = balanced[name] * te["Exposure"].values
        results[name] = {
            "balance_factor_from_train": round(float(factor), 4),
            "test_loss_ratio_attritional": round(float(te["amount_capped"].sum() / prem_attr.sum()), 4),
            "test_loss_ratio_total_with_loading": round(float(te["amount"].sum() / (prem_attr.sum() * large_loading)), 4),
            "lorenz_gini_capped_cost": round(lorenz_gini(te["amount_capped"].values, balanced[name], te["Exposure"].values), 4),
            "mean_attritional_pure_premium": round(float(np.average(balanced[name], weights=te["Exposure"])), 2),
            "mean_total_pure_premium": round(float(np.average(balanced[name], weights=te["Exposure"]) * large_loading), 2),
        }
        print(f"{name:16s} {results[name]}")

    excess_share_test = 1 - te["amount_capped"].sum() / te["amount"].sum()
    results["test_large_claim_excess_share"] = round(float(excess_share_test), 4)
    print(f"Large-claim loading (train): {large_loading:.4f} | excess share of cost on test: {excess_share_test:.3%}")

    a, b = "GLM tariff", "LightGBM tariff"
    # Paired bootstrap of the Gini difference (LightGBM - GLM) on capped cost
    rng = np.random.default_rng(42)
    y, w = te["amount_capped"].values, te["Exposure"].values
    diffs = []
    for _ in range(300):
        idx = rng.integers(0, len(y), len(y))
        diffs.append(lorenz_gini(y[idx], balanced[b][idx], w[idx]) - lorenz_gini(y[idx], balanced[a][idx], w[idx]))
    point = results[b]["lorenz_gini_capped_cost"] - results[a]["lorenz_gini_capped_cost"]
    results["gini_gain_lgbm_vs_glm"] = [round(point, 4), *[round(float(q), 4) for q in np.quantile(diffs, [0.025, 0.975])]]
    print("Gini gain LightGBM vs GLM [95% CI]:", results["gini_gain_lgbm_vs_glm"])

    dl = double_lift(te["amount_capped"].values, balanced[a], balanced[b], te["Exposure"].values)
    results["double_lift_capped"] = dl.to_dict(orient="records")
    print("Double lift (sorted by LightGBM/GLM premium ratio):")
    print(dl.to_string(index=False))
    (REPORTS / "pricing_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    out = te[["IDpol", "Exposure", "amount", "amount_capped", "ClaimNb", "DrivAge", "BonusMalus", "VehAge"]].copy()
    for name, pp in balanced.items():
        out[f"pp_{name}"] = pp
    out.to_parquet(REPORTS / "test_pure_premiums.parquet", index=False)


if __name__ == "__main__":
    main()
