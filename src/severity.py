"""Claim severity: Gamma GLM vs LightGBM (Gamma objective) on the average capped claim per policy,
weighted by the number of claims. Only policies with recorded claim amounts are used."""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from config import MODEL_DIR, PROCESSED, REPORTS, SEED
from features import GLM_FORMULA, gbm_frame, glm_frame
from metrics import gamma_deviance


def main() -> None:
    df = glm_frame(pd.read_parquet(PROCESSED / "policies.parquet"))
    df = df[(df["n_sev"] > 0) & (df["amount_capped"] > 0)].copy()
    df["avg_claim"] = df["amount_capped"] / df["n_sev"]
    tr, va, te = (df[df.split == s].copy() for s in ("train", "valid", "test"))
    print(f"Severity rows: train {len(tr):,}  valid {len(va):,}  test {len(te):,}")

    glm = smf.glm(f"avg_claim ~ {GLM_FORMULA}", data=tr, family=sm.families.Gamma(sm.families.links.Log()),
                  var_weights=tr["n_sev"]).fit()

    dtr = lgb.Dataset(gbm_frame(tr), tr["avg_claim"], weight=tr["n_sev"], free_raw_data=False)
    dva = lgb.Dataset(gbm_frame(va), va["avg_claim"], weight=va["n_sev"], reference=dtr)
    params = {"objective": "gamma", "learning_rate": 0.03, "num_leaves": 7, "min_data_in_leaf": 200,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
              "verbosity": -1, "seed": SEED, "num_threads": 8}
    lgbm = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False)])

    mean_sev = float(np.average(tr["avg_claim"], weights=tr["n_sev"]))

    # Model selection on validation: keep a severity model only if it beats the portfolio average.
    valid_dev = {
        "Null (average claim)": gamma_deviance(va["avg_claim"], np.full(len(va), mean_sev), va["n_sev"]),
        "Gamma GLM": gamma_deviance(va["avg_claim"], glm.predict(va).values, va["n_sev"]),
        "LightGBM (Gamma)": gamma_deviance(va["avg_claim"], lgbm.predict(gbm_frame(va), num_iteration=lgbm.best_iteration), va["n_sev"]),
    }
    selected = min(valid_dev, key=valid_dev.get)
    print("Validation Gamma deviance:", {k: round(v, 5) for k, v in valid_dev.items()}, "-> selected:", selected)

    preds = {
        "Null (average claim)": np.full(len(te), mean_sev),
        "Gamma GLM": glm.predict(te).values,
        "LightGBM (Gamma)": lgbm.predict(gbm_frame(te), num_iteration=lgbm.best_iteration),
    }
    results = {}
    for name, mu in preds.items():
        results[name] = {"gamma_deviance": round(gamma_deviance(te["avg_claim"], mu, te["n_sev"]), 5),
                         "mean_predicted": round(float(np.average(mu, weights=te["n_sev"])), 1),
                         "mean_actual": round(float(np.average(te["avg_claim"], weights=te["n_sev"])), 1)}
        print(f"{name:24s} {results[name]}")
    null_dev = results["Null (average claim)"]["gamma_deviance"]
    for name in ("Gamma GLM", "LightGBM (Gamma)"):
        results[name]["deviance_explained_vs_null"] = round(1 - results[name]["gamma_deviance"] / null_dev, 4)
    results["_meta"] = {"lgbm_iterations": int(lgbm.best_iteration), "mean_severity_train": round(mean_sev, 1),
                        "valid_gamma_deviance": {k: round(v, 5) for k, v in valid_dev.items()},
                        "selected_on_validation": selected}
    (REPORTS / "severity_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    joblib.dump({"glm": glm, "lgbm": lgbm, "mean_sev": mean_sev, "selected": selected}, MODEL_DIR / "severity.joblib")


if __name__ == "__main__":
    main()
