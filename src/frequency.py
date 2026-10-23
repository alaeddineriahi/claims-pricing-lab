"""Claim frequency: Poisson GLM (actuarial standard) vs LightGBM with a Poisson objective.

Both use log(exposure) as an offset, so they model claims per policy-year. Hyper-parameters are
chosen on the validation set; the test set is scored once.
"""
import itertools
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from config import MODEL_DIR, PROCESSED, REPORTS, SEED
from features import GBM_FEATURES, GLM_FORMULA, gbm_frame, glm_frame
from metrics import calibration_by_decile, lorenz_gini, poisson_deviance

MONOTONE = {"BonusMalus": 1}


def unit_poisson_dev(y, mu):
    """Per-row Poisson deviance contributions (for the paired bootstrap)."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return 2 * (term - (y - mu))


def paired_bootstrap(dev_a, dev_b, w, n_boot=1000):
    """Mean exposure-weighted deviance difference (a - b) with a 95% CI."""
    rng = np.random.default_rng(SEED)
    n = len(w)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append((np.sum(w[idx] * dev_a[idx]) - np.sum(w[idx] * dev_b[idx])) / np.sum(w[idx]))
    point = (np.sum(w * dev_a) - np.sum(w * dev_b)) / np.sum(w)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def fit_glm(train: pd.DataFrame):
    model = smf.glm(f"ClaimNb ~ {GLM_FORMULA}", data=train, family=sm.families.Poisson(),
                    offset=train["logExposure"]).fit()
    return model


def fit_lgbm(train: pd.DataFrame, valid: pd.DataFrame):
    Xtr, Xva = gbm_frame(train), gbm_frame(valid)
    dtr = lgb.Dataset(Xtr, train["ClaimNb"], init_score=np.log(train["Exposure"]), free_raw_data=False)
    dva = lgb.Dataset(Xva, valid["ClaimNb"], init_score=np.log(valid["Exposure"]), reference=dtr)
    constraints = [MONOTONE.get(c, 0) for c in GBM_FEATURES]
    best = None
    for leaves, min_leaf in itertools.product([15, 31, 63], [200, 1000]):
        params = {"objective": "poisson", "learning_rate": 0.05, "num_leaves": leaves,
                  "min_data_in_leaf": min_leaf, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                  "bagging_freq": 1, "lambda_l2": 1.0, "monotone_constraints": constraints,
                  "verbosity": -1, "seed": SEED, "num_threads": 8}
        booster = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
        rate = booster.predict(Xva, num_iteration=booster.best_iteration)
        dev = poisson_deviance(valid["ClaimNb"].values, valid["Exposure"].values, rate)
        print(f"  leaves={leaves:3d} min_leaf={min_leaf:5d} iters={booster.best_iteration:4d} valid dev={dev:.5f}")
        if best is None or dev < best[0]:
            best = (dev, booster, params)
    return best[1], best[2]


def main() -> None:
    df = glm_frame(pd.read_parquet(PROCESSED / "policies.parquet"))
    tr, va, te = (df[df.split == s].copy() for s in ("train", "valid", "test"))

    glm = fit_glm(tr)
    print(f"GLM fitted: {len(glm.params)} parameters, AIC {glm.aic:,.0f}")
    lgbm, params = fit_lgbm(tr, va)

    base_rate = tr["ClaimNb"].sum() / tr["Exposure"].sum()
    preds = {
        "Null (portfolio average)": np.full(len(te), base_rate),
        "Poisson GLM": glm.predict(te, offset=te["logExposure"]).values / te["Exposure"].values,
        "LightGBM (Poisson)": lgbm.predict(gbm_frame(te), num_iteration=lgbm.best_iteration),
    }

    y, e = te["ClaimNb"].values, te["Exposure"].values
    results, unit = {}, {}
    for name, rate in preds.items():
        results[name] = {
            "poisson_deviance_x100": round(100 * poisson_deviance(y, e, rate), 4),
            "lorenz_gini": round(lorenz_gini(y, rate, e), 4) if name != "Null (portfolio average)" else 0.0,
            "predicted_claims": round(float((rate * e).sum()), 1),
            "actual_claims": int(y.sum()),
        }
        unit[name] = unit_poisson_dev(y / e, rate)
        print(f"{name:28s} {results[name]}")

    d, lo, hi = paired_bootstrap(unit["LightGBM (Poisson)"], unit["Poisson GLM"], e)
    results["lgbm_minus_glm_deviance_x100"] = [round(100 * d, 4), round(100 * lo, 4), round(100 * hi, 4)]
    print("LightGBM - GLM deviance (x100) [95% CI]:", results["lgbm_minus_glm_deviance_x100"])

    cal = {name: calibration_by_decile(y, rate * e, e).round(5).to_dict(orient="records")
           for name, rate in preds.items() if not name.startswith("Null")}
    results["calibration_deciles"] = cal
    results["_meta"] = {"n_test": int(len(te)), "test_exposure": round(float(e.sum()), 1),
                        "lgbm_params": {k: v for k, v in params.items() if k != "monotone_constraints"},
                        "lgbm_iterations": int(lgbm.best_iteration)}
    (REPORTS / "frequency_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Relativity table (multiplicative factors with 95% CI), the format actuaries price with
    ci = glm.conf_int()
    rel = pd.DataFrame({"coef": glm.params, "relativity": np.exp(glm.params),
                        "ci_low": np.exp(ci[0]), "ci_high": np.exp(ci[1]), "p_value": glm.pvalues}).round(4)
    rel.to_csv(REPORTS / "glm_frequency_relativities.csv")

    out = te[["IDpol", "ClaimNb", "Exposure", "amount", "amount_capped", "n_sev"]].copy()
    out["freq_glm"] = preds["Poisson GLM"]
    out["freq_lgbm"] = preds["LightGBM (Poisson)"]
    out.to_parquet(REPORTS / "test_frequency_predictions.parquet", index=False)
    joblib.dump({"glm": glm, "lgbm": lgbm, "base_rate": base_rate}, MODEL_DIR / "frequency.joblib")


if __name__ == "__main__":
    main()
