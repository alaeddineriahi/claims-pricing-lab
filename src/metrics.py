"""Actuarial evaluation: Poisson/Gamma deviance, Lorenz-curve Gini, calibration by decile."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance


def poisson_deviance(claims, exposure, rate):
    """Exposure-weighted mean Poisson deviance of the observed frequency vs the predicted rate."""
    return float(mean_poisson_deviance(claims / exposure, rate, sample_weight=exposure))


def gamma_deviance(y, mu, weight=None):
    return float(mean_gamma_deviance(y, mu, sample_weight=weight))


def lorenz_gini(y, pred, weight):
    """Ordered Lorenz curve Gini (Frees, Meyers & Cummings 2011): policies are sorted by predicted
    *rate* per unit of exposure (not by expected amount, which would mix risk with exposure length);
    the curve plots cumulative exposure vs cumulative claims. Higher = better risk ranking."""
    order = np.argsort(pred)
    y, w = np.asarray(y)[order], np.asarray(weight)[order]
    cum_w = np.concatenate([[0], np.cumsum(w) / w.sum()])
    cum_y = np.concatenate([[0], np.cumsum(y) / y.sum()])
    return float(1 - 2 * np.trapezoid(cum_y, cum_w))


def calibration_by_decile(actual, predicted, weight, n=10) -> pd.DataFrame:
    df = pd.DataFrame({"actual": actual, "pred": predicted, "w": weight})
    df["rate_pred"] = df["pred"] / df["w"]
    df["decile"] = pd.qcut(df["rate_pred"].rank(method="first"), n, labels=False) + 1
    g = df.groupby("decile").agg(exposure=("w", "sum"), actual=("actual", "sum"), predicted=("pred", "sum"))
    g["actual_rate"] = g["actual"] / g["exposure"]
    g["predicted_rate"] = g["predicted"] / g["exposure"]
    return g.reset_index()
