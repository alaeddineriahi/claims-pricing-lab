"""Feature preparation for the GLMs (actuarial binning) and for gradient boosting (raw factors).

GLM preprocessing follows Noll, Salzmann & Wüthrich (2018): capped vehicle power as categories,
binned vehicle and driver age, capped bonus-malus, log density, area as an ordinal score.
"""
import numpy as np
import pandas as pd

CAT_GBM = ["VehBrand", "VehGas", "Region", "Area"]
NUM_GBM = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
GBM_FEATURES = NUM_GBM + CAT_GBM

GLM_FORMULA = ("C(VehPowerGLM) + C(VehAgeGLM) + C(DrivAgeGLM) + BonusMalusGLM + C(VehBrand) "
               "+ C(VehGas) + logDensity + AreaGLM + C(Region)")


def glm_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["VehPowerGLM"] = out["VehPower"].clip(upper=9).astype(int)
    out["VehAgeGLM"] = pd.cut(out["VehAge"], [-1, 0, 10, 200], labels=["0", "1-10", "11+"]).astype(str)
    out["DrivAgeGLM"] = pd.cut(out["DrivAge"], [17, 20, 25, 30, 40, 50, 70, 200],
                               labels=["18-20", "21-25", "26-30", "31-40", "41-50", "51-70", "71+"]).astype(str)
    out["BonusMalusGLM"] = out["BonusMalus"].clip(upper=150)
    out["logDensity"] = np.log(out["Density"].clip(lower=1))
    out["AreaGLM"] = out["Area"].map({k: i for i, k in enumerate("ABCDEF", start=1)})
    out["logExposure"] = np.log(out["Exposure"])
    return out


def gbm_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df[GBM_FEATURES].copy()
    for c in CAT_GBM:
        X[c] = X[c].astype("category")
    return X
