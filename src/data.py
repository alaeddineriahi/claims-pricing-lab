"""Download, clean and split freMTPL2 (French motor third-party liability, 678k policies).

Data handling follows the reference case study: Noll, Salzmann & Wüthrich (2018), "Case Study:
French Motor Third-Party Liability Claims", SSRN 3164764; and Wüthrich & Merz (2023), "Statistical
Foundations of Actuarial Learning and its Applications", Springer (open access).
"""
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config import (CLAIMNB_CAP, EXPOSURE_CAP, LARGE_CLAIM_QUANTILE, PROCESSED, RAW, SEED, SPLIT, URLS)

RISK_FACTORS = ["Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region"]


def download() -> None:
    for name, url in URLS.items():
        path = RAW / name
        if not path.exists():
            urllib.request.urlretrieve(url, path)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    freq = pd.read_parquet(RAW / "freMTPL2freq.parquet")
    sev = pd.read_parquet(RAW / "freMTPL2sev.parquet")
    freq["IDpol"] = freq["IDpol"].astype(np.int64)
    sev["IDpol"] = sev["IDpol"].astype(np.int64)
    for c in ["Area", "VehBrand", "Region", "VehGas"]:
        freq[c] = freq[c].astype(str)
    freq["ClaimNb"] = freq["ClaimNb"].astype(int).clip(upper=CLAIMNB_CAP)
    freq["Exposure"] = freq["Exposure"].clip(upper=EXPOSURE_CAP)
    return freq, sev


def group_split(freq: pd.DataFrame) -> pd.Series:
    """Rows sharing all risk factors are likely the same policyholder over several periods
    (Wüthrich & Merz 2023), so splits are made by group to avoid train/test leakage."""
    groups = freq[RISK_FACTORS].astype(str).agg("|".join, axis=1)
    groups = pd.factorize(groups)[0]
    split = pd.Series("train", index=freq.index)
    gss = GroupShuffleSplit(n_splits=1, test_size=SPLIT["test"], random_state=SEED)
    tr_idx, te_idx = next(gss.split(freq, groups=groups))
    split.iloc[te_idx] = "test"
    gss2 = GroupShuffleSplit(n_splits=1, test_size=SPLIT["valid"] / (SPLIT["train"] + SPLIT["valid"]),
                             random_state=SEED)
    tr2, va2 = next(gss2.split(freq.iloc[tr_idx], groups=groups[tr_idx]))
    split.iloc[tr_idx[va2]] = "valid"
    return split, groups


def main() -> None:
    download()
    freq, sev = load()
    freq["split"], freq["group"] = group_split(freq)

    # Severity: claim amounts per policy. Large claims are capped; the excess becomes a loading.
    cap = float(sev["ClaimAmount"].quantile(LARGE_CLAIM_QUANTILE))
    sev["ClaimAmountCapped"] = sev["ClaimAmount"].clip(upper=cap)
    agg = sev.groupby("IDpol").agg(n_sev=("ClaimAmount", "size"), amount=("ClaimAmount", "sum"),
                                   amount_capped=("ClaimAmountCapped", "sum")).reset_index()
    freq = freq.merge(agg, on="IDpol", how="left")
    for c in ["n_sev", "amount", "amount_capped"]:
        freq[c] = freq[c].fillna(0)
    freq["n_sev"] = freq["n_sev"].astype(int)

    orphan_sev = int((~sev["IDpol"].isin(freq["IDpol"])).sum())
    claims_without_amount = int(((freq["ClaimNb"] > 0) & (freq["n_sev"] == 0)).sum())
    print(f"Policies: {len(freq):,}  | claims: {freq.ClaimNb.sum():,}  | severity records: {len(sev):,}")
    print(f"Large-claim cap (P{LARGE_CLAIM_QUANTILE * 100:.1f}): {cap:,.0f}  | excess share of total: "
          f"{1 - sev.ClaimAmountCapped.sum() / sev.ClaimAmount.sum():.3%}")
    print(f"Severity rows without a policy: {orphan_sev}  | policies with claims but no amount: {claims_without_amount:,}")
    print(freq.groupby("split").agg(policies=("IDpol", "size"), exposure=("Exposure", "sum"),
                                    claims=("ClaimNb", "sum"))
          .assign(freq=lambda d: (d.claims / d.exposure).round(4)))
    n_overlap = len(set(freq.loc[freq.split == "train", "group"]) & set(freq.loc[freq.split == "test", "group"]))
    print("Groups shared between train and test:", n_overlap)

    freq.to_parquet(PROCESSED / "policies.parquet", index=False)
    pd.Series({"large_claim_cap": cap}).to_json(PROCESSED / "constants.json")


if __name__ == "__main__":
    main()
