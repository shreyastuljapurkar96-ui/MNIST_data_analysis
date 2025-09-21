import numpy as np, pandas as pd
from .schema import ux_cols


def compute_ux_features(df_ux: pd.DataFrame, use_abs: bool=True) -> pd.DataFrame:
    Ucols = ux_cols()
    missing = [c for c in Ucols if c not in df_ux.columns]
    if missing: raise ValueError(f"Missing ux columns: {missing}")
    U = df_ux[Ucols].to_numpy(dtype=float)
    if use_abs: U = np.abs(U)
    med = np.median(U, axis=1)
    p95 = np.percentile(U, 95, axis=1, method="linear")
    umax = U.max(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        r_p95 = p95 / np.where(med==0, np.nan, med)
        r_max = umax / np.where(med==0, np.nan, med)
    r_p95 = np.nan_to_num(r_p95, nan=1.0, posinf=1e6, neginf=1.0)
    r_max = np.nan_to_num(r_max, nan=1.0, posinf=1e6, neginf=1.0)
    return pd.DataFrame({
        "ux_s12_med": med,
        "ux_s12_p95": p95,
        "ux_s12_max": umax,
        "R_local_p95": r_p95,
        "R_local_max": r_max,
    }, index=df_ux.index)
