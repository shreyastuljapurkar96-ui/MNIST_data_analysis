import numpy as np, pandas as pd


def make_failure_labels(df: pd.DataFrame, drop_thresh: float=0.20) -> pd.DataFrame:
    A = (df["force_peak_step"].to_numpy() < 12) & (df["F_drop_ratio"].to_numpy() >= drop_thresh)
    B = (df["E_softening_ratio"].to_numpy() >= 0.20)
    C = (df["end_slope"].to_numpy() <= 0.0)
    score = A.astype(int) + B.astype(int) + C.astype(int)
    flag = (score >= 2).astype(int)
    return pd.DataFrame({"failure_score": score, "failure_flag": flag}, index=df.index)
