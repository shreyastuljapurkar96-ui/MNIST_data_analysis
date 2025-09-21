import numpy as np, pandas as pd
from .schema import force_cols, energy_cols


def compute_force_features(df_force: pd.DataFrame) -> pd.DataFrame:
    Fcols = force_cols()
    missing = [c for c in Fcols if c not in df_force.columns]
    if missing: raise ValueError(f"Missing force columns: {missing}")
    F = df_force[Fcols].to_numpy(dtype=float)
    F_peak = F.max(axis=1)
    k_peak = F.argmax(axis=1)
    k_next = np.minimum(k_peak + 1, F.shape[1]-1)
    F_next = F[np.arange(F.shape[0]), k_next]
    with np.errstate(divide='ignore', invalid='ignore'):
        F_drop_ratio = (F_peak - F_next) / np.where(F_peak==0, np.nan, F_peak)
    F_drop_ratio = np.nan_to_num(F_drop_ratio, nan=0.0)
    F_drop_ratio = np.maximum(F_drop_ratio, 0.0)
    end_slope = F[:,-1] - F[:,-2]
    return pd.DataFrame({
        "F_peak": F_peak,
        "force_peak_step": k_peak,
        "F_drop_ratio": F_drop_ratio,
        "end_slope": end_slope,
    }, index=df_force.index)


def compute_energy_features(df_energy: pd.DataFrame) -> pd.DataFrame:
    Ecols = energy_cols()
    missing = [c for c in Ecols if c not in df_energy.columns]
    if missing: raise ValueError(f"Missing energy columns: {missing}")
    E = df_energy[Ecols].to_numpy(dtype=float)
    E_peak = E.max(axis=1)
    last = E[:,-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        E_final_ratio = last / np.where(E_peak==0, np.nan, E_peak)
    E_final_ratio = np.clip(np.nan_to_num(E_final_ratio, nan=0.0), 0.0, 1.0)
    E_softening_ratio = 1.0 - E_final_ratio
    return pd.DataFrame({
        "E_peak": E_peak,
        "E_final_ratio": E_final_ratio,
        "E_softening_ratio": E_softening_ratio,
    }, index=df_energy.index)
