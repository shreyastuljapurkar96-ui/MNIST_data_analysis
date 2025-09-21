import pandas as pd, numpy as np, os


def _reader(path: str, chunksize: int, sep, engine):
    # Robust text reader: auto-detect separator if sep is None (python engine).
    if sep is None:
        return pd.read_csv(path, header=None, engine=engine, chunksize=chunksize)
    else:
        return pd.read_csv(path, header=None, sep=sep, engine=engine, chunksize=chunksize)


def name_force(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [f"F_step{idx:02d}" for idx in range(df.shape[1])]
    return df


def name_energy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [f"E_step{idx:02d}" for idx in range(df.shape[1])]
    return df


def name_ux(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [f"ux_s12_cell{idx:03d}" for idx in range(df.shape[1])]
    return df


def append_csv(path: str, df: pd.DataFrame, write_header: bool):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, mode=("w" if write_header else "a"), header=write_header)
