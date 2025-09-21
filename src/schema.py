from typing import List

def force_cols() -> List[str]:
    return [f"F_step{idx:02d}" for idx in range(13)]  # 00..12

def energy_cols() -> List[str]:
    return [f"E_step{idx:02d}" for idx in range(13)]  # 00..12

def ux_cols() -> List[str]:
    # 784 cells for the 28x28 field at final step (step12)
    return [f"ux_s12_cell{idx:03d}" for idx in range(28*28)]

META_COLS = ["sample_id", "split", "load_mode"]
