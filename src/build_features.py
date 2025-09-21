import argparse, time, numpy as np, pandas as pd, yaml
from .io_utils import _reader, name_force, name_energy, name_ux, append_csv
from .features_curves import compute_force_features, compute_energy_features
from .features_displacement import compute_ux_features
from .labeling import make_failure_labels


def _load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _process_split(split, cfg):
    io, params, paths, outputs = cfg["io"], cfg["params"], cfg["paths"], cfg["outputs"]
    rxnx_path = paths[f"rxnx_{split}"]; psi_path = paths[f"psi_{split}"]; ux_path = paths[f"dispx_s12_{split}"]
    out_csv = outputs[f"{split}_csv"]; write_header = True
    subset_enable = bool(cfg.get("subset", {}).get("enable", True))
    subset_limit = int(cfg.get("subset", {}).get(split, 0)) if subset_enable else None
    seen_rows = 0
    sep = io.get("sep", None); engine = io.get("engine", "python"); chunksize = int(io.get("chunksize", 10000))
    r_force = _reader(rxnx_path, chunksize, sep, engine)
    r_energy = _reader(psi_path, chunksize, sep, engine)
    r_ux = _reader(ux_path, chunksize, sep, engine)
    sample_id_offset = 0 if split == "train" else 60000
    load_mode = "uniaxial"
    t0 = time.time()
    while True:
        try:
            dfF_raw = next(r_force); dfE_raw = next(r_energy); dfU_raw = next(r_ux)
        except StopIteration:
            break
        n = min(len(dfF_raw), len(dfE_raw), len(dfU_raw))
        if n == 0: break
        dfF_raw = dfF_raw.iloc[:n].reset_index(drop=True)
        dfE_raw = dfE_raw.iloc[:n].reset_index(drop=True)
        dfU_raw = dfU_raw.iloc[:n].reset_index(drop=True)
        dfF = name_force(dfF_raw); dfE = name_energy(dfE_raw); dfU = name_ux(dfU_raw)
        if subset_enable:
            remain = subset_limit - seen_rows
            if remain <= 0: break
            if n > remain:
                dfF = dfF.iloc[:remain]; dfE = dfE.iloc[:remain]; dfU = dfU.iloc[:remain]; n = remain
        ffeat = compute_force_features(dfF)
        efeat = compute_energy_features(dfE)
        ufeat = compute_ux_features(dfU, use_abs=bool(params.get("compute_abs_displacement", True)))
        compact = pd.concat([ffeat, efeat, ufeat], axis=1)
        labels = make_failure_labels(compact, drop_thresh=float(params.get("drop_threshold", 0.20)))
        compact = pd.concat([compact, labels], axis=1)
        sample_ids = sample_id_offset + np.arange(seen_rows, seen_rows + n)
        compact.insert(0, "sample_id", sample_ids); compact.insert(1, "split", split); compact.insert(2, "load_mode", load_mode)
        append_csv(out_csv, compact, write_header=write_header); write_header = False
        seen_rows += n
        print(f"[{split}] wrote {n} rows (total {seen_rows}) to {out_csv}")
        if subset_enable and seen_rows >= subset_limit: break
    dt = time.time() - t0
    print(f"[{split}] DONE in {dt:.2f}s; total rows: {seen_rows}; output: {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="Build compact features and labels")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = _load_config(args.config)
    _process_split("train", cfg)
    _process_split("test", cfg)


if __name__ == "__main__":
    main()
