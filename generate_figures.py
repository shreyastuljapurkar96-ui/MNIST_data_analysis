import argparse
import datetime
import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration & Constants ---

OUTPUT_DIR = "./figures"
REQUIRED_COLUMNS = ["y_true", "y_pred_prob", "F_peak"]
RISK_BAND_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# --- Helper Functions ---

def add_risk_bands(df):
    """Adds a 'risk_band' column to the DataFrame based on y_pred_prob."""
    labels = [f"{int(low*100)}–{int(high*100)}%" for low, high in zip(RISK_BAND_EDGES[:-1], RISK_BAND_EDGES[1:])]
    df['risk_band'] = pd.cut(
        df['y_pred_prob'],
        bins=RISK_BAND_EDGES,
        labels=labels,
        right=True,
        include_lowest=True
    )
    return df

def safe_save(fig, path):
    """Saves a matplotlib figure and prints a confirmation message."""
    try:
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved: {os.path.abspath(path)}")
    except Exception as e:
        print(f"[ERROR] Failed to save {path}: {e}")
        plt.close(fig)

def get_metadata(fig_id, filename, status="success", notes=""):
    """Creates a standardized metadata dictionary for a figure."""
    return {
        "id": fig_id,
        "filename": filename,
        "status": status,
        "notes": notes,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

# --- Figure Generation Functions ---

def fig_4_8(df, path):
    """Generates Figure 4.8: Banded Performance–Risk scatter plot."""
    try:
        df_banded = add_risk_bands(df.copy())
        
        grouped = df_banded.groupby('risk_band', observed=True)
        
        summary = grouped.agg(
            x=('y_pred_prob', 'mean'),
            y=('F_peak', 'mean'),
            y_err=('F_peak', 'std'),
            n=('y_pred_prob', 'size')
        ).reset_index()

        if summary.empty:
            return get_metadata("4.8", os.path.basename(path), "skipped", "No data available for any risk band.")

        fig, ax = plt.subplots()
        
        sizes = summary['n'] / summary['n'].max() * 500  # Scale marker size
        
        ax.errorbar(summary['x'], summary['y'], yerr=summary['y_err'], fmt='none', ecolor='gray', capsize=5, zorder=1)
        ax.scatter(summary['x'], summary['y'], s=sizes, zorder=2)

        for i, row in summary.iterrows():
            ax.annotate(row['risk_band'], (row['x'], row['y']), textcoords="offset points", xytext=(0,10), ha='center')

        ax.set_title("Figure 4.8: Banded Performance–Risk")
        ax.set_xlabel("Mean predicted failure probability (per band)")
        ax.set_ylabel("Mean F_peak (per band)")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        safe_save(fig, path)
        return get_metadata("4.8", os.path.basename(path))

    except Exception as e:
        return get_metadata("4.8", os.path.basename(path), "error", str(e))

def fig_4_12(df, path):
    """Generates Figure 4.12: Localization Violin + Box plots."""
    optional_cols = [col for col in ['R_local_p95', 'R_local_max'] if col in df.columns]
    
    if not optional_cols:
        note = "Skipped: Optional columns 'R_local_p95' and 'R_local_max' not found."
        print(note)
        return get_metadata("4.12", os.path.basename(path), "skipped", note)

    try:
        # We will generate one plot containing all available metrics
        all_data = []
        labels = []
        for metric in optional_cols:
            non_fail = df[df['y_true'] == 0][metric].dropna()
            fail = df[df['y_true'] == 1][metric].dropna()
            if not non_fail.empty:
                all_data.append(non_fail)
                labels.append(f"{metric}\n(non-fail)")
            if not fail.empty:
                all_data.append(fail)
                labels.append(f"{metric}\n(fail)")
        
        if not all_data:
            return get_metadata("4.12", os.path.basename(path), "skipped", "No valid data for localization metrics.")

        fig, ax = plt.subplots(figsize=(len(all_data) * 2.5, 6))

        # Violin plot
        violin_parts = ax.violinplot(all_data, showmeans=False, showmedians=False, showextrema=False)
        for pc in violin_parts['bodies']:
            pc.set_facecolor('lightgray')
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        # Box plot overlay
        ax.boxplot(all_data, showfliers=True, patch_artist=True,
                    boxprops=dict(facecolor='none', edgecolor='black'),
                    medianprops=dict(color='black'),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'))

        ax.set_title("Figure 4.12: Localization Performance")
        ax.set_ylabel("Value")
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')
        
        safe_save(fig, path)
        return get_metadata("4.12", os.path.basename(path))

    except Exception as e:
        return get_metadata("4.12", os.path.basename(path), "error", str(e))


def fig_4_17(df, path):
    """Generates Figure 4.17: Histogram of Predicted Probabilities."""
    try:
        probs_non_fail = df[df['y_true'] == 0]['y_pred_prob']
        probs_fail = df[df['y_true'] == 1]['y_pred_prob']
        
        fig, ax = plt.subplots()
        
        ax.hist(probs_non_fail, bins=25, range=(0, 1), density=True, alpha=0.7, label="non-fail (y=0)")
        ax.hist(probs_fail, bins=25, range=(0, 1), density=True, alpha=0.7, label="fail (y=1)")
        
        ax.set_title("Figure 4.17: Histogram of Predicted Probabilities")
        ax.set_xlabel("Predicted failure probability")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        safe_save(fig, path)
        return get_metadata("4.17", os.path.basename(path))
        
    except Exception as e:
        return get_metadata("4.17", os.path.basename(path), "error", str(e))


def fig_4_20(df, path):
    """Generates Figure 4.20: Box/Whisker of F_peak by Risk Band."""
    try:
        df_banded = add_risk_bands(df.copy())
        
        grouped = df_banded.groupby('risk_band', observed=True)['F_peak']
        
        bands = [group.dropna() for name, group in grouped]
        band_labels = [name for name, group in grouped]

        if not bands:
            note = "Skipped: No data points fell into any risk bands."
            print(note)
            return get_metadata("4.20", os.path.basename(path), "skipped", note)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(bands, labels=band_labels, showfliers=False)
        
        ax.set_title("Figure 4.20: F_peak Distribution by Risk Band")
        ax.set_xlabel("Risk band (by predicted probability)")
        ax.set_ylabel("F_peak")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')
        
        safe_save(fig, path)
        return get_metadata("4.20", os.path.basename(path))

    except Exception as e:
        return get_metadata("4.20", os.path.basename(path), "error", str(e))

def fig_4_21(df, path):
    """Generates Figure 4.21: Hexbin of F_peak vs. predicted probability."""
    notes = ""
    if 'split' in df.columns:
        df_test = df[df['split'].str.lower() == 'test']
        notes = "Data filtered by 'split == test'."
    else:
        df_test = df.copy()
        notes = "Column 'split' not found; using all data."
        print(f"Note: {notes}")
    
    if df_test.empty:
        note = "Skipped: No data available for the test split (or at all)."
        print(note)
        return get_metadata("4.21", os.path.basename(path), "skipped", note)

    try:
        fig, ax = plt.subplots()
        
        hb = ax.hexbin(
            df_test['y_pred_prob'], df_test['F_peak'],
            gridsize=60, bins='log', mincnt=1,
        )
        
        fig.colorbar(hb, ax=ax, label="log10(count)")
        
        ax.set_title("Figure 4.21: F_peak vs. Failure Probability (Test Set)")
        ax.set_xlabel("Predicted failure probability")
        ax.set_ylabel("F_peak")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        safe_save(fig, path)
        return get_metadata("4.21", os.path.basename(path), "success", notes)

    except Exception as e:
        return get_metadata("4.21", os.path.basename(path), "error", str(e))

def artifact_map_figure(entries, path):
    """Renders a table image listing all generated artifacts."""
    try:
        col_labels = ["Figure", "Filename", "Status", "Notes", "Generated"]
        
        cell_text = []
        for entry in entries:
            ts = datetime.datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S UTC')
            cell_text.append([entry['id'], entry['filename'], entry['status'], entry['notes'], ts])
            
        fig, ax = plt.subplots(figsize=(14, len(entries) * 0.7 + 1))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        
        ax.set_title("Artifact Map — Generated Assets", pad=20)
        safe_save(fig, path)
        return get_metadata("4.22", os.path.basename(path), "success")
        
    except Exception as e:
        return get_metadata("4.22", os.path.basename(path), "error", str(e))

# --- Main Execution ---

def main():
    """Main function to parse arguments, load data, and generate figures."""
    parser = argparse.ArgumentParser(description="Generate figures from a summary results CSV.")
    parser.add_argument(
        "--csv",
        default="summary_results.csv",
        help="Path to the input CSV file (default: summary_results.csv)"
    )
    args = parser.parse_args()

    # --- Setup ---
    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f"Error: Input CSV not found at '{csv_path}'")
        sys.exit(1)
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Data Loading and Validation ---
    print(f"Loading data from: {os.path.abspath(csv_path)}")
    df = pd.read_csv(csv_path)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    # Coerce numeric types safely
    for col in ['y_true', 'y_pred_prob', 'F_peak']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN in key columns
    initial_rows = len(df)
    df.dropna(subset=['y_pred_prob', 'F_peak'], inplace=True)
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} rows with NaN in 'y_pred_prob' or 'F_peak'.")
    
    # Clip probabilities
    df['y_pred_prob'] = df['y_pred_prob'].clip(0, 1)

    print(f"Data loaded and validated. Found {len(df)} usable rows.")

    # --- Figure Generation ---
    figure_metadata = []
    
    figure_defs = [
        (fig_4_8, "Figure_4_8_banded_performance_risk.png"),
        (fig_4_12, "Figure_4_12_localization_violin_box.png"),
        (fig_4_17, "Figure_4_17_hist_pred_probs.png"),
        (fig_4_20, "Figure_4_20_box_Fpeak_by_risk_band.png"),
        (fig_4_21, "Figure_4_21_hexbin_Fpeak_vs_pfail_test.png"),
    ]
    
    for fig_func, filename in figure_defs:
        output_path = os.path.join(OUTPUT_DIR, filename)
        metadata = fig_func(df, output_path)
        figure_metadata.append(metadata)

    # --- Artifact Map and Manifest ---
    # Generate the artifact map image itself
    map_path = os.path.join(OUTPUT_DIR, "Figure_4_22_artifact_map.png")
    map_metadata = artifact_map_figure(figure_metadata, map_path)
    figure_metadata.append(map_metadata) # Add map's own metadata to the list

    # Write the JSON manifest
    manifest = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_csv": os.path.abspath(csv_path),
        "risk_band_edges": RISK_BAND_EDGES,
        "figures": figure_metadata,
    }

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Saved: {os.path.abspath(manifest_path)}")

if __name__ == "__main__":
    main()
