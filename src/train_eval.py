import json, os
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, roc_curve,
                             confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score)
import matplotlib.pyplot as plt

REPORTS_DIR = "reports"

FEATURES = [
    "F_peak","force_peak_step","F_drop_ratio","end_slope",
    "E_peak","E_final_ratio","E_softening_ratio",
    "ux_s12_med","ux_s12_p95","ux_s12_max","R_local_p95","R_local_max"
]


def load_data(train_csv, test_csv):
    df_tr = pd.read_csv(train_csv)
    df_te = pd.read_csv(test_csv)
    X_tr = df_tr[FEATURES].values
    y_tr = df_tr["failure_flag"].values.astype(int)
    X_te = df_te[FEATURES].values
    y_te = df_te["failure_flag"].values.astype(int)
    return df_tr, df_te, X_tr, y_tr, X_te, y_te


def split_train_valid(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def train_models(X_tr, y_tr):
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    gboost = GradientBoostingClassifier()
    logreg.fit(X_tr, y_tr)
    gboost.fit(X_tr, y_tr)
    return logreg, gboost


def calibrate_model(model, X_val, y_val, method="isotonic"):
    calib = CalibratedClassifierCV(model, cv="prefit", method=method)
    calib.fit(X_val, y_val)
    return calib


def pick_threshold(y_true, probs, strategy="f1"):
    # You can choose other strategies; here we maximize F1 on validation
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.05, 0.95, 181):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def eval_and_plot(y_true, probs, threshold, prefix):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    preds = (probs >= threshold).astype(int)

    # Metrics
    roc_auc = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    cm = confusion_matrix(y_true, preds).tolist()

    metrics = {
        "threshold": threshold, "roc_auc": roc_auc, "pr_auc": pr_auc,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "confusion_matrix": cm
    }
    with open(os.path.join(REPORTS_DIR, f"{prefix}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC — {prefix}")
    plt.savefig(os.path.join(REPORTS_DIR, f"{prefix}_roc.png"), dpi=160)
    plt.close()

    # PR curve
    precs, recs, _ = precision_recall_curve(y_true, probs)
    plt.figure()
    plt.plot(recs, precs)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR — {prefix}")
    plt.savefig(os.path.join(REPORTS_DIR, f"{prefix}_pr.png"), dpi=160)
    plt.close()


def pareto_plot(df_test, probs, prefix="pareto"):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    # X=risk, Y=performance, color=localization
    x = probs
    y = df_test["F_peak"].values
    c = df_test["R_local_p95"].values
    plt.figure()
    sc = plt.scatter(x, y, c=c)  # default colormap
    plt.xlabel("Predicted failure probability (risk)")
    plt.ylabel("F_peak (performance)")
    plt.title("Pareto: performance vs risk")
    cb = plt.colorbar(sc)
    cb.set_label("R_local_p95 (localization)")
    plt.savefig(os.path.join(REPORTS_DIR, f"{prefix}.png"), dpi=160)
    plt.close()


def main(train_csv="data/processed/features_train_compact.csv",
         test_csv="data/processed/features_test_compact.csv",
         seed=42):

    df_tr, df_te, X_tr, y_tr, X_te, y_te = load_data(train_csv, test_csv)
    X_tr_sub, X_val, y_tr_sub, y_val = split_train_valid(X_tr, y_tr, test_size=0.2, seed=seed)

    # Train two baselines
    logreg, gboost = train_models(X_tr_sub, y_tr_sub)

    # Calibrate on validation
    calib_log = calibrate_model(logreg, X_val, y_val, method="isotonic")
    calib_gb  = calibrate_model(gboost, X_val, y_val, method="isotonic")

    # Pick thresholds on validation
    val_probs_log = calib_log.predict_proba(X_val)[:,1]
    val_probs_gb  = calib_gb.predict_proba(X_val)[:,1]
    thr_log = pick_threshold(y_val, val_probs_log, strategy="f1")
    thr_gb  = pick_threshold(y_val, val_probs_gb, strategy="f1")

    # Evaluate on test
    test_probs_log = calib_log.predict_proba(X_te)[:,1]
    test_probs_gb  = calib_gb.predict_proba(X_te)[:,1]
    eval_and_plot(y_te, test_probs_log, thr_log, prefix="logreg_test")
    eval_and_plot(y_te, test_probs_gb,  thr_gb,  prefix="gboost_test")

    # Pareto (use the better of the two; here use gboost)
    pareto_plot(df_te, test_probs_gb, prefix="pareto_test")


if __name__ == "__main__":
    main()
