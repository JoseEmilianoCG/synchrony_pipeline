import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def generate_report(feat_mat):
    # --- Helper: Cliff's Delta ---
    def cliffs_delta(x, y):
        nx, ny = len(x), len(y)
        gt = sum(i > j for i in x for j in y)
        lt = sum(i < j for i in x for j in y)
        return (gt - lt) / (nx * ny)

    feature_cols = [c for c in feat_mat.columns if c not in ["Session", "Target"]]

    results = []

    # =========================
    # PER SESSION ANALYSIS
    # =========================
    for session in feat_mat["Session"].unique():
        df_sess = feat_mat[feat_mat["Session"] == session]

        df_shared = df_sess[df_sess["Target"] == 0]
        df_ind = df_sess[df_sess["Target"] == 1]

        for feat in feature_cols:
            x = df_shared[feat].values
            y = df_ind[feat].values

            # Ensure equal length for paired test
            n = min(len(x), len(y))
            x = x[:n]
            y = y[:n]

            # --- Wilcoxon ---
            stat, pval = wilcoxon(x, y)

            # --- Effect size ---
            delta = cliffs_delta(x, y)

            # --- Trend ---
            mean_shared = np.mean(x)
            mean_ind = np.mean(y)

            trend = "increase" if mean_ind > mean_shared else "decrease"

            results.append(
                {
                    "Session": session,
                    "Feature": feat,
                    "Mean_Shared": mean_shared,
                    "Mean_Individual": mean_ind,
                    "Trend": trend,
                    "p_value": pval,
                    "Cliffs_delta": delta,
                }
            )

    report = pd.DataFrame(results)

    # =========================
    # FDR CORRECTION
    # =========================
    reject, pvals_corr, _, _ = multipletests(report["p_value"], method="fdr_bh")

    report["p_value_fdr"] = pvals_corr
    report["Significant"] = reject

    return report
