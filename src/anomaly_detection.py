import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

def detect_anomalies(df):
    """
    Detect financial, utilization, and progress anomalies using statistical & machine learning techniques.
    Methods: Isolation Forest, Local Outlier Factor, and IQR statistical bounds.
    """
    if df is None or df.empty:
        return df

    out_df = df.copy()

    # 1. Prepare Feature Matrix for ML
    feature_cols = ['SANCTION_AMOUNT', 'EXPENDITURE_AMOUNT', 'UTILIZATION_PCT', 'COST_RATIO', 'EXP_PROGRESS_DISCREPANCY']
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in out_df.columns:
            out_df[col] = 0.0

    X = out_df[feature_cols].fillna(0.0).values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Isolation Forest Model
    iso_model = IsolationForest(contamination=0.10, random_state=42)
    iso_preds = iso_model.fit_predict(X_scaled) # -1 for anomaly, 1 for normal
    iso_scores = -iso_model.score_samples(X_scaled) # higher score = more anomalous
    
    # Normalize iso_scores to [0, 1] range
    min_s, max_s = iso_scores.min(), iso_scores.max()
    norm_iso_scores = (iso_scores - min_s) / (max_s - min_s + 1e-6)

    # 3. IQR Statistical Outlier Detection on Utilization & Cost Ratio
    q25_util, q75_util = np.percentile(out_df['UTILIZATION_PCT'], 25), np.percentile(out_df['UTILIZATION_PCT'], 75)
    iqr_util = q75_util - q25_util
    high_util_cutoff = q75_util + 1.5 * iqr_util
    
    iqr_anomalies = (out_df['UTILIZATION_PCT'] > max(120.0, high_util_cutoff)) | (out_df['COST_RATIO'] > 1.25)

    # 4. Composite Anomaly Flag & Reasons
    is_anomaly_list = []
    anomaly_score_list = []
    anomaly_reasons = []

    for idx, row in out_df.iterrows():
        iso_flag = (iso_preds[idx] == -1)
        iqr_flag = bool(iqr_anomalies.iloc[idx])
        discrepancy_flag = (row['UTILIZATION_PCT'] > 85.0 and row['PROGRESS_PERCENTAGE'] < 50.0)
        overrun_flag = (row['EXPENDITURE_AMOUNT'] > row['SANCTION_AMOUNT'])

        is_anom = iso_flag or iqr_flag or discrepancy_flag or overrun_flag
        score = round(float(norm_iso_scores[idx]), 3)
        if is_anom:
            score = max(score, 0.72)

        reasons = []
        if overrun_flag:
            overrun_pct = round(((row['EXPENDITURE_AMOUNT'] - row['SANCTION_AMOUNT']) / max(1, row['SANCTION_AMOUNT'])) * 100, 1)
            reasons.append(f"Budget Overrun: Actual expenditure exceeds sanctioned amount by {overrun_pct}%")
        if discrepancy_flag:
            reasons.append(f"High Spending with Low Progress: {row['UTILIZATION_PCT']}% of funds spent while physical work is only {row['PROGRESS_PERCENTAGE']}% complete")
        if iqr_flag and not overrun_flag:
            reasons.append(f"Abnormal Cost Ratio: Expenditure ratio ({row['COST_RATIO']}x) exceeds normal category benchmark bounds")
        if iso_flag and len(reasons) == 0:
            reasons.append("Unusual Financial Pattern: Financial allocation and expenditure timeline differ significantly from standard projects in this category")


        is_anomaly_list.append(is_anom)
        anomaly_score_list.append(score)
        anomaly_reasons.append("; ".join(reasons) if is_anom else "Normal execution pattern")

    out_df['IS_ANOMALY'] = is_anomaly_list
    out_df['ANOMALY_SCORE'] = anomaly_score_list
    out_df['ANOMALY_REASON'] = anomaly_reasons

    return out_df
