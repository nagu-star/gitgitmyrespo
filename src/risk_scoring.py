import pandas as pd
import numpy as np

def compute_risk_scoring(df):
    """
    Calculates a transparent composite Risk Score (0-100) and assigns Risk Levels
    (LOW, MEDIUM, HIGH, CRITICAL) with clear risk factors.
    """
    if df is None or df.empty:
        return df

    out_df = df.copy()

    risk_scores = []
    risk_levels = []
    risk_factors_list = []

    for idx, row in out_df.iterrows():
        base_score = 15.0
        factors = []

        # 1. Cost Overrun Risk (+35 pts)
        cost_ratio = row.get('COST_RATIO', 1.0)
        exp_amt = row.get('EXPENDITURE_AMOUNT', 0.0)
        sanc_amt = row.get('SANCTION_AMOUNT', 0.0)
        if exp_amt > sanc_amt:
            overrun_pct = ((exp_amt - sanc_amt) / max(1.0, sanc_amt)) * 100.0
            pts = min(35.0, 15.0 + (overrun_pct * 0.4))
            base_score += pts
            factors.append(f"Cost Overrun (+{round(pts,1)} pts): Expenditure exceeds sanction by {round(overrun_pct,1)}%")

        # 2. Utilization & Progress Discrepancy (+30 pts)
        util_pct = row.get('UTILIZATION_PCT', 0.0)
        progress_pct = row.get('PROGRESS_PERCENTAGE', 0.0)
        discrepancy = util_pct - progress_pct

        if util_pct > 80.0 and progress_pct < 50.0:
            base_score += 25.0
            factors.append(f"Incomplete High Expenditure (+25 pts): {round(util_pct,1)}% funds spent with only {round(progress_pct,1)}% work completion")
        elif util_pct < 20.0 and row.get('WORK_STATUS') == 'Delayed':
            base_score += 20.0
            factors.append(f"Low Utilization & Delay (+20 pts): Prolonged project with only {round(util_pct,1)}% fund utilization")

        # 3. Anomaly Score Impact (+20 pts)
        anom_score = row.get('ANOMALY_SCORE', 0.0)
        if row.get('IS_ANOMALY', False):
            anom_pts = min(20.0, anom_score * 20.0)
            base_score += anom_pts
            factors.append(f"Statistical Anomaly (+{round(anom_pts,1)} pts): High multivariate anomaly index ({anom_score})")

        # 4. Duplicate Similarity Risk (+15 pts)
        if row.get('IS_DUPLICATE_FLAG', False):
            base_score += 15.0
            factors.append("Duplicate Risk (+15 pts): Identified as potential duplicate of another work in the same region")

        # 5. Cross-Scheme Double Funding (+30 pts)
        if row.get('IS_CROSS_SCHEME_DUPLICATE', False):
            cs_name = row.get('CROSS_SCHEME_MATCH_NAME', 'Parallel Scheme')
            dist_m = row.get('CROSS_SCHEME_DISTANCE_METERS', 0.0)
            base_score += 30.0
            factors.append(f"Cross-Scheme Double Funding (+30 pts): Asset overlap with {cs_name} ({dist_m}m away)")

        # 6. Cartel & Split-Tendering (+25 pts)
        if row.get('IS_SPLIT_TENDER', False):
            base_score += 25.0
            factors.append(f"Split-Tendering (+25 pts): Sub-₹25L package cluster awarded under high vendor concentration")

        # 7. S-Curve Progress Stagnation (+20 pts)
        if row.get('IS_STAGNANT_SCURVE', False):
            base_score += 20.0
            factors.append("S-Curve Stagnation (+20 pts): Critical financial expenditure lead velocity relative to physical work completion")

        # Cap score between 0 and 100
        final_score = round(max(5.0, min(100.0, base_score)), 1)

        # Risk Level Classification
        if final_score >= 75.0:
            level = 'CRITICAL'
        elif final_score >= 55.0:
            level = 'HIGH'
        elif final_score >= 35.0:
            level = 'MEDIUM'
        else:
            level = 'LOW'

        risk_scores.append(final_score)
        risk_levels.append(level)
        risk_factors_list.append("; ".join(factors) if factors else "Standard implementation profile")

    out_df['RISK_SCORE'] = risk_scores
    out_df['RISK_LEVEL'] = risk_levels
    out_df['RISK_FACTORS'] = risk_factors_list

    return out_df
