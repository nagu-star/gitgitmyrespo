import pandas as pd
import numpy as np

def engineer_mplads_features(df):
    """
    Computes key financial ratios, benchmark cost deviations, and progress features.
    """
    if df is None or df.empty:
        return df

    fe_df = df.copy()

    # 1. Financial Utilization & Overrun metrics
    sanc = fe_df['SANCTION_AMOUNT'].replace(0, np.nan)
    exp = fe_df['EXPENDITURE_AMOUNT']

    fe_df['UTILIZATION_PCT'] = np.where(sanc > 0, (exp / sanc) * 100.0, 0.0)
    fe_df['UTILIZATION_PCT'] = fe_df['UTILIZATION_PCT'].round(2)

    fe_df['REMAINING_AMOUNT'] = np.maximum(0.0, fe_df['SANCTION_AMOUNT'] - exp).round(2)
    fe_df['COST_DIFFERENCE'] = (exp - fe_df['SANCTION_AMOUNT']).round(2)
    fe_df['COST_RATIO'] = np.where(sanc > 0, exp / sanc, 1.0).round(3)

    # 2. Category & State Benchmark Comparisons
    cat_means = fe_df.groupby('WORK_CATEGORY')['SANCTION_AMOUNT'].transform('mean')
    fe_df['CATEGORY_AVG_COST'] = cat_means.round(2)
    fe_df['COST_DEVIATION_FROM_CAT'] = np.where(cat_means > 0, (fe_df['SANCTION_AMOUNT'] - cat_means) / cat_means, 0.0).round(3)

    # 3. Progress vs Expenditure Discrepancy
    # High expenditure but low progress flag
    exp_progress_gap = fe_df['UTILIZATION_PCT'] - fe_df['PROGRESS_PERCENTAGE']
    fe_df['EXP_PROGRESS_DISCREPANCY'] = exp_progress_gap.round(2)

    # 4. Prolonged / Delayed flags based on dates or status
    if 'RECOMMENDATION_DATE' in fe_df.columns and 'SANCTION_DATE' in fe_df.columns:
        fe_df['SANCTION_DELAY_DAYS'] = (fe_df['SANCTION_DATE'] - fe_df['RECOMMENDATION_DATE']).dt.days.fillna(0).astype(int)
    else:
        fe_df['SANCTION_DELAY_DAYS'] = 0

    return fe_df
