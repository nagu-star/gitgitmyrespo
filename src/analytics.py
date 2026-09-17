import pandas as pd
import numpy as np

def filter_dataset(df, state=None, district=None, mp=None, tenure=None, category=None, status=None, risk_level=None):
    """
    Apply interactive filters to the dataset.
    """
    if df is None or df.empty:
        return df

    filtered = df.copy()

    if state and state != 'All States':
        filtered = filtered[filtered['STATE_NAME'] == state]
    if district and district != 'All Districts':
        filtered = filtered[filtered['DISTRICT_NAME'] == district]
    if mp and mp != 'All MPs':
        filtered = filtered[filtered['MP_NAME'] == mp]
    if tenure and tenure != 'All Tenures':
        filtered = filtered[filtered['TENURE'] == tenure]
    if category and category != 'All Categories':
        filtered = filtered[filtered['WORK_CATEGORY'] == category]
    if status and status != 'All Statuses':
        filtered = filtered[filtered['WORK_STATUS'] == status]
    if risk_level and risk_level != 'All Risk Levels':
        if 'RISK_LEVEL' in filtered.columns:
            filtered = filtered[filtered['RISK_LEVEL'] == risk_level]

    return filtered

def compute_kpis(df):
    """
    Compute summary KPIs from dataset.
    """
    if df is None or df.empty:
        return {
            'total_works': 0,
            'total_sanctioned': 0.0,
            'total_expenditure': 0.0,
            'remaining_amount': 0.0,
            'utilization_rate': 0.0,
            'completed_works': 0,
            'ongoing_works': 0,
            'pending_works': 0,
            'delayed_works': 0,
            'high_risk_works': 0,
            'anomalous_works': 0,
            'duplicate_works': 0
        }

    total_works = len(df)
    sanc_sum = float(df['SANCTION_AMOUNT'].sum())
    exp_sum = float(df['EXPENDITURE_AMOUNT'].sum())
    rem_sum = max(0.0, sanc_sum - exp_sum)
    util_rate = round((exp_sum / sanc_sum * 100.0), 2) if sanc_sum > 0 else 0.0

    comp_cnt = int((df['WORK_STATUS'] == 'Completed').sum())
    ong_cnt = int((df['WORK_STATUS'] == 'Ongoing').sum())
    pend_cnt = int((df['WORK_STATUS'].isin(['Sanctioned / Pending', 'Pending'])).sum())
    del_cnt = int((df['WORK_STATUS'] == 'Delayed').sum())

    high_risk_cnt = int((df['RISK_LEVEL'].isin(['HIGH', 'CRITICAL'])).sum()) if 'RISK_LEVEL' in df.columns else 0
    anom_cnt = int((df['IS_ANOMALY'] == True).sum()) if 'IS_ANOMALY' in df.columns else 0
    dup_cnt = int((df['IS_DUPLICATE_FLAG'] == True).sum()) if 'IS_DUPLICATE_FLAG' in df.columns else 0

    return {
        'total_works': total_works,
        'total_sanctioned': sanc_sum,
        'total_expenditure': exp_sum,
        'remaining_amount': rem_sum,
        'utilization_rate': util_rate,
        'completed_works': comp_cnt,
        'ongoing_works': ong_cnt,
        'pending_works': pend_cnt,
        'delayed_works': del_cnt,
        'high_risk_works': high_risk_cnt,
        'anomalous_works': anom_cnt,
        'duplicate_works': dup_cnt
    }

def aggregate_by_state(df):
    """Aggregate metrics state-wise."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    grouped = df.groupby('STATE_NAME').agg(
        Total_Works=('WORK_ID', 'count'),
        Total_Sanctioned=('SANCTION_AMOUNT', 'sum'),
        Total_Expenditure=('EXPENDITURE_AMOUNT', 'sum'),
        Completed_Works=('WORK_STATUS', lambda x: (x == 'Completed').sum()),
        Ongoing_Works=('WORK_STATUS', lambda x: (x == 'Ongoing').sum()),
        Delayed_Works=('WORK_STATUS', lambda x: (x == 'Delayed').sum()),
        High_Risk_Works=('RISK_LEVEL', lambda x: (x.isin(['HIGH', 'CRITICAL'])).sum() if 'RISK_LEVEL' in df.columns else 0)
    ).reset_index()

    grouped['Utilization_Pct'] = np.where(grouped['Total_Sanctioned'] > 0, (grouped['Total_Expenditure'] / grouped['Total_Sanctioned']) * 100.0, 0.0).round(2)
    return grouped.sort_values(by='Total_Sanctioned', ascending=False)

def aggregate_by_category(df):
    """Aggregate metrics work-category wise."""
    if df is None or df.empty:
        return pd.DataFrame()

    grouped = df.groupby('WORK_CATEGORY').agg(
        Total_Works=('WORK_ID', 'count'),
        Total_Sanctioned=('SANCTION_AMOUNT', 'sum'),
        Total_Expenditure=('EXPENDITURE_AMOUNT', 'sum'),
        Avg_Sanction_Cost=('SANCTION_AMOUNT', 'mean'),
        Completed_Works=('WORK_STATUS', lambda x: (x == 'Completed').sum()),
        High_Risk_Works=('RISK_LEVEL', lambda x: (x.isin(['HIGH', 'CRITICAL'])).sum() if 'RISK_LEVEL' in df.columns else 0)
    ).reset_index()

    grouped['Utilization_Pct'] = np.where(grouped['Total_Sanctioned'] > 0, (grouped['Total_Expenditure'] / grouped['Total_Sanctioned']) * 100.0, 0.0).round(2)
    return grouped.sort_values(by='Total_Sanctioned', ascending=False)

def aggregate_by_mp(df):
    """Aggregate metrics MP-wise without performance ranking."""
    if df is None or df.empty:
        return pd.DataFrame()

    grouped = df.groupby('MP_NAME').agg(
        State=('STATE_NAME', 'first'),
        Total_Works=('WORK_ID', 'count'),
        Total_Sanctioned=('SANCTION_AMOUNT', 'sum'),
        Total_Expenditure=('EXPENDITURE_AMOUNT', 'sum'),
        Completed_Works=('WORK_STATUS', lambda x: (x == 'Completed').sum()),
        Ongoing_Works=('WORK_STATUS', lambda x: (x == 'Ongoing').sum()),
        High_Risk_Works=('RISK_LEVEL', lambda x: (x.isin(['HIGH', 'CRITICAL'])).sum() if 'RISK_LEVEL' in df.columns else 0)
    ).reset_index()

    grouped['Utilization_Pct'] = np.where(grouped['Total_Sanctioned'] > 0, (grouped['Total_Expenditure'] / grouped['Total_Sanctioned']) * 100.0, 0.0).round(2)
    return grouped
