import pandas as pd
import numpy as np

def compute_hhi_concentration(vendor_series):
    """
    Compute Herfindahl-Hirschman Index (HHI) for vendor market concentration.
    HHI ranges from 0 to 1.0 (or 0 to 10,000). Values > 0.25 indicate high concentration/cartel risk.
    """
    counts = vendor_series.value_counts()
    total = len(vendor_series)
    if total == 0:
        return 0.0
    shares = counts / total
    hhi = (shares ** 2).sum()
    return round(float(hhi), 3)

def detect_vendor_cartels_and_splits(works_df, tender_threshold_lakhs=25.0):
    """
    Detect vendor cartels using HHI and flag split-tendering:
    works budgeted between ₹15L and ₹25L awarded to the same vendor in the same district/time window
    to bypass public tender threshold rules.
    """
    df = works_df.copy()
    
    # District-level HHI score calculation
    district_hhi_map = {}
    for district, group in df.groupby('DISTRICT_NAME'):
        if 'VENDOR_NAME' in group.columns:
            district_hhi_map[district] = compute_hhi_concentration(group['VENDOR_NAME'])
        else:
            district_hhi_map[district] = 0.0
            
    df['DISTRICT_VENDOR_HHI'] = df['DISTRICT_NAME'].map(district_hhi_map)
    df['IS_HIGH_CARTEL_RISK'] = df['DISTRICT_VENDOR_HHI'] > 0.25
    
    # Split-tendering detection
    df['IS_SPLIT_TENDER'] = False
    df['SPLIT_TENDER_CLUSTER_COUNT'] = 0
    df['SPLIT_TENDER_REASON'] = None
    df['CARTEL_RISK_SCORE'] = 0.0

    threshold_amt = tender_threshold_lakhs * 100000  # ₹25 Lakhs in Rupees
    lower_bound = 0.60 * threshold_amt               # ₹15 Lakhs

    if 'SANCTION_AMOUNT' in df.columns and 'VENDOR_NAME' in df.columns:
        # Group by District and Vendor
        for (district, vendor), group in df.groupby(['DISTRICT_NAME', 'VENDOR_NAME']):
            # Filter works in the split-tender target financial range (e.g., ₹15L to ₹24.9L)
            sub_threshold_works = group[
                (group['SANCTION_AMOUNT'] >= lower_bound) & 
                (group['SANCTION_AMOUNT'] < threshold_amt)
            ]
            
            # If a single vendor gets 2 or more sub-threshold works in the same district
            if len(sub_threshold_works) >= 2:
                for idx in sub_threshold_works.index:
                    df.at[idx, 'IS_SPLIT_TENDER'] = True
                    df.at[idx, 'SPLIT_TENDER_CLUSTER_COUNT'] = len(sub_threshold_works)
                    df.at[idx, 'SPLIT_TENDER_REASON'] = (
                        f"Vendor '{vendor}' awarded {len(sub_threshold_works)} packages "
                        f"in {district} just below statutory ₹{tender_threshold_lakhs:.0f}L threshold."
                    )

    # Compute Cartel Risk Score (0-100)
    for idx, row in df.iterrows():
        score = 0.0
        if row.get('IS_SPLIT_TENDER', False):
            score += 65.0 + min(25.0, row.get('SPLIT_TENDER_CLUSTER_COUNT', 0) * 5)
        if row.get('IS_HIGH_CARTEL_RISK', False):
            score += row.get('DISTRICT_VENDOR_HHI', 0.0) * 30.0
        df.at[idx, 'CARTEL_RISK_SCORE'] = round(min(100.0, score), 1)

    return df
