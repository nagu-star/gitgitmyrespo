import pandas as pd

def generate_risk_alerts(df, min_risk_level='MEDIUM'):
    """
    Dynamically generates structured risk alerts for decision support.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    alert_levels = ['MEDIUM', 'HIGH', 'CRITICAL'] if min_risk_level == 'MEDIUM' else ['HIGH', 'CRITICAL']
    high_risk_df = df[df['RISK_LEVEL'].isin(alert_levels)].copy()

    alerts_list = []
    for idx, row in high_risk_df.iterrows():
        issue = "Potential Irregularity / Requires Investigation"
        if row.get('IS_CROSS_SCHEME_DUPLICATE', False):
            issue = "Cross-Scheme Double Funding Overlap"
        elif row.get('IS_SPLIT_TENDER', False):
            issue = "Split-Tender Evasion & Vendor Cartel Flag"
        elif row.get('IS_GEO_MISMATCH', False):
            issue = "Photo Geofence Mismatch (>100m Variance)"
        elif row.get('IS_DUPLICATE_PHOTO', False):
            issue = "Reused Progress Photo Fraud"
        elif row.get('IS_STAGNANT_SCURVE', False):
            issue = "S-Curve Progress Stagnation / Financial Lead"
        elif row.get('EXPENDITURE_AMOUNT', 0) > row.get('SANCTION_AMOUNT', 0):
            issue = "Cost Overrun & Financial Discrepancy"
        elif row.get('IS_DUPLICATE_FLAG', False):
            issue = "Potential Duplicate Work Detected"
        elif row.get('WORK_STATUS') == 'Delayed':
            issue = "Prolonged Execution Delay"

        # Suggested Review
        if issue == "Cross-Scheme Double Funding Overlap":
            review = f"Cross-verify work location with {row.get('CROSS_SCHEME_MATCH_NAME', 'parallel scheme')} database before releasing funds."
        elif issue == "Split-Tender Evasion & Vendor Cartel Flag":
            review = "Freeze direct contract awards; consolidate packages into open competitive e-tenders."
        elif issue == "Photo Geofence Mismatch (>100m Variance)":
            review = "Require mandatory re-upload of geotagged image at exact sanctioned coordinates via mobile portal."
        elif issue == "S-Curve Progress Stagnation / Financial Lead":
            review = "Hold financial disbursement; demand physical stage-completion milestone audit."
        elif issue == "Cost Overrun & Financial Discrepancy":
            review = "Audit expenditure Vouchers and verify revised sanction approvals."
        elif issue == "Potential Duplicate Work Detected":
            review = "Inspect site location and cross-verify with previously sanctioned works."
        elif issue == "Prolonged Execution Delay":
            review = "Issue reminder to Implementing District Authority (IDA) and request progress report."
        else:
            review = "Conduct physical site verification and review financial audit trail."

        evidence = f"Sanction: ₹{row['SANCTION_AMOUNT']:,.2f} | Exp: ₹{row['EXPENDITURE_AMOUNT']:,.2f} | Utilization: {row.get('UTILIZATION_PCT',0)}% | Progress: {row.get('PROGRESS_PERCENTAGE',0)}%"

        alerts_list.append({
            'Work ID': row['WORK_ID'],
            'Location': f"{row['STATE_NAME']}, {row['DISTRICT_NAME']}",
            'MP Name': row['MP_NAME'],
            'Work Category': row['WORK_CATEGORY'],
            'Issue': issue,
            'Risk Level': row['RISK_LEVEL'],
            'Risk Score': row['RISK_SCORE'],
            'Reason for Flagging': row.get('RISK_FACTORS', row.get('ANOMALY_REASON', 'Requires verification')),
            'Supporting Indicator': evidence,
            'Suggested Review': review
        })

    alerts_df = pd.DataFrame(alerts_list)
    return alerts_df.sort_values(by='Risk Score', ascending=False) if not alerts_df.empty else alerts_df
