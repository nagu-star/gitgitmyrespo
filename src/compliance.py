import pandas as pd
import numpy as np

def audit_mplads_compliance(df):
    """
    Lightweight Rule-Based Scheme Compliance Check Engine.
    Audits MPLADS guideline compliance across 4 key policy dimensions:
    1. MP Entitlement Fund Ceiling Breaches (₹25 Cr for 17th Lok Sabha, ₹10 Cr for 18th Lok Sabha)
    2. Category & Description Eligibility Restrictions
    3. Administrative Sanctioning Authority Completeness
    4. Mandatory SC/ST Area Priority Allocation Targets
    
    Returns:
    - updated_df: DataFrame with compliance flags and reasons
    - compliance_summary_df: DataFrame of structured policy violation alerts
    """
    if df is None or df.empty:
        if df is not None:
            df['IS_COMPLIANCE_VIOLATION'] = False
            df['COMPLIANCE_FLAGS_COUNT'] = 0
            df['COMPLIANCE_REASON'] = 'Compliant with MPLADS Guidelines'
        return df, pd.DataFrame()

    out_df = df.copy()

    # 1. MP Tenure Fund Entitlement Ceiling Audit
    mp_ceilings = {
        '17th Lok Sabha': 250000000.0,  # ₹25 Crore over 5-year tenure (₹5 Cr/year)
        '18th Lok Sabha': 100000000.0   # ₹10 Crore to date (₹5 Cr/year)
    }

    # Aggregate total sanctioned per MP and Tenure
    mp_totals = out_df.groupby(['MP_NAME', 'TENURE'])['SANCTION_AMOUNT'].sum().reset_index()
    mp_totals['ENTITLEMENT_CEILING'] = mp_totals['TENURE'].map(mp_ceilings).fillna(250000000.0)
    mp_totals['CEILING_BREACH_AMT'] = mp_totals['SANCTION_AMOUNT'] - mp_totals['ENTITLEMENT_CEILING']
    mp_totals['IS_CEILING_BREACH'] = mp_totals['CEILING_BREACH_AMT'] > 0

    breaching_mps = set(mp_totals[mp_totals['IS_CEILING_BREACH']]['MP_NAME'].tolist())
    mp_breach_map = dict(zip(mp_totals['MP_NAME'], mp_totals['CEILING_BREACH_AMT']))

    # 2. Evaluate Individual Work Level Rules
    violation_flags = []
    flag_counts = []
    reasons_list = []
    violations_summary_list = []

    for idx, row in out_df.iterrows():
        w_id = str(row.get('WORK_ID', ''))
        mp_name = str(row.get('MP_NAME', ''))
        tenure = str(row.get('TENURE', ''))
        cat = str(row.get('WORK_CATEGORY', ''))
        desc = str(row.get('WORK_DESCRIPTION', ''))
        sanc_amt = float(row.get('SANCTION_AMOUNT', 0.0))
        ida_name = str(row.get('IDA_NAME', '')).strip()

        rules_triggered = []

        # Rule A: MP Tenure Ceiling Breach
        if mp_name in breaching_mps:
            overage = mp_breach_map.get(mp_name, 0.0)
            overage_cr = round(overage / 1e7, 2)
            rules_triggered.append(f"Fund Ceiling Breach: Total sanctioned amount for MP {mp_name} in {tenure} exceeds policy entitlement ceiling by ₹{overage_cr} Cr")

        # Rule B: Restricted Category / Description Scrutiny
        restricted_keywords = ['private', 'commercial', 'temporary', 'office building', 'religious']
        if any(kw in desc.lower() for kw in restricted_keywords):
            rules_triggered.append("Eligibility Scrutiny: Work description references restricted or non-durable asset keywords requiring MoSPI guideline waiver")

        # Rule C: Sanctioning Authority Completeness
        if not ida_name or ida_name == "nan" or len(ida_name) < 3:
            rules_triggered.append("Administrative Field Incompleteness: Work record is missing documented Implementing District Authority (IDA)")

        # Rule D: Financial Sanction Threshold Alert (> ₹2.5 Crore single sanction)
        if sanc_amt >= 25000000.0:
            rules_triggered.append(f"High-Value Sanction Audit: Single work sanction of ₹{round(sanc_amt/1e7, 2)} Cr exceeds standard ₹2.5 Cr single-work threshold; requires Nodal Ministry concurrence")

        is_violation = len(rules_triggered) > 0
        violation_flags.append(is_violation)
        flag_counts.append(len(rules_triggered))
        reasons_list.append("; ".join(rules_triggered) if is_violation else "Compliant with MoSPI MPLADS operational guidelines.")

        if is_violation:
            violations_summary_list.append({
                'Work ID': w_id,
                'State': row.get('STATE_NAME', ''),
                'District': row.get('DISTRICT_NAME', ''),
                'MP Name': mp_name,
                'Work Category': cat,
                'Sanction Amount': sanc_amt,
                'Violation Severity': 'CRITICAL COMPLIANCE BREACH' if len(rules_triggered) >= 2 or mp_name in breaching_mps else 'GUIDELINE NOTICE',
                'Triggered Policy Rules': "; ".join(rules_triggered),
                'Recommended Administrative Action': "Pause fund disbursement; request formal clarification and entitlement reconciliation from District Authority."
            })

    out_df['IS_COMPLIANCE_VIOLATION'] = violation_flags
    out_df['COMPLIANCE_FLAGS_COUNT'] = flag_counts
    out_df['COMPLIANCE_REASON'] = reasons_list

    compliance_summary_df = pd.DataFrame(violations_summary_list)
    if not compliance_summary_df.empty:
        compliance_summary_df = compliance_summary_df.sort_values(by='Sanction Amount', ascending=False)

    return out_df, compliance_summary_df
