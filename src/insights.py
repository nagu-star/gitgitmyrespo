import pandas as pd

def generate_work_explanation(row):
    """
    Generates explainable AI diagnosis for an individual work item.
    Answers: WHY WAS THIS FLAGGED?
    Produces a clear, structured breakdown:
    - Work ID, Location, MP/Constituency
    - Financial & Progress summary (Handling missing data as 'Data unavailable')
    - Risk Score & Level
    - Numbered Risk Reasons
    - Recommended Action for Officers
    """
    if row is None or (isinstance(row, pd.Series) and row.empty):
        return {
            "title": "No Work Selected",
            "reasons": [],
            "recommended_action": "Select a work item to inspect insights.",
            "data_status": "Unavailable",
            "data_status_badge": "✕ Unavailable",
            "delay_info": "Data unavailable"
        }

    w_id = str(row.get('WORK_ID', 'N/A'))
    risk_level = str(row.get('RISK_LEVEL', 'LOW'))
    risk_score = float(row.get('RISK_SCORE', 0.0))
    state = str(row.get('STATE_NAME', 'N/A'))
    district = str(row.get('DISTRICT_NAME', 'N/A'))
    mp_name = str(row.get('MP_NAME', 'N/A'))
    constituency = str(row.get('CONSTITUENCY', 'N/A'))
    category = str(row.get('WORK_CATEGORY', 'N/A'))
    status = str(row.get('WORK_STATUS', 'N/A'))

    # Financial & Progress
    sanc = row.get('SANCTION_AMOUNT')
    exp = row.get('EXPENDITURE_AMOUNT')
    recom = row.get('RECOMMENDED_AMOUNT') # May be NaN/missing in API
    prog = row.get('PROGRESS_PERCENTAGE')

    # Formatting financial amounts with "Data unavailable" handling
    def _format_amt(v):
        if pd.isna(v) or v is None:
            return "Data unavailable"
        v = float(v)
        if abs(v) >= 1e7:
            return f"₹{v / 1e7:,.2f} Cr"
        elif abs(v) >= 1e5:
            return f"₹{v / 1e5:,.2f} Lakh"
        else:
            return f"₹{v:,.2f}"

    recom_str = _format_amt(recom)
    sanc_str = _format_amt(sanc)
    exp_str = _format_amt(exp)
    prog_str = f"{round(float(prog), 1)}%" if pd.notna(prog) else "Data unavailable"

    # Delay Info
    proj_delay = row.get('PROJECTED_DELAY_MONTHS')
    if status == 'Delayed':
        if pd.notna(proj_delay) and float(proj_delay) > 0:
            delay_info = f"Work Delayed (Projected slippage: {round(float(proj_delay), 1)} months)"
        else:
            delay_info = "Significant delay in work completion"
    elif pd.notna(proj_delay) and float(proj_delay) > 3.0:
        delay_info = f"Potential delay detected ({round(float(proj_delay), 1)} months predicted)"
    else:
        delay_info = "On Schedule / Normal Progress"

    # Data Completeness Check
    missing_fields = []
    if pd.isna(sanc) or sanc == 0.0: missing_fields.append("Sanction Amount")
    if pd.isna(exp): missing_fields.append("Expenditure Amount")
    if pd.isna(recom): missing_fields.append("Recommended Amount")
    if pd.isna(prog): missing_fields.append("Work Progress %")

    if len(missing_fields) == 0:
        data_status = "Available"
        data_status_badge = "✓ Available"
    elif len(missing_fields) <= 2:
        data_status = "Partially Available"
        data_status_badge = f"⚠ Partially Available ({', '.join(missing_fields)} missing in API)"
    else:
        data_status = "Unavailable"
        data_status_badge = "✕ Insufficient Data for Risk Assessment"

    # Numbered Reasons Generation
    reasons = []

    # 1. Cost Overrun / Ratio
    if pd.notna(exp) and pd.notna(sanc) and float(sanc) > 0:
        exp_f, sanc_f = float(exp), float(sanc)
        if exp_f > sanc_f:
            diff = exp_f - sanc_f
            pct = round((diff / sanc_f) * 100.0, 1)
            reasons.append(f"Unusual expenditure pattern: Actual expenditure ({exp_str}) exceeds sanctioned budget ({sanc_str}) by {pct}%")
        elif (exp_f / sanc_f) > 0.85 and pd.notna(prog) and float(prog) < 40.0:
            reasons.append(f"Progress discrepancy: High financial utilization ({round(exp_f/sanc_f*100,1)}%) despite low reported physical progress ({prog_str})")

    # 2. Delay & Progress Velocity
    if status == 'Delayed' or (pd.notna(prog) and float(prog) < 30.0 and status in ['Ongoing', 'Incomplete with High Exp']):
        reasons.append(f"Significant delay in work completion: Work progress ({prog_str}) lags behind expected schedule milestone")

    # 3. Anomaly Detection (Isolation Forest)
    if row.get('IS_ANOMALY', False):
        anom_score = row.get('ANOMALY_SCORE', 0.0)
        reasons.append(f"Statistical multivariate anomaly detected by Isolation Forest model (Anomaly Index: {anom_score})")

    # 4. Duplicate Similarity
    if row.get('IS_DUPLICATE_FLAG', False):
        reasons.append("Duplicate work risk: High textual and financial similarity matched with another work in the same district")

    # 5. Geofence / EXIF Mismatch
    if row.get('IS_GEO_MISMATCH', False):
        dist_m = row.get('GEOFENCE_DISTANCE_METERS', 0.0)
        reasons.append(f"Photo geofence GPS variance: EXIF photo location variance is {dist_m}m from sanctioned coordinates")

    # 6. Cartel / Split Tender
    if row.get('IS_SPLIT_TENDER', False):
        reasons.append("Split-tendering bypass flag: Sub-₹25L work package cluster awarded under high vendor concentration")

    # 7. S-Curve Stagnation
    if row.get('IS_STAGNANT_SCURVE', False):
        reasons.append("S-Curve progress stagnation: Financial release lead velocity severely outpaces physical execution speed")

    # 8. Cross-Scheme Double Funding
    if row.get('IS_CROSS_SCHEME_DUPLICATE', False):
        cs_name = row.get('CROSS_SCHEME_MATCH_NAME', 'Parallel Scheme')
        dist_m = row.get('CROSS_SCHEME_DISTANCE_METERS', 0.0)
        reasons.append(f"Cross-scheme double funding: Spatial asset overlap matched with {cs_name} ({dist_m}m proximity)")

    # Fallback if no specific flags
    if not reasons:
        if risk_score >= 50.0:
            reasons.append("Elevated risk parameters based on combined financial allocation and district execution metrics")
        else:
            reasons.append("Normal execution profile: All financial, physical progress, and compliance bounds are satisfied")

    # Score Point Breakdown Calculation
    raw_factors = row.get('RISK_FACTORS', '')
    score_breakdown = [{'factor': 'Base Risk Baseline', 'pts': 15.0, 'description': 'Standard baseline risk allocation for monitored public works'}]
    
    if pd.notna(raw_factors) and isinstance(raw_factors, str) and raw_factors != 'Standard implementation profile':
        factor_items = [f.strip() for f in raw_factors.split(';') if f.strip()]
        for item in factor_items:
            score_breakdown.append({
                'factor': item.split('(')[0].strip() if '(' in item else item.split(':')[0].strip(),
                'pts': item, # retains exact (+XX.X pts) detail string
                'description': item
            })

    # Convert to numbered list format
    numbered_reasons = [f"{idx+1}. {r}" for idx, r in enumerate(reasons)]

    # Recommended Action
    if risk_score >= 75.0 or risk_level == 'CRITICAL':
        rec_action = "Officer verification required: Immediate physical site inspection by District Authority and detailed audit of expenditure vouchers."
    elif risk_score >= 55.0 or risk_level == 'HIGH':
        rec_action = "Officer verification required: Request updated progress photos, technical sanction documents, and stage completion certificate from Implementing District Agency (IDA)."
    elif risk_score >= 35.0 or risk_level == 'MEDIUM':
        rec_action = "Routine administrative review: Monitor physical progress milestones and track next tranche disbursement."
    else:
        rec_action = "Proceed with standard implementation and routine periodic reporting."

    return {
        'work_id': w_id,
        'mp_name': mp_name,
        'constituency': constituency,
        'state': state,
        'district': district,
        'category': category,
        'status': status,
        'recommended_amount_str': recom_str,
        'sanctioned_amount_str': sanc_str,
        'expenditure_amount_str': exp_str,
        'progress_pct_str': prog_str,
        'delay_info': delay_info,
        'risk_score': round(risk_score, 1),
        'risk_level': risk_level,
        'reasons': numbered_reasons,
        'score_breakdown': score_breakdown,
        'risk_factors_summary': raw_factors if pd.notna(raw_factors) else '; '.join(reasons),
        'recommended_action': rec_action,
        'data_status': data_status,
        'data_status_badge': data_status_badge,
        'title': f"AI Diagnostic Report — Work ID {w_id} (Risk Score: {round(risk_score, 1)}/100, Level: {risk_level})",
        'responsible_ai_notice': "AI identifies risk indicators; final verification is performed by the authorized officer."
    }

