import pandas as pd

def generate_work_explanation(row):
    """
    Generates explainable AI diagnosis for an individual work item.
    Answers: WHY WAS THIS FLAGGED?
    """
    if row is None or (isinstance(row, pd.Series) and row.empty):
        return {"title": "No Work Selected", "reasons": [], "recommendation": "Select a work item to inspect insights."}

    w_id = row.get('WORK_ID', 'N/A')
    risk_level = row.get('RISK_LEVEL', 'LOW')
    risk_score = row.get('RISK_SCORE', 0.0)

    reasons = []
    
    # 1. Expenditure vs Sanction
    exp = row.get('EXPENDITURE_AMOUNT', 0.0)
    sanc = row.get('SANCTION_AMOUNT', 0.0)
    if exp > sanc:
        diff = exp - sanc
        pct = round((diff / max(1.0, sanc)) * 100, 1)
        reasons.append(f"✓ Expenditure Overrun: Actual expenditure (₹{exp:,.2f}) exceeds sanctioned amount (₹{sanc:,.2f}) by ₹{diff:,.2f} (+{pct}%)")

    # 2. Utilization vs Progress
    util = row.get('UTILIZATION_PCT', 0.0)
    prog = row.get('PROGRESS_PERCENTAGE', 0.0)
    if util > 80.0 and prog < 50.0:
        reasons.append(f"✓ Progress Discrepancy: High financial utilization ({util}%) despite low reported physical progress ({prog}%)")
    elif util < 25.0 and row.get('WORK_STATUS') == 'Delayed':
        reasons.append(f"✓ Low Utilization & Prolonged Delay: Project is marked as Delayed with only {util}% funds utilized")

    # 3. Anomaly Detection
    if row.get('IS_ANOMALY', False):
        reasons.append(f"✓ Statistical Multivariate Outlier: Isolation Forest anomaly index = {row.get('ANOMALY_SCORE', 0.0)}")

    # 4. Duplicate Similarity
    if row.get('IS_DUPLICATE_FLAG', False):
        reasons.append(f"✓ Potential Duplicate Detected: {row.get('DUPLICATE_INFO', 'High text description & financial similarity with another work')}")

    if not reasons:
        reasons.append("✓ Normal Execution: All financial, progress, and timing parameters fall within standard operational bounds.")

    explanation = {
        'work_id': w_id,
        'title': f"AI Diagnostic Report for {w_id} (Risk Level: {risk_level}, Score: {risk_score}/100)",
        'reasons': reasons,
        'responsible_ai_notice': "IMPORTANT: Anomaly ≠ Fraud. Risk Score ≠ Proof of Wrongdoing. This AI alert highlights statistical and procedural patterns requiring human-in-the-loop review.",
        'suggested_action': "District Nodal Officer should request updated progress photos, audit vouchers, and technical sanction documents from the Implementing District Agency (IDA)."
    }
    return explanation
