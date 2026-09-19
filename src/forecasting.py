import pandas as pd
import numpy as np

def compute_predictive_early_warnings(df):
    """
    Predictive / Early-Warning Engine for ongoing MPLADS works.
    Projects final cost overrun and completion delays BEFORE works cross reactive anomaly thresholds.
    
    Returns:
    - updated_df: DataFrame with predictive metrics
    - early_warnings_df: Summary DataFrame of flagged early-warning works
    """
    if df is None or df.empty:
        if df is not None:
            df['PREDICTED_FINAL_COST'] = 0.0
            df['PROJECTED_OVERRUN_PCT'] = 0.0
            df['PROJECTED_DELAY_MONTHS'] = 0.0
            df['EARLY_WARNING_SCORE'] = 0.0
            df['EARLY_WARNING_LEVEL'] = 'NORMAL'
            df['EARLY_WARNING_REASON'] = 'Normal Execution Trajectory'
            df['IS_EARLY_WARNING'] = False
        return df, pd.DataFrame()

    out_df = df.copy()

    predicted_costs = []
    overrun_pcts = []
    projected_delays = []
    warning_scores = []
    warning_levels = []
    warning_reasons = []
    is_early_warning_flags = []
    warnings_summary_list = []

    for idx, row in out_df.iterrows():
        sanc = max(1.0, float(row.get('SANCTION_AMOUNT', 0.0)))
        exp = float(row.get('EXPENDITURE_AMOUNT', 0.0))
        prog = float(row.get('PROGRESS_PERCENTAGE', 0.0))
        status = str(row.get('WORK_STATUS', ''))
        util = float(row.get('UTILIZATION_PCT', (exp / sanc) * 100.0))
        cat = str(row.get('WORK_CATEGORY', 'General'))
        w_id = str(row.get('WORK_ID', ''))

        # Only evaluate ongoing or non-completed works for future projection
        if status in ['Completed'] or prog >= 100.0:
            predicted_costs.append(exp if exp > 0 else sanc)
            overrun_pcts.append(round(((exp - sanc) / sanc) * 100.0, 1) if exp > sanc else 0.0)
            projected_delays.append(0.0)
            warning_scores.append(10.0)
            warning_levels.append('NORMAL')
            warning_reasons.append('Work already completed.')
            is_early_warning_flags.append(False)
            continue

        # 1. Spend-to-Progress Trajectory Extrapolation
        effective_prog = max(5.0, prog)
        spend_progress_ratio = util / effective_prog  # > 1.0 means spending faster than progressing physically

        # Projected final expenditure at 100% completion
        if util > 0 and prog > 0:
            projected_final_cost = round(sanc * (util / prog), 2)
        else:
            projected_final_cost = round(sanc, 2)

        # Projected Cost Overrun Percentage at completion
        proj_overrun_pct = round(((projected_final_cost - sanc) / sanc) * 100.0, 1)

        # 2. Time-to-Progress Speed Extrapolation
        cat_benchmark_months = 12.0
        if 'Roads' in cat or 'Bridges' in cat:
            cat_benchmark_months = 18.0
        elif 'School' in cat or 'Health' in cat:
            cat_benchmark_months = 14.0

        year = int(row.get('YEAR', 2024))
        current_year = 2026
        elapsed_months = max(3.0, (current_year - year) * 12.0 + 6.0)

        monthly_progress_rate = prog / elapsed_months  # % per month
        if monthly_progress_rate > 0:
            remaining_months_needed = (100.0 - prog) / monthly_progress_rate
        else:
            remaining_months_needed = 24.0

        total_projected_months = elapsed_months + remaining_months_needed
        proj_delay_months = round(max(0.0, total_projected_months - cat_benchmark_months), 1)

        # 3. Early Warning Criteria & Score Calculation (0 - 100)
        warning_score = 15.0
        reasons = []

        # Criterion A: Trajectory Overrun Early Warning
        if spend_progress_ratio > 1.25 and util >= 30.0:
            pts = min(40.0, (spend_progress_ratio - 1.0) * 35.0)
            warning_score += pts
            pace_diff_pct = round((spend_progress_ratio - 1.0) * 100.0, 1)
            reasons.append(f"Expenditure pace is {pace_diff_pct}% ahead of physical progress pace; projected final cost ₹{projected_final_cost:,.2f} (+{proj_overrun_pct}% overrun at 100% completion)")

        # Criterion B: Pace Degradation / Delay Projection
        if proj_delay_months >= 4.0 and prog < 70.0:
            pts = min(35.0, proj_delay_months * 3.0)
            warning_score += pts
            reasons.append(f"Physical progress pace ({round(monthly_progress_rate, 2)}%/month) indicates completion delay of ~{proj_delay_months} months beyond expected schedule")

        # Criterion C: High Spend Acceleration at Low Progress
        if util > 65.0 and prog < 45.0:
            warning_score += 25.0
            reasons.append(f"Critical early trajectory mismatch: {round(util,1)}% funds spent with physical progress lagging at {round(prog,1)}%")

        warning_score = round(max(5.0, min(100.0, warning_score)), 1)
        is_early_warning = (warning_score >= 50.0 or proj_overrun_pct >= 15.0 or proj_delay_months >= 6.0)

        if warning_score >= 75.0:
            level = 'HIGH RISK WARNING'
        elif warning_score >= 50.0:
            level = 'MODERATE WARNING'
        elif warning_score >= 30.0:
            level = 'LOW WARNING'
        else:
            level = 'NORMAL'

        predicted_costs.append(projected_final_cost)
        overrun_pcts.append(max(0.0, proj_overrun_pct))
        projected_delays.append(proj_delay_months)
        warning_scores.append(warning_score)
        warning_levels.append(level)
        warning_reasons.append("; ".join(reasons) if reasons else "Project trajectory aligns with standard target benchmarks.")
        is_early_warning_flags.append(is_early_warning)

        if is_early_warning:
            warnings_summary_list.append({
                'Work ID': w_id,
                'State': row.get('STATE_NAME', ''),
                'District': row.get('DISTRICT_NAME', ''),
                'MP Name': row.get('MP_NAME', ''),
                'Work Category': cat,
                'Sanction Amount': sanc,
                'Current Expenditure': exp,
                'Current Progress (%)': prog,
                'Projected Final Cost': projected_final_cost,
                'Projected Overrun (%)': max(0.0, proj_overrun_pct),
                'Projected Delay (Months)': proj_delay_months,
                'Early Warning Score': warning_score,
                'Early Warning Level': level,
                'Trajectory Diagnosis': "; ".join(reasons) if reasons else "Predictive trajectory alert"
            })

    out_df['PREDICTED_FINAL_COST'] = predicted_costs
    out_df['PROJECTED_OVERRUN_PCT'] = overrun_pcts
    out_df['PROJECTED_DELAY_MONTHS'] = projected_delays
    out_df['EARLY_WARNING_SCORE'] = warning_scores
    out_df['EARLY_WARNING_LEVEL'] = warning_levels
    out_df['EARLY_WARNING_REASON'] = warning_reasons
    out_df['IS_EARLY_WARNING'] = is_early_warning_flags

    early_warnings_df = pd.DataFrame(warnings_summary_list)
    if not early_warnings_df.empty:
        early_warnings_df = early_warnings_df.sort_values(by='Early Warning Score', ascending=False)

    return out_df, early_warnings_df
