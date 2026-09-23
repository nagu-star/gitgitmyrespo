import pandas as pd
import numpy as np

def compute_scurve_delay_velocity(works_df):
    """
    Track physical vs financial progress milestones across project lifecycle.
    Flag works with high financial expenditure velocity but low physical progress (Stagnant projects).
    """
    df = works_df.copy()
    
    df['FINANCIAL_PROGRESS_PCT'] = 0.0
    df['PROGRESS_GAP_PCT'] = 0.0
    df['IS_STAGNANT_SCURVE'] = False
    df['DELAY_VELOCITY_SCORE'] = 0.0
    df['SCURVE_STATUS_FLAG'] = 'On Track'

    for idx, row in df.iterrows():
        sanc = row.get('SANCTION_AMOUNT', 0.0)
        exp = row.get('EXPENDITURE_AMOUNT', 0.0)
        phys = row.get('PROGRESS_PERCENTAGE', 0.0)
        planned = row.get('PLANNED_PROGRESS_PERCENT', phys)
        
        fin_pct = (exp / sanc * 100.0) if sanc > 0 else 0.0
        df.at[idx, 'FINANCIAL_PROGRESS_PCT'] = round(fin_pct, 1)
        
        # Progress gap: Financial release vs physical completion OR Planned physical vs actual physical
        fin_phys_gap = fin_pct - phys
        planned_phys_gap = planned - phys
        df.at[idx, 'PROGRESS_GAP_PCT'] = round(max(fin_phys_gap, planned_phys_gap), 1)
        
        # High financial release (>65%) with low physical progress (<35%) = Stagnant Fraud Pattern
        if (fin_pct >= 65.0 and phys <= 35.0) or (planned_phys_gap > 35.0 and row.get('WORK_STATUS') == 'Delayed'):
            df.at[idx, 'IS_STAGNANT_SCURVE'] = True
            df.at[idx, 'SCURVE_STATUS_FLAG'] = 'Critical Financial Lead Velocity / Stagnant Work'
            delay_score = min(100.0, 50.0 + max(fin_phys_gap, planned_phys_gap) * 0.8)
            df.at[idx, 'DELAY_VELOCITY_SCORE'] = round(delay_score, 1)
        elif fin_phys_gap > 20.0 or planned_phys_gap > 20.0:
            df.at[idx, 'SCURVE_STATUS_FLAG'] = 'Moderate Progress Lag'
            df.at[idx, 'DELAY_VELOCITY_SCORE'] = round(min(100.0, max(fin_phys_gap, planned_phys_gap) * 1.2), 1)
        else:
            df.at[idx, 'SCURVE_STATUS_FLAG'] = 'On Schedule'
            df.at[idx, 'DELAY_VELOCITY_SCORE'] = 0.0

    return df
