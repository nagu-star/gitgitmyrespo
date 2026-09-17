import pandas as pd
import numpy as np

def audit_data_quality(df):
    """
    Perform transparent Data Quality Audit on the dataset.
    Returns:
    - audit_summary: dict of overall metrics
    - missing_df: DataFrame of field-wise missing value counts and percentages
    """
    if df is None or df.empty:
        return {
            'total_records': 0,
            'valid_records': 0,
            'duplicate_records': 0,
            'missing_total': 0,
            'data_quality_score': 0.0
        }, pd.DataFrame()
        
    total_records = len(df)
    duplicate_records = int(df.duplicated(subset=['WORK_ID']).sum()) if 'WORK_ID' in df.columns else int(df.duplicated().sum())
    
    missing_counts = df.isnull().sum()
    missing_total = int(missing_counts.sum())
    
    field_audit = []
    for col in df.columns:
        null_cnt = int(missing_counts[col])
        null_pct = round((null_cnt / total_records) * 100, 2)
        dtype = str(df[col].dtype)
        
        # Check invalid values if financial
        invalid_cnt = 0
        if col in ['SANCTION_AMOUNT', 'EXPENDITURE_AMOUNT', 'ESTIMATED_COST']:
            invalid_cnt = int((df[col] < 0).sum())
            
        field_audit.append({
            'Field Name': col,
            'Data Type': dtype,
            'Missing Count': null_cnt,
            'Missing Percentage (%)': null_pct,
            'Invalid Values': invalid_cnt,
            'Status': 'Valid' if null_cnt == 0 and invalid_cnt == 0 else 'Attention Required'
        })
        
    missing_df = pd.DataFrame(field_audit)
    
    # Calculate Data Quality Score (0-100%)
    valid_records = total_records - duplicate_records
    quality_score = max(0.0, min(100.0, round(100.0 - (missing_total / (total_records * len(df.columns)) * 100), 2)))
    
    audit_summary = {
        'total_records': total_records,
        'valid_records': valid_records,
        'duplicate_records': duplicate_records,
        'missing_total': missing_total,
        'data_quality_score': quality_score,
        'available_fields_count': len(df.columns)
    }
    
    return audit_summary, missing_df
