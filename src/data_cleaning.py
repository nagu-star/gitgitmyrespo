import pandas as pd
import numpy as np

def clean_currency_string(val):
    """Convert Indian currency string (e.g. ₹1,261.99 Crore or ₹12,61,99,31,638.69) to float INR."""
    if pd.isna(val) or val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    
    val_str = str(val).replace('₹', '').replace('\xa0', '').replace(',', '').strip()
    
    if 'Crore' in val_str or 'Cr' in val_str:
        num_part = val_str.replace('Crore', '').replace('Cr', '').strip()
        try:
            return float(num_part) * 1e7
        except ValueError:
            return 0.0
    elif 'Lakh' in val_str or 'L' in val_str:
        num_part = val_str.replace('Lakh', '').replace('L', '').strip()
        try:
            return float(num_part) * 1e5
        except ValueError:
            return 0.0
    else:
        try:
            return float(val_str)
        except ValueError:
            return 0.0

def format_inr(val):
    """Format numeric float INR into human-readable Indian currency string (Crores/Lakhs/Rupees)."""
    if pd.isna(val) or val is None:
        return "₹0.00"
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e7:
        return f"{sign}₹{abs_val / 1e7:,.2f} Cr"
    elif abs_val >= 1e5:
        return f"{sign}₹{abs_val / 1e5:,.2f} Lakh"
    else:
        return f"{sign}₹{abs_val:,.2f}"

def clean_works_df(df):
    """Clean and standardize the works DataFrame."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    clean_df = df.copy()
    
    # Financial fields
    financial_cols = ['SANCTION_AMOUNT', 'EXPENDITURE_AMOUNT', 'ESTIMATED_COST']
    for col in financial_cols:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].apply(clean_currency_string)
            
    # Text standardization
    text_cols = ['STATE_NAME', 'DISTRICT_NAME', 'CONSTITUENCY', 'MP_NAME', 'WORK_CATEGORY', 'WORK_DESCRIPTION', 'WORK_STATUS']
    for col in text_cols:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].astype(str).str.strip()
            
    # Numeric progress percentage
    if 'PROGRESS_PERCENTAGE' in clean_df.columns:
        clean_df['PROGRESS_PERCENTAGE'] = pd.to_numeric(clean_df['PROGRESS_PERCENTAGE'], errors='coerce').fillna(0.0)
        clean_df['PROGRESS_PERCENTAGE'] = clean_df['PROGRESS_PERCENTAGE'].clip(0.0, 100.0)
        
    # Date parsing
    date_cols = ['RECOMMENDATION_DATE', 'SANCTION_DATE', 'COMPLETION_DATE']
    for col in date_cols:
        if col in clean_df.columns:
            clean_df[col] = pd.to_datetime(clean_df[col], errors='coerce')
            
    # Year
    if 'YEAR' in clean_df.columns:
        clean_df['YEAR'] = pd.to_numeric(clean_df['YEAR'], errors='coerce').fillna(2024).astype(int)
        
    return clean_df
