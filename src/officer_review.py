import os
import json
import pandas as pd
from datetime import datetime

REVIEWS_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw', 'officer_reviews.json')

def load_officer_reviews():
    """Load persisted officer reviews from JSON file."""
    if os.path.exists(REVIEWS_FILE_PATH):
        try:
            with open(REVIEWS_FILE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Error loading officer reviews] {e}")
    return {}

def save_officer_review(work_id, review_status, officer_remark="", officer_role="Authorized Officer"):
    """Save or update an officer review for a specific Work ID."""
    reviews = load_officer_reviews()
    
    work_id_str = str(work_id)
    reviews[work_id_str] = {
        "status": review_status,
        "remark": officer_remark,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "officer_role": officer_role
    }
    
    os.makedirs(os.path.dirname(REVIEWS_FILE_PATH), exist_ok=True)
    try:
        with open(REVIEWS_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(reviews, f, indent=2)
        return True
    except Exception as e:
        print(f"[Error saving officer review] {e}")
        return False

def merge_officer_reviews_into_df(df):
    """
    Merge persisted officer review statuses and remarks into the main works DataFrame.
    Defaults unreviewed items to 'Pending Review'.
    """
    if df is None or df.empty:
        return df

    out_df = df.copy()
    reviews = load_officer_reviews()

    statuses = []
    remarks = []
    timestamps = []
    officer_roles = []

    for idx, row in out_df.iterrows():
        w_id = str(row.get('WORK_ID', ''))
        if w_id in reviews:
            rev_info = reviews[w_id]
            statuses.append(rev_info.get("status", "Pending Review"))
            remarks.append(rev_info.get("remark", ""))
            timestamps.append(rev_info.get("timestamp", ""))
            officer_roles.append(rev_info.get("officer_role", "Authorized Officer"))
        else:
            statuses.append("Pending Review")
            remarks.append("")
            timestamps.append("")
            officer_roles.append("")

    out_df['REVIEW_STATUS'] = statuses
    out_df['OFFICER_REMARK'] = remarks
    out_df['REVIEW_TIMESTAMP'] = timestamps
    out_df['OFFICER_ROLE'] = officer_roles

    return out_df
