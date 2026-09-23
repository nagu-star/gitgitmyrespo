import os
import sys
import pickle
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.data_ingestion import build_mplads_dataset
from src.data_cleaning import clean_works_df
from src.data_validation import audit_data_quality
from src.feature_engineering import engineer_mplads_features
from src.anomaly_detection import detect_anomalies
from src.duplicate_detection import detect_duplicate_works
from src.photo_geofence import audit_photo_geofence_and_hashes
from src.risk_scoring import compute_risk_scoring
from src.alerts import generate_risk_alerts
from src.forecasting import compute_predictive_early_warnings
from src.compliance import audit_mplads_compliance

def precompute():
    print("Precomputing MPLADS AI analytics pipeline...")
    t0 = time.time()
    
    states_df, state_tiles_df, mps_df, raw_works_df, metadata = build_mplads_dataset()
    cleaned_df = clean_works_df(raw_works_df)
    fe_df = engineer_mplads_features(cleaned_df)
    anom_df = detect_anomalies(fe_df)
    dup_df, duplicates_matrix = detect_duplicate_works(anom_df)
    photo_df = audit_photo_geofence_and_hashes(dup_df)
    risk_df = compute_risk_scoring(photo_df)
    forecasted_df, early_warnings_df = compute_predictive_early_warnings(risk_df)
    compliant_df, compliance_summary_df = audit_mplads_compliance(forecasted_df)
    alerts_df = generate_risk_alerts(compliant_df)
    audit_summary, missing_df = audit_data_quality(raw_works_df)
    
    bundle = {
        'states_df': states_df,
        'state_tiles_df': state_tiles_df,
        'mps_df': mps_df,
        'works_df': compliant_df,
        'duplicates_matrix': duplicates_matrix,
        'alerts_df': alerts_df,
        'early_warnings_df': early_warnings_df,
        'compliance_summary_df': compliance_summary_df,
        'audit_summary': audit_summary,
        'missing_df': missing_df,
        'metadata': metadata
    }
    
    out_path = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'pipeline_bundle.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(bundle, f)
        
    print(f"Precomputed bundle saved to {out_path} in {round(time.time() - t0, 2)}s!")

if __name__ == '__main__':
    precompute()
