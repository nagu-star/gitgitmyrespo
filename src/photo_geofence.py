import os
import sys

# Ensure root project directory is in sys.path when executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np

try:
    from src.cross_scheme import _haversine_distance
except ModuleNotFoundError:
    from cross_scheme import _haversine_distance

def audit_photo_geofence_and_hashes(works_df, distance_threshold_meters=100.0):
    """
    Audit progress photos for spatial geofence mismatches (EXIF GPS vs Sanctioned Work location > 100m)
    and detect duplicate perceptual photo hashes across different works or constituencies.
    """
    df = works_df.copy()
    
    df['IS_GEO_MISMATCH'] = False
    df['GEOFENCE_DISTANCE_METERS'] = np.nan
    df['IS_DUPLICATE_PHOTO'] = False
    df['DUPLICATE_PHOTO_MATCH_WORK_ID'] = None
    df['PHOTO_INTEGRITY_RISK_SCORE'] = 0.0

    if 'LATITUDE' not in df.columns or 'EXIF_LATITUDE' not in df.columns:
        return df

    # 1. Geofence Distance Audit
    for idx, row in df.iterrows():
        lat, lon = row.get('LATITUDE'), row.get('LONGITUDE')
        exif_lat, exif_lon = row.get('EXIF_LATITUDE'), row.get('EXIF_LONGITUDE')
        
        if pd.notna(lat) and pd.notna(lon) and pd.notna(exif_lat) and pd.notna(exif_lon):
            dist = _haversine_distance(lat, lon, exif_lat, exif_lon)
            df.at[idx, 'GEOFENCE_DISTANCE_METERS'] = round(dist, 1)
            if dist > distance_threshold_meters:
                df.at[idx, 'IS_GEO_MISMATCH'] = True

    # 2. Duplicate Photo Hash Audit
    if 'PHOTO_HASH' in df.columns:
        hash_map = {}
        for idx, row in df.iterrows():
            phash = row.get('PHOTO_HASH')
            if pd.notna(phash) and str(phash).strip() != "":
                if phash in hash_map:
                    first_idx, first_work_id = hash_map[phash]
                    df.at[idx, 'IS_DUPLICATE_PHOTO'] = True
                    df.at[idx, 'DUPLICATE_PHOTO_MATCH_WORK_ID'] = first_work_id
                    df.at[first_idx, 'IS_DUPLICATE_PHOTO'] = True
                    df.at[first_idx, 'DUPLICATE_PHOTO_MATCH_WORK_ID'] = row['WORK_ID']
                else:
                    hash_map[phash] = (idx, row['WORK_ID'])

    # 3. Calculate Photo Integrity Risk Score (0-100)
    for idx, row in df.iterrows():
        score = 0.0
        if row.get('IS_GEO_MISMATCH', False):
            dist = row.get('GEOFENCE_DISTANCE_METERS', 0.0)
            score += min(60.0, 30.0 + (dist / 100.0) * 10.0)
        if row.get('IS_DUPLICATE_PHOTO', False):
            score += 45.0
            
        df.at[idx, 'PHOTO_INTEGRITY_RISK_SCORE'] = round(min(100.0, score), 1)

    return df
