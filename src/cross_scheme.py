import pandas as pd
import numpy as np

def _haversine_distance(lat1, lon1, lat2, lon2):
    """Compute Haversine distance in meters between two lat/lon pairs."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def generate_simulated_cross_schemes(works_df):
    """Generate reference projects from parallel central/state schemes for cross-matching."""
    schemes = ['PMGSY (Pradhan Mantri Gram Sadak Yojana)', 'Jal Jeevan Mission', 'Smart Cities Mission', 'Swachh Bharat Abhiyan']
    cross_records = []
    
    np.random.seed(101)
    if 'LATITUDE' not in works_df.columns:
        return pd.DataFrame()
        
    sample_works = works_df.dropna(subset=['LATITUDE', 'LONGITUDE']).sample(min(len(works_df), 120), random_state=42)
    
    for idx, row in sample_works.iterrows():
        # Inject ~12% true spatial overlap for cross-scheme duplicate funding demonstration
        is_overlap = np.random.rand() < 0.12
        scheme_name = np.random.choice(schemes)
        
        if is_overlap:
            offset_lat = np.random.uniform(-0.0002, 0.0002) # ~20 meters
            offset_lon = np.random.uniform(-0.0002, 0.0002)
            cross_desc = f"Parallel Asset Construction: {row['WORK_DESCRIPTION']}"
        else:
            offset_lat = np.random.uniform(-0.05, 0.05) # ~5 kilometers
            offset_lon = np.random.uniform(-0.05, 0.05)
            cross_desc = f"Standard Scheme Works under {scheme_name}"

        cross_records.append({
            'CROSS_SCHEME_ID': f"SCHEME-REF-{1000+idx}",
            'PARALLEL_SCHEME_NAME': scheme_name,
            'SCHEME_PROJECT_TITLE': cross_desc,
            'CROSS_LATITUDE': row['LATITUDE'] + offset_lat,
            'CROSS_LONGITUDE': row['LONGITUDE'] + offset_lon,
            'SANCTIONED_AMOUNT_INR': row['ESTIMATED_COST'],
            'STATE_NAME': row['STATE_NAME'],
            'DISTRICT_NAME': row['DISTRICT_NAME']
        })
        
    return pd.DataFrame(cross_records)

def detect_cross_scheme_funding(works_df, cross_df=None):
    """
    Detect potential double-funding where the same asset is claimed under MPLADS
    and another scheme (PMGSY, Jal Jeevan, etc.) based on spatial proximity (<100m).
    """
    df = works_df.copy()
    
    if cross_df is None or cross_df.empty:
        cross_df = generate_simulated_cross_schemes(df)
        
    df['IS_CROSS_SCHEME_DUPLICATE'] = False
    df['CROSS_SCHEME_MATCH_NAME'] = None
    df['CROSS_SCHEME_MATCH_TITLE'] = None
    df['CROSS_SCHEME_DISTANCE_METERS'] = np.nan
    df['CROSS_SCHEME_RISK_SCORE'] = 0.0

    if cross_df.empty or 'LATITUDE' not in df.columns:
        return df, cross_df

    # Match works in same district
    for idx, row in df.iterrows():
        if pd.isna(row.get('LATITUDE')) or pd.isna(row.get('LONGITUDE')):
            continue
            
        district_matches = cross_df[cross_df['DISTRICT_NAME'] == row['DISTRICT_NAME']]
        if district_matches.empty:
            continue
            
        min_dist = float('inf')
        best_match = None
        
        for _, c_row in district_matches.iterrows():
            dist = _haversine_distance(
                row['LATITUDE'], row['LONGITUDE'],
                c_row['CROSS_LATITUDE'], c_row['CROSS_LONGITUDE']
            )
            if dist < min_dist:
                min_dist = dist
                best_match = c_row
                
        # Flag if distance is within 100 meters
        if min_dist <= 100.0 and best_match is not None:
            df.at[idx, 'IS_CROSS_SCHEME_DUPLICATE'] = True
            df.at[idx, 'CROSS_SCHEME_MATCH_NAME'] = best_match['PARALLEL_SCHEME_NAME']
            df.at[idx, 'CROSS_SCHEME_MATCH_TITLE'] = best_match['SCHEME_PROJECT_TITLE']
            df.at[idx, 'CROSS_SCHEME_DISTANCE_METERS'] = round(min_dist, 1)
            # Risk penalty inversely proportional to distance
            risk_score = min(100.0, max(60.0, 100.0 - min_dist * 0.4))
            df.at[idx, 'CROSS_SCHEME_RISK_SCORE'] = round(risk_score, 1)

    return df, cross_df
