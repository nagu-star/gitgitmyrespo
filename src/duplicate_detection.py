import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def detect_duplicate_works(df, similarity_threshold=0.75):
    """
    Detect potentially duplicate or highly similar works using fast vectorized
    TF-IDF text similarity and location/financial matching.
    """
    if df is None or df.empty or len(df) < 2:
        if df is not None:
            df['IS_DUPLICATE_FLAG'] = False
            df['DUPLICATE_INFO'] = "No duplicate detected"
        return df, pd.DataFrame()

    out_df = df.copy()
    out_df['IS_DUPLICATE_FLAG'] = False
    out_df['DUPLICATE_INFO'] = "No duplicate detected"

    descriptions = out_df['WORK_DESCRIPTION'].fillna('').tolist()
    
    # Vectorized TF-IDF Cosine Similarity
    vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    sim_matrix = cosine_similarity(tfidf_matrix)

    # Get upper triangle indices where similarity >= 0.65
    row_indices, col_indices = np.where(np.triu(sim_matrix, k=1) >= 0.65)

    duplicate_pairs = []

    for i, j in zip(row_indices, col_indices):
        sim_score = sim_matrix[i, j]
        row_a = out_df.iloc[i]
        row_b = out_df.iloc[j]

        same_state = (row_a['STATE_NAME'] == row_b['STATE_NAME'])
        same_cat = (row_a['WORK_CATEGORY'] == row_b['WORK_CATEGORY'])
        sanc_a = max(1.0, float(row_a['SANCTION_AMOUNT']))
        sanc_b = max(1.0, float(row_b['SANCTION_AMOUNT']))
        amt_diff_pct = abs(sanc_a - sanc_b) / sanc_a

        if sim_score >= similarity_threshold or (sim_score >= 0.65 and same_state and same_cat and amt_diff_pct < 0.15):
            sim_pct = round(float(sim_score) * 100.0, 1)

            out_df.at[row_a.name, 'IS_DUPLICATE_FLAG'] = True
            out_df.at[row_b.name, 'IS_DUPLICATE_FLAG'] = True

            common_chars = f"Category: {row_a['WORK_CATEGORY']} | Location: {row_a['STATE_NAME']}"
            reason = f"High text similarity ({sim_pct}%) & matching financial scope between {row_a['WORK_ID']} and {row_b['WORK_ID']}"

            out_df.at[row_a.name, 'DUPLICATE_INFO'] = f"Potential Duplicate with {row_b['WORK_ID']} ({sim_pct}% similarity)"
            out_df.at[row_b.name, 'DUPLICATE_INFO'] = f"Potential Duplicate with {row_a['WORK_ID']} ({sim_pct}% similarity)"

            duplicate_pairs.append({
                'Work A ID': row_a['WORK_ID'],
                'Work B ID': row_b['WORK_ID'],
                'State': row_a['STATE_NAME'],
                'Category': row_a['WORK_CATEGORY'],
                'Sanction Amount A': row_a['SANCTION_AMOUNT'],
                'Sanction Amount B': row_b['SANCTION_AMOUNT'],
                'Similarity Score (%)': sim_pct,
                'Common Characteristics': common_chars,
                'Status': 'Potential Duplicate / Requires Verification',
                'Reason for Flagging': reason
            })

    duplicates_matrix_df = pd.DataFrame(duplicate_pairs)
    return out_df, duplicates_matrix_df
