# MPLADS AI-Powered Monitoring & Analytics Platform

An AI-powered system designed to detect anomalies, fraud indicators, delays, cost overruns, duplicate works, and financial inefficiencies in the Member of Parliament Local Area Development Scheme (MPLADS), using data extracted directly from the official MoSPI MPLADS portal ([https://mplads.mospi.gov.in](https://mplads.mospi.gov.in)).

---

## 🏛️ Official Data Source & API Integration

The platform connects to the official MoSPI MPLADS portal REST API endpoints:
- `/rest/PreLoginDashboardData/getStateData`: 36 Indian States and Union Territories.
- `/rest/PreLoginDashboardData/getTenureData`: Lok Sabha & Rajya Sabha tenures (17th, 18th Lok Sabha, etc.).
- `/rest/PreLoginDashboardData/getMpAndConstCombo`: MPs and Parliamentary Constituencies mapping.
- `/rest/PreLoginDashboardData/getConstituencyData`: State-wise constituency mappings.
- `/rest/PreLoginDashboardData/getTilesData`: Real aggregate financial limits, expenditures, sanctioned works, completed works, and recommended works.
- `/rest/PreLoginCitizenWorkRcmdRest/getReviewDetailsByWork`: Citizen reviews and ratings.

---

## 🚀 Core Features & AI Workflow

The platform implements this end-to-end workflow:
```
MPLADS DATA ↓ DATA PROCESSING ↓ ANALYTICS ↓ ANOMALY DETECTION ↓ RISK SCORING ↓ ALERTS ↓ EXPLAINABLE AI INSIGHTS ↓ DECISION DASHBOARD ↓ HUMAN ACTION
```

1. **Fund Utilization Monitoring**:
   - Calculates Allocated Limit, Sanctioned Amount, Expenditure, Remaining Funds, and Utilization Percentage (`Expenditure / Sanctioned`).
   - Identifies low fund utilization, unusually high utilization, and expenditure variances.
2. **Project / Work Monitoring**:
   - Monitors recommended, sanctioned, ongoing, completed, pending, and delayed works.
3. **Cost Overrun & Anomaly Detection**:
   - Uses **Isolation Forest**, **Local Outlier Factor (LOF)**, and **IQR Statistical Outliers** to flag abnormal cost overruns and progress discrepancies.
4. **Duplicate Work Detection**:
   - Employs **TF-IDF Vector Space Cosine Similarity** on work descriptions, categories, and financial bounds, labeling pairs as `Potential Duplicate / Requires Verification`.
5. **Transparent Risk Scoring**:
   - Assigns composite Risk Scores (0–100) and Risk Levels (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) with itemized evidence.
6. **Explainable AI Insights**:
   - Answers *"WHY WAS THIS FLAGGED?"* for every alert with clear diagnostic bullet points.
7. **Role-Based Views**:
   - Dedicated views for **Member of Parliament**, **District Authority**, **State Nodal Authority**, and **Ministry / MoSPI**.
8. **Data Quality Transparency**:
   - Complete Data Quality Audit showing total records, valid records, missing fields, duplicates, and extraction timestamps.

---

## 🛠️ Project Architecture

```
public gri/
├── app.py                      # Main Streamlit Web Application
├── data/
│   ├── raw/                    # Raw JSON files extracted from MoSPI portal
│   ├── processed/              # Processed datasets
│   └── metadata/               # Data ingestion & audit metadata
├── src/
│   ├── data_ingestion.py       # MoSPI REST API ingestion engine
│   ├── data_cleaning.py        # Currency parser & text standardization
│   ├── data_validation.py      # Data quality auditor
│   ├── feature_engineering.py  # Financial ratios & benchmark ratios
│   ├── analytics.py            # Aggregations & multi-level breakdown
│   ├── anomaly_detection.py    # Isolation Forest & IQR outlier engine
│   ├── duplicate_detection.py  # TF-IDF Cosine Similarity duplicate finder
│   ├── risk_scoring.py         # Composite transparent risk scoring
│   ├── alerts.py               # Dynamic explainable alerts generator
│   └── insights.py             # Explainable AI diagnostic report engine
├── tests/
│   └── test_pipeline.py        # Automated test suite
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation
```

---

## ⚙️ Quick Start Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Unit Test Suite**:
   ```bash
   python tests/test_pipeline.py
   ```

3. **Launch Dashboard App**:
   ```bash
   streamlit run app.py
   ```
