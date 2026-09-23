# 🏛️ AuditGrievance AI: Production-Grade MPLADS Anomaly & Fraud Detection Platform

> **An AI-powered audit copilot engineered for MoSPI to safeguard ₹8,300+ Cr in MPLADS funds.** By replacing passive chart dashboards with explainable machine learning, spatial integrity checks, and financial forensic algorithms, AuditGrievance AI detects hidden corruption patterns, cost inflation, ghost assets, and split-tendering before funds are disbursed.

---

## ⭐ Why This Stands Out (SIH 2026 USPs)

Unlike standard hackathon dashboards that merely plot financial utilization, **AuditGrievance AI** operates as an audit-grade intelligence platform:

- 🔄 **Cross-Scheme Double Funding Detector**: Prevents contractors from claiming the same physical infrastructure under MPLADS and parallel central/state schemes (e.g., PMGSY, Jal Jeevan Mission, Smart Cities).
- 🏷️ **Vendor Cartel & Split-Tender Flagging**: Detects market concentration (HHI) and flags artificial work splitting (e.g., multiple sub-₹25L packages) designed to bypass mandatory e-tendering rules.
- 📍 **Photo & Geotag Integrity Verification**: Detects spatial spoofing (EXIF GPS vs sanctioned location mismatch >100m) and identifies duplicate progress photos using perceptual hashing.
- 📈 **S-Curve Progress & Delay Velocity**: Tracks physical vs. financial milestone burn rates to flag stagnant projects where 90%+ funds were disbursed with minimal physical completion.
- 🔍 **Explainable AI (XAI) Diagnostic Cards**: Provides human-auditor-readable reasoning ("Why flagged?") alongside composite 0–100 Risk Scores for every alert.

---

## 📊 Data Availability, Governance & Demo Sandbox

> [!IMPORTANT]
> **Data Transparency Note**: As of 2026, the official MoSPI eSAKSHI portal enforces session-gated, role-based access control. No unauthenticated public row-level REST API or bulk CSV dataset exists for individual 18th Lok Sabha work items. 

To deliver a credible, production-ready prototype, AuditGrievance AI utilizes a **hybrid data architecture**:
1. **Live Official Aggregates**: Real-time HTTP ingestion from public MoSPI REST endpoints for State, Tenure, Constituency, and Aggregate Financial Tiles.
2. **Synthetic Audit Sandbox**: High-fidelity, rule-based synthetic work-level records engineered to replicate real-world MPLADS work distributions, cost matrices, vendor allocations, and edge-case fraud vectors for hackathon demonstration.

| Data Type | Source | Access Status | Prototype Usage |
| :--- | :--- | :--- | :--- |
| **State & Union Territory Metadata** | MoSPI REST API (`/getStateData`) | Public / Live | Dynamic state dropdowns & regional mapping |
| **Tenure & MP Mappings** | MoSPI REST API (`/getTenureData`, `/getMpAndConstCombo`) | Public / Live | MP & constituency filter resolution |
| **Aggregate Financial & Work Counts** | MoSPI REST API (`/getTilesData`) | Public / Live | Macro financial limit & work tally verification |
| **Work-Level Financials & Vendors** | Synthetic MoSPI Sandbox Generator | Simulated Sandbox | Micro-level anomaly detection & audit algorithms |
| **Asset Photos & EXIF Geotags** | Synthetic Spatial Engine | Simulated Sandbox | Geofence verification & photo duplication checks |

---

## 🚀 Core Features & AI Workflow

### 🔄 End-to-End System Workflow

```
[ MoSPI Public APIs + Sandbox Work Data ]
                  │
                  ▼
      [ Ingestion & Data Cleaning ]
                  │
                  ▼
     [ Multi-Dimensional Analytics ]
                  │
                  ▼
  [ ML Outlier & Forensic Fraud Engines ]
  ├── Isolation Forest & LOF (Cost/Progress)
  ├── TF-IDF Vector Cosine (Duplicate Works)
  ├── Cross-Scheme Spatial Overlap Engine
  ├── HHI Vendor Cartel & Split-Tender Analyzer
  └── Photo EXIF Geofence Auditor
                  │
                  ▼
      [ Composite Risk Scoring (0-100) ]
                  │
                  ▼
    [ Explainable AI (XAI) Alert Engine ]
                  │
                  ▼
[ Multi-Role Audit Dashboard (MP / Nodal / MoSPI) ]
```

### 🎯 Key Capabilities

1. **Cross-Scheme Double Funding Detection (USP)**:
   - Evaluates work coordinates and descriptions against simulated national scheme databases (PMGSY, Jal Jeevan, Smart City).
   - Flagged when spatial proximity (<50m) and asset type match across different funding sources.

2. **Vendor Cartel & Split-Tender Detection (USP)**:
   - Computes **Herfindahl-Hirschman Index (HHI)** per district to detect vendor monopolies.
   - Identifies suspicious clusters of works budgeted just below statutory tender thresholds (e.g., ₹24.5 Lakhs) awarded to identical contractors within short time windows.

3. **Photo & Geotag Integrity Verification (USP)**:
   - Audits project progress photos by cross-referencing embedded EXIF coordinates against official sanctioned locations (>100m variance threshold).
   - Uses perceptual hashing to flag reused progress photos across different works or constituencies.

4. **S-Curve Delay Velocity & Milestone Tracking (USP)**:
   - Plots planned vs. actual progress curves across work lifecycles.
   - Flagged when financial expenditure rate significantly leads physical progress (e.g., 90% funds released at 15% physical progress).

5. **Financial Anomaly & Cost Overrun Detection**:
   - Leverages **Isolation Forest**, **Local Outlier Factor (LOF)**, and **IQR Statistical Outlier Filtering** to flag extreme cost variances relative to work categories and district benchmarks.

6. **Semantic Duplicate Work Identification**:
   - Applies **TF-IDF Vector Space Representation** and **Cosine Similarity** on work titles, descriptions, and financial bounds to catch redundant project approvals.

### 🛡️ Risk Scoring & Explainability Engine

Every analyzed work item receives a **Composite Risk Score (0–100)** categorizing it into `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL` risk tiers:

$$\text{Risk Score} = w_1 S_{\text{cost}} + w_2 S_{\text{dup}} + w_3 S_{\text{split}} + w_4 S_{\text{geo}} + w_5 S_{\text{cross}}$$

For every flagged anomaly, the platform generates an **Explainable AI (XAI) Diagnostic Card**:
- 📌 **Primary Flag**: *Split-Tender Pattern Detected*
- 💡 **Diagnostic Reasoning**: *Work #4928 (₹24.8L) and Work #4931 (₹24.7L) approved within 48 hrs for same road segment under Contractor X, bypassing ₹25L tender threshold.*
- 🛡️ **Recommended Audit Action**: *Freeze fund disbursement; initiate District Magistrate physical verification.*

---

## 🔄 Data Flow for SIH Demo

```
 ┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
 │ 1. Ingest Official   │────>│ 2. Generate Demo     │────>│ 3. Feature           │
 │    MoSPI Aggregates  │     │    Work-Level Data   │     │    Engineering       │
 └──────────────────────┘     └──────────────────────┘     └──────────────────────┘
                                                                      │
 ┌──────────────────────┐     ┌──────────────────────┐                ▼
 │ 6. Render Executive  │<────│ 5. Calculate Risk    │<────┌──────────────────────┐
 │    Audit Dashboard   │     │    Score & XAI Cards │     │ 4. Run AI Forensic   │
 └──────────────────────┘     └──────────────────────┘     │    & ML Engines      │
                                                           └──────────────────────┘
```

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

### 1. Prerequisites
Ensure Python 3.9+ is installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Execute Automated Unit & Pipeline Tests
```bash
python tests/test_pipeline.py
```

### 4. Launch Executive Audit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to interact with the platform.
