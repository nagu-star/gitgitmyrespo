# pyrefly: ignore [missing-import]
import pickle
# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
import os
import sys

# Set pandas option to allow larger styled tables
pd.set_option("styler.render.max_elements", 2500000)

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_ingestion import build_mplads_dataset
from src.data_cleaning import clean_currency_string, clean_works_df, format_inr
from src.data_validation import audit_data_quality
from src.feature_engineering import engineer_mplads_features
from src.anomaly_detection import detect_anomalies
from src.duplicate_detection import detect_duplicate_works
from src.risk_scoring import compute_risk_scoring
from src.alerts import generate_risk_alerts
from src.insights import generate_work_explanation
from src.forecasting import compute_predictive_early_warnings
from src.compliance import audit_mplads_compliance
from src.analytics import (
    filter_dataset, compute_kpis, aggregate_by_state, 
    aggregate_by_category, aggregate_by_mp
)
from src.officer_review import load_officer_reviews, save_officer_review, merge_officer_reviews_into_df

# Page configuration
st.set_page_config(
    page_title="MPLADS Risk Intelligence & Duplicate Work Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS Styling (Clean, Minimal, Modern Professional Analytics)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    .main {
        background-color: #0b0f19;
        color: #f8fafc;
    }
    
    /* Enterprise Header Bar */
    .app-header {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 24px;
    }
    
    .app-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .app-subtitle {
        font-size: 0.88rem;
        color: #94a3b8;
        margin-top: 4px;
        margin-bottom: 12px;
    }
    
    /* Tag Pills */
    .gov-tag {
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid #334155;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Enterprise Metric / KPI Cards */
    .kpi-container {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px 20px;
        height: 100%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .kpi-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.2;
    }
    .kpi-subtext {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Banners & Panels */
    .principle-banner {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-left: 4px solid #38bdf8;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 20px;
        color: #e2e8f0;
        font-size: 0.88rem;
    }
    
    .explanation-panel {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 20px;
        margin-top: 16px;
    }

    /* Risk Score Component Card & Progress Bar */
    .risk-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .risk-score-display {
        font-size: 2rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1;
    }
    .risk-bar-container {
        background: #0f172a;
        border-radius: 4px;
        height: 10px;
        width: 100%;
        margin-top: 10px;
        margin-bottom: 8px;
        overflow: hidden;
        border: 1px solid #334155;
    }
    .risk-bar-fill-critical { background: #ef4444; height: 100%; }
    .risk-bar-fill-high { background: #f97316; height: 100%; }
    .risk-bar-fill-medium { background: #f59e0b; height: 100%; }
    .risk-bar-fill-low { background: #10b981; height: 100%; }
    
    /* Risk Reason Chips / Pills */
    .reason-chip {
        display: inline-block;
        background: #0f172a;
        color: #e2e8f0;
        border: 1px solid #334155;
        border-left: 3.5px solid #f97316;
        border-radius: 6px;
        padding: 7px 12px;
        font-size: 0.84rem;
        font-weight: 500;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .reason-chip-critical {
        border-left-color: #ef4444;
    }

    /* Risk Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .badge-critical { background: #7f1d1d; color: #fca5a5; border: 1px solid #991b1b; }
    .badge-high { background: #7c2d12; color: #fdba74; border: 1px solid #9a3412; }
    .badge-medium { background: #78350f; color: #fde047; border: 1px solid #92400e; }
    .badge-low { background: #064e3b; color: #6ee7b7; border: 1px solid #065f46; }
    .badge-info { background: #0c4a6e; color: #7dd3fc; border: 1px solid #075985; }

    /* Duplicate Record Match Cards */
    .dup-card {
        background: #0f172a;
        border: 1px solid #334155;
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 14px;
    }
    .dup-card-confirmed {
        border-left-color: #ef4444;
    }

    /* Custom Streamlit Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: transparent;
        border-radius: 6px 6px 0 0;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0 16px;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b;
        color: #38bdf8;
        border-color: #334155 #334155 transparent #334155;
    }

    /* Standardized Empty State */
    .empty-state {
        background: #0f172a;
        border: 1px dashed #334155;
        border-radius: 8px;
        padding: 32px;
        text-align: center;
        margin: 16px 0;
    }
    .empty-state-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #cbd5e1;
        margin-bottom: 4px;
    }
    .empty-state-desc {
        font-size: 0.82rem;
        color: #64748b;
    }
    
    /* Streamlit Metric Overrides */
    div[data-testid="stMetric"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 12px 16px;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc;
        font-size: 1.4rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

def render_empty_state(title="No Records Found", message="No work items match the selected filter criteria. Try adjusting the filter parameters in the control panel."):
    """Standardized professional empty-state renderer."""
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-state-title">{title}</div>
        <div class="empty-state-desc">{message}</div>
    </div>
    """, unsafe_allow_html=True)

# Plotly Theme Setup
PLOTLY_THEME = "plotly_dark"
PLOTLY_COLOR_DISCRETE = {'CRITICAL': '#ef4444', 'HIGH': '#f97316', 'MEDIUM': '#f59e0b', 'LOW': '#10b981'}

def style_plotly_chart(fig, height=360):
    fig.update_layout(
        template=PLOTLY_THEME,
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
        margin=dict(l=20, r=20, t=30, b=30),
        xaxis=dict(gridcolor='rgba(255, 255, 255, 0.05)', title_font=dict(size=11)),
        yaxis=dict(gridcolor='rgba(255, 255, 255, 0.05)', title_font=dict(size=11)),
        legend=dict(font=dict(size=11))
    )
    return fig

# Instant Data Loading from precomputed pickle bundle
@st.cache_data(ttl=86400)
def load_processed_data():
    bundle_path = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'pipeline_bundle.pkl')
    if os.path.exists(bundle_path):
        try:
            with open(bundle_path, 'rb') as f:
                b = pickle.load(f)
                return (
                    b['states_df'], b['state_tiles_df'], b['mps_df'], 
                    b['works_df'], b.get('duplicates_matrix', pd.DataFrame()),
                    b.get('alerts_df', pd.DataFrame()), 
                    b.get('early_warnings_df', pd.DataFrame()),
                    b.get('compliance_summary_df', pd.DataFrame()),
                    b.get('audit_summary', {}), b.get('missing_df', pd.DataFrame()), b.get('metadata', {})
                )
        except Exception as e:
            st.warning(f"Note: Precomputed bundle could not be unpickled ({e}). Recomputing pipeline...")

    # Fallback compute if bundle is missing
    states_df, state_tiles_df, mps_df, raw_works_df, metadata = build_mplads_dataset()
    cleaned_df = clean_works_df(raw_works_df)
    fe_df = engineer_mplads_features(cleaned_df)
    anom_df = detect_anomalies(fe_df)
    dup_df, duplicates_matrix = detect_duplicate_works(anom_df)
    risk_df = compute_risk_scoring(dup_df)
    forecasted_df, early_warnings_df = compute_predictive_early_warnings(risk_df)
    compliant_df, compliance_summary_df = audit_mplads_compliance(forecasted_df)
    alerts_df = generate_risk_alerts(compliant_df)
    audit_summary, missing_df = audit_data_quality(raw_works_df)
    return states_df, state_tiles_df, mps_df, compliant_df, duplicates_matrix, alerts_df, early_warnings_df, compliance_summary_df, audit_summary, missing_df, metadata

states_df, state_tiles_df, mps_df, raw_works_df, duplicates_matrix, alerts_df, early_warnings_df, compliance_summary_df, audit_summary, missing_df, metadata = load_processed_data()

# Merge persisted officer reviews into works_df
works_df = merge_officer_reviews_into_df(raw_works_df)

# Header Banner
st.markdown("""
<div class="app-header">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
        <div>
            <h1 class="app-title">MPLADS Risk Intelligence & Duplicate Work Detection System</h1>
            <div class="app-subtitle">Ministry of Statistics and Programme Implementation (MoSPI) &mdash; Forensic Analytics Platform</div>
            <div style="margin-top: 8px;">
                <span class="gov-tag" style="border-color: #ef4444; color: #fca5a5; font-weight: 700;">1. Risk Scoring (0-100)</span>
                <span class="gov-tag" style="border-color: #f97316; color: #fdba74; font-weight: 700;">2. Risk Reasons Diagnosis</span>
                <span class="gov-tag" style="border-color: #38bdf8; color: #38bdf8; font-weight: 700;">3. Duplicate & Cross-Scheme Detection</span>
                <span class="gov-tag">Officer Audit Workflow</span>
            </div>
        </div>
        <div style="background: #1e293b; padding: 12px 16px; border-radius: 6px; border: 1px solid #334155; min-width: 220px;">
            <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">System Data Integration</div>
            <div style="font-size: 0.88rem; font-weight: 700; color: #38bdf8; margin-top: 2px;">mplads.mospi.gov.in REST API</div>
            <div style="font-size: 0.75rem; color: #10b981; margin-top: 4px; font-weight: 600;">Connected (Public Aggregates)</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Control Center
st.sidebar.markdown("### Control Center")

user_role = st.sidebar.selectbox(
    "Dashboard User Role",
    ["Ministry / MoSPI", "State Nodal Authority", "District Authority", "Member of Parliament"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Filter Criteria")

# Session State Filter Defaults
if 'state_filter' not in st.session_state: st.session_state.state_filter = 'All States'
if 'district_filter' not in st.session_state: st.session_state.district_filter = 'All Districts'
if 'mp_filter' not in st.session_state: st.session_state.mp_filter = 'All MPs'
if 'tenure_filter' not in st.session_state: st.session_state.tenure_filter = 'All Tenures'
if 'cat_filter' not in st.session_state: st.session_state.cat_filter = 'All Categories'
if 'status_filter' not in st.session_state: st.session_state.status_filter = 'All Statuses'
if 'risk_filter' not in st.session_state: st.session_state.risk_filter = 'All Risk Levels'
if 'rev_status_filter' not in st.session_state: st.session_state.rev_status_filter = 'All Review Statuses'

def reset_all_filters():
    st.session_state.state_filter = 'All States'
    st.session_state.district_filter = 'All Districts'
    st.session_state.mp_filter = 'All MPs'
    st.session_state.tenure_filter = 'All Tenures'
    st.session_state.cat_filter = 'All Categories'
    st.session_state.status_filter = 'All Statuses'
    st.session_state.risk_filter = 'All Risk Levels'
    st.session_state.rev_status_filter = 'All Review Statuses'

# State Filter
state_list = ['All States'] + sorted(list(works_df['STATE_NAME'].unique()))
selected_state = st.sidebar.selectbox("State / Union Territory", state_list, key='state_filter')

# Cascading District Filter
if selected_state != 'All States':
    dist_options = sorted(list(works_df[works_df['STATE_NAME'] == selected_state]['DISTRICT_NAME'].unique()))
    dist_list = ['All Districts'] + dist_options
    selected_district = st.sidebar.selectbox("District", dist_list, key='district_filter')
else:
    st.sidebar.selectbox("District", ["Select State First"], disabled=True)
    selected_district = 'All Districts'

# MP Filter
if selected_state != 'All States':
    mp_list = ['All MPs'] + sorted(list(works_df[works_df['STATE_NAME'] == selected_state]['MP_NAME'].unique()))
else:
    mp_list = ['All MPs'] + sorted(list(works_df['MP_NAME'].unique()))
selected_mp = st.sidebar.selectbox("Member of Parliament (MP)", mp_list, key='mp_filter')

# Tenure Filter
tenure_list = ['All Tenures'] + sorted(list(works_df['TENURE'].unique()))
selected_tenure = st.sidebar.selectbox("Lok Sabha / Tenure", tenure_list, key='tenure_filter')

# Category Filter
cat_list = ['All Categories'] + sorted(list(works_df['WORK_CATEGORY'].unique()))
selected_cat = st.sidebar.selectbox("Work Category", cat_list, key='cat_filter')

# Status Filter
status_list = ['All Statuses'] + sorted(list(works_df['WORK_STATUS'].unique()))
selected_status = st.sidebar.selectbox("Implementation Status", status_list, key='status_filter')

# Risk Level Filter
risk_list = ['All Risk Levels', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
selected_risk = st.sidebar.selectbox("Risk Severity Level", risk_list, key='risk_filter')

# Officer Review Status Filter
rev_status_list = ['All Review Statuses', 'Pending Review', 'Verified', 'Needs Investigation', 'False Positive']
selected_rev_status = st.sidebar.selectbox("Officer Review Status", rev_status_list, key='rev_status_filter')

st.sidebar.button("Reset Filters", on_click=reset_all_filters, use_container_width=True)

# Apply Filters
filtered_df = filter_dataset(
    works_df,
    state=selected_state,
    district=selected_district,
    mp=selected_mp,
    tenure=selected_tenure,
    category=selected_cat,
    status=selected_status,
    risk_level=selected_risk,
    review_status=selected_rev_status
)

def render_active_filter_banner():
    active_filters = []
    if selected_state != 'All States': active_filters.append(f"State: **{selected_state}**")
    if selected_district != 'All Districts': active_filters.append(f"District: **{selected_district}**")
    if selected_mp != 'All MPs': active_filters.append(f"MP: **{selected_mp}**")
    if selected_tenure != 'All Tenures': active_filters.append(f"Tenure: **{selected_tenure}**")
    if selected_cat != 'All Categories': active_filters.append(f"Category: **{selected_cat}**")
    if selected_status != 'All Statuses': active_filters.append(f"Status: **{selected_status}**")
    if selected_risk != 'All Risk Levels': active_filters.append(f"Risk Level: **{selected_risk}**")
    if selected_rev_status != 'All Review Statuses': active_filters.append(f"Review: **{selected_rev_status}**")
    
    scope_str = " | ".join(active_filters) if active_filters else "All Data Scope (No Active Filters)"
    st.markdown(f"""
    <div style="font-size: 0.8rem; color: #38bdf8; background: #0f172a; padding: 6px 14px; border-radius: 6px; border: 1px solid #1e293b; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
        <div><strong>Active Selection Scope:</strong> {scope_str}</div>
        <div style="color: #94a3b8; font-size: 0.78rem;"><strong>{len(filtered_df):,}</strong> records matched</div>
    </div>
    """, unsafe_allow_html=True)

# Compute Filtered KPIs
kpis = compute_kpis(filtered_df)

# Calculate total duplicate count (intra-district + cross-scheme)
filtered_work_ids = set(filtered_df['WORK_ID'].unique()) if not filtered_df.empty else set()
intra_dup_count = len(duplicates_matrix[
    duplicates_matrix['Work A ID'].isin(filtered_work_ids) & 
    duplicates_matrix['Work B ID'].isin(filtered_work_ids)
]) if (duplicates_matrix is not None and not duplicates_matrix.empty and filtered_work_ids) else 0

cs_dup_count = int(filtered_df['IS_CROSS_SCHEME_DUPLICATE'].sum()) if 'IS_CROSS_SCHEME_DUPLICATE' in filtered_df.columns else 0
total_duplicate_flags = intra_dup_count + cs_dup_count

# Reorganized Navigation Tabs Structure (Highlights Core Features)
tab_overview, tab_risk_mon, tab_dup_detect, tab_work_detail, tab_officer_rev, tab_advanced = st.tabs([
    "Overview Dashboard",
    "Risk Score & Reasons",
    "Duplicate & Cross-Scheme Detection",
    "Work Record Inspector",
    "Officer Verification Portal",
    "Architecture & Analytics"
])

# ==========================================
# TAB 1: OVERVIEW DASHBOARD (Executive Risk Flow)
# ==========================================
with tab_overview:
    st.markdown(f"### Risk & Intelligence Overview &mdash; {user_role} Scope")
    render_active_filter_banner()
    
    st.markdown("""
    <div class="principle-banner">
        <strong>Governance Principle:</strong> Automated risk scores and detection flags assist in identifying financial and administrative outliers. Official verification and audit decisions remain with designated authority officers.
    </div>
    """, unsafe_allow_html=True)
    
    if filtered_df.empty:
        render_empty_state("No Records Found", "No work records match the selected filter parameters. Adjust your selections in the sidebar control panel.")
    else:
        # PROMINENT SUMMARY CARDS (Top 5 Essential Metrics)
        c1, c2, c3, c4, c5 = st.columns(5)
        
        with c1:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Total Records</div>
                <div class="kpi-value">{kpis['total_works']:,}</div>
                <div class="kpi-subtext">Monitored Public Works</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="kpi-container" style="border-left: 3.5px solid #f97316;">
                <div class="kpi-title" style="color: #fdba74;">High-Risk Records</div>
                <div class="kpi-value" style="color: #fdba74;">{kpis['high_risk_works']:,}</div>
                <div class="kpi-subtext">Score 55.0 &ndash; 74.9</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            st.markdown(f"""
            <div class="kpi-container" style="border-left: 3.5px solid #ef4444;">
                <div class="kpi-title" style="color: #fca5a5;">Critical-Risk Records</div>
                <div class="kpi-value" style="color: #fca5a5;">{kpis['critical_risk_works']:,}</div>
                <div class="kpi-subtext">Score &ge; 75.0 (Action Required)</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="kpi-container" style="border-left: 3.5px solid #38bdf8;">
                <div class="kpi-title" style="color: #38bdf8;">Potential Duplicates</div>
                <div class="kpi-value" style="color: #38bdf8;">{total_duplicate_flags:,}</div>
                <div class="kpi-subtext">Intra-District & Cross-Scheme</div>
            </div>
            """, unsafe_allow_html=True)

        with c5:
            st.markdown(f"""
            <div class="kpi-container" style="border-left: 3.5px solid #f59e0b;">
                <div class="kpi-title" style="color: #fde047;">Detected Anomalies</div>
                <div class="kpi-value" style="color: #fde047;">{kpis['anomalous_works']:,}</div>
                <div class="kpi-subtext">Multivariate Outlier Flags</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Overview Section 1: Financial & Severity Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("##### State Financial Allocation vs Expenditure")
            state_agg = aggregate_by_state(filtered_df).head(12)
            if not state_agg.empty:
                fig_state = go.Figure()
                fig_state.add_trace(go.Bar(x=state_agg['STATE_NAME'], y=state_agg['Total_Sanctioned']/1e7, name='Sanctioned (Cr)', marker_color='#2563eb'))
                fig_state.add_trace(go.Bar(x=state_agg['STATE_NAME'], y=state_agg['Total_Expenditure']/1e7, name='Expenditure (Cr)', marker_color='#10b981'))
                fig_state.update_layout(barmode='group', xaxis_tickangle=-45, yaxis_title="Amount (₹ Crore)")
                st.plotly_chart(style_plotly_chart(fig_state, 320), use_container_width=True)
            else:
                render_empty_state("No Data", "Insufficient data for state financial comparison.")

        with col_chart2:
            st.markdown("##### Risk Level Severity Distribution")
            if 'RISK_LEVEL' in filtered_df.columns and not filtered_df.empty:
                risk_counts = filtered_df['RISK_LEVEL'].value_counts().reset_index()
                risk_counts.columns = ['Risk Level', 'Count']
                fig_risk = px.pie(
                    risk_counts, values='Count', names='Risk Level', 
                    color='Risk Level', color_discrete_map=PLOTLY_COLOR_DISCRETE,
                    hole=0.45
                )
                st.plotly_chart(style_plotly_chart(fig_risk, 320), use_container_width=True)
            else:
                render_empty_state("No Data", "No risk level classification available.")

        # Overview Section 2: Primary Risk Flag Reasons Distribution
        st.markdown("##### Primary Administrative Risk Reasons across Active Selection")
        driver_counts = {
            'Budget Overrun': int((filtered_df['EXPENDITURE_AMOUNT'] > filtered_df['SANCTION_AMOUNT']).sum()),
            'High Spending vs Low Progress': int(((filtered_df['UTILIZATION_PCT'] > 80.0) & (filtered_df['PROGRESS_PERCENTAGE'] < 50.0)).sum()),
            'Unusual Spending Pattern': int(filtered_df['IS_ANOMALY'].sum()) if 'IS_ANOMALY' in filtered_df.columns else 0,
            'Possible Duplicate Project': int(filtered_df['IS_DUPLICATE_FLAG'].sum()) if 'IS_DUPLICATE_FLAG' in filtered_df.columns else 0,
            'Double-Funding Alert (Cross-Scheme)': int(filtered_df['IS_CROSS_SCHEME_DUPLICATE'].sum()) if 'IS_CROSS_SCHEME_DUPLICATE' in filtered_df.columns else 0,
            'Potential Tender Splitting': int(filtered_df['IS_SPLIT_TENDER'].sum()) if 'IS_SPLIT_TENDER' in filtered_df.columns else 0,
            'Funds Disbursed vs Work Stalled': int(filtered_df['IS_STAGNANT_SCURVE'].sum()) if 'IS_STAGNANT_SCURVE' in filtered_df.columns else 0,
        }
        drivers_df = pd.DataFrame(list(driver_counts.items()), columns=['Risk Flag Reason', 'Count']).sort_values(by='Count', ascending=True)
        fig_drivers = px.bar(
            drivers_df, x='Count', y='Risk Flag Reason', orientation='h',
            color='Count', color_continuous_scale='Reds',
            labels={'Count': 'Number of Flagged Works', 'Risk Flag Reason': 'Administrative Risk Reason'}
        )
        st.plotly_chart(style_plotly_chart(fig_drivers, 300), use_container_width=True)

        st.markdown("---")
        
        # High-Risk Priority Watchlist Table
        st.markdown("##### High & Critical Risk Priority Watchlist")
        st.markdown("Top priority work records sorted by **Risk Score (Highest First)** requiring administrative audit attention.")
        
        high_risk_df = filtered_df[filtered_df['RISK_LEVEL'].isin(['CRITICAL', 'HIGH'])].sort_values(by='RISK_SCORE', ascending=False)
        if not high_risk_df.empty:
            display_watchlist = high_risk_df[['WORK_ID', 'STATE_NAME', 'DISTRICT_NAME', 'WORK_CATEGORY', 'SANCTION_AMOUNT', 'EXPENDITURE_AMOUNT', 'PROGRESS_PERCENTAGE', 'RISK_SCORE', 'RISK_LEVEL', 'RISK_FACTORS', 'REVIEW_STATUS']].head(15).rename(columns={
                'WORK_ID': 'Work ID',
                'STATE_NAME': 'State',
                'DISTRICT_NAME': 'District',
                'WORK_CATEGORY': 'Work Category',
                'SANCTION_AMOUNT': 'Sanctioned Budget',
                'EXPENDITURE_AMOUNT': 'Expenditure',
                'PROGRESS_PERCENTAGE': 'Progress (%)',
                'RISK_SCORE': 'Risk Score',
                'RISK_LEVEL': 'Risk Level',
                'RISK_FACTORS': 'Risk Reasons & Factors',
                'REVIEW_STATUS': 'Audit Status'
            })
            st.dataframe(
                display_watchlist.style.format({
                    'Sanctioned Budget': lambda x: format_inr(x),
                    'Expenditure': lambda x: format_inr(x),
                    'Progress (%)': '{:.1f}%',
                    'Risk Score': '{:.1f}'
                }),
                use_container_width=True,
                height=380
            )
        else:
            st.success("No high or critical risk records in active selection scope.")

# ==========================================
# TAB 2: RISK SCORE & REASONS MONITORING
# ==========================================
with tab_risk_mon:
    st.markdown("### Risk Score & Reason Diagnostic Center")
    render_active_filter_banner()
    st.markdown("""
    Displays all monitored public work records ordered by **Risk Score (Highest First)**. Synthesizes budget overrun factors, progress discrepancies, project duplication flags, and policy compliance into clear administrative reasons.
    """)
    
    if filtered_df.empty:
        render_empty_state("No Records Found", "No work items match the active selection criteria.")
    else:
        risk_mon_df = filtered_df.sort_values(by='RISK_SCORE', ascending=False).copy()
        
        def get_all_risk_reasons(row):
            factors = row.get('RISK_FACTORS', '')
            if pd.notna(factors) and factors != 'Standard implementation profile':
                return str(factors)
            anom = row.get('ANOMALY_REASON', '')
            if pd.notna(anom) and anom != 'Normal execution pattern':
                return str(anom)
            return "Standard execution baseline parameters"

        risk_mon_df['ALL_RISK_REASONS'] = risk_mon_df.apply(get_all_risk_reasons, axis=1)
        risk_mon_df['DELAY_INFORMATION'] = risk_mon_df.apply(
            lambda r: f"Delayed ({r.get('PROJECTED_DELAY_MONTHS', 0.0)} mo)" if r.get('WORK_STATUS') == 'Delayed' else "On Schedule",
            axis=1
        )

        display_cols = [
            'WORK_ID', 'DISTRICT_NAME', 'WORK_CATEGORY', 
            'SANCTION_AMOUNT', 'EXPENDITURE_AMOUNT', 'PROGRESS_PERCENTAGE', 
            'DELAY_INFORMATION', 'RISK_SCORE', 'RISK_LEVEL', 
            'ALL_RISK_REASONS', 'REVIEW_STATUS'
        ]

        display_df = risk_mon_df[display_cols].rename(columns={
            'WORK_ID': 'Work ID',
            'DISTRICT_NAME': 'District',
            'WORK_CATEGORY': 'Work Category',
            'SANCTION_AMOUNT': 'Sanctioned Budget',
            'EXPENDITURE_AMOUNT': 'Expenditure',
            'PROGRESS_PERCENTAGE': 'Progress (%)',
            'DELAY_INFORMATION': 'Delay Info',
            'RISK_SCORE': 'Risk Score (0-100)',
            'RISK_LEVEL': 'Risk Level',
            'ALL_RISK_REASONS': 'Why is this risky? (Administrative Reasons)',
            'REVIEW_STATUS': 'Audit Status'
        })

        st.dataframe(
            display_df.style.format({
                'Sanctioned Budget': lambda x: format_inr(x),
                'Expenditure': lambda x: format_inr(x),
                'Progress (%)': '{:.1f}%',
                'Risk Score (0-100)': '{:.1f}'
            }),
            use_container_width=True,
            height=460
        )

        st.markdown("---")
        st.markdown("#### 🔍 Interactive Risk Score & Reason Inspector")
        st.markdown("Select any work item below to view its exact score formula additions and itemized risk reason chips.")
        
        selected_mon_id = st.selectbox(
            "Select Work ID for Detailed Inspection",
            risk_mon_df['WORK_ID'].tolist(),
            key="sb_risk_mon_inspector"
        )
        if selected_mon_id:
            mon_row = risk_mon_df[risk_mon_df['WORK_ID'] == selected_mon_id].iloc[0]
            mon_diag = generate_work_explanation(mon_row)
            
            c_ins1, c_ins2 = st.columns([1, 1.2])
            with c_ins1:
                r_sc = mon_diag['risk_score']
                r_lvl = mon_diag['risk_level']
                b_cls = "badge-critical" if r_lvl == 'CRITICAL' else ("badge-high" if r_lvl == 'HIGH' else ("badge-medium" if r_lvl == 'MEDIUM' else "badge-low"))
                bar_cls = "risk-bar-fill-critical" if r_lvl == 'CRITICAL' else ("risk-bar-fill-high" if r_lvl == 'HIGH' else ("risk-bar-fill-medium" if r_lvl == 'MEDIUM' else "risk-bar-fill-low"))

                st.markdown(f"""
                <div class="risk-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">PROMINENT RISK SCORE</div>
                            <div class="risk-score-display">{r_sc} <span style="font-size: 1rem; color: #94a3b8;">/ 100</span></div>
                        </div>
                        <div>
                            <span class="badge {b_cls}">{r_lvl} RISK</span>
                        </div>
                    </div>
                    <div class="risk-bar-container">
                        <div class="{bar_cls}" style="width: {min(100, max(5, r_sc))}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.write(f"**Work ID:** `{selected_mon_id}`")
                st.write(f"**Location:** {mon_diag['district']}, {mon_diag['state']}")
                st.write(f"**MP / Constituency:** {mon_diag['mp_name']} ({mon_diag['constituency']})")
                st.write(f"**Category:** {mon_diag['category']}")
                st.write(f"**Sanctioned Budget:** {mon_diag['sanctioned_amount_str']}")
                st.write(f"**Actual Expenditure:** {mon_diag['expenditure_amount_str']}")
                st.write(f"**Physical Progress:** {mon_diag['progress_pct_str']}")
                st.write(f"**Status:** `{mon_diag['status']}`")
                
            with c_ins2:
                st.markdown("##### Why is this record risky?")
                st.markdown("<div style='margin-bottom: 12px;'>", unsafe_allow_html=True)
                for r_item in mon_diag['reasons']:
                    clean_r = r_item.split('. ', 1)[-1] if '. ' in r_item else r_item
                    c_chip_cls = "reason-chip-critical" if r_lvl in ['HIGH', 'CRITICAL'] else ""
                    st.markdown(f"<div class='reason-chip {c_chip_cls}'>⚠️ {clean_r}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("**Exact Risk Score Formula Point Additions:**")
                for item in mon_diag['score_breakdown']:
                    st.markdown(f"&bull; **{item['factor']}**: `{item['pts']}`")
                st.info(f"**Recommended Action:** {mon_diag['recommended_action']}")

# ==========================================
# TAB 3: DUPLICATE & CROSS-SCHEME DETECTION
# ==========================================
with tab_dup_detect:
    st.markdown("### Duplicate & Cross-Scheme Detection Center")
    render_active_filter_banner()
    st.markdown("""
    Identifies **confirmed duplicate projects** within MPLADS (text & financial similarity) and **potential cross-scheme double-funding overlaps** with parallel national programs (PMGSY, Jal Jeevan Mission, etc.) based on spatial GIS coordinates.
    """)

    # Filter duplicate records
    filtered_work_ids = set(filtered_df['WORK_ID'].unique()) if not filtered_df.empty else set()
    
    # 1. Intra-District Duplicates (Confirmed TF-IDF Matches)
    intra_dups = duplicates_matrix[
        duplicates_matrix['Work A ID'].isin(filtered_work_ids) & 
        duplicates_matrix['Work B ID'].isin(filtered_work_ids)
    ] if (duplicates_matrix is not None and not duplicates_matrix.empty and filtered_work_ids) else pd.DataFrame()

    # 2. Cross-Scheme Overlaps (Potential GIS Double-Funding Matches)
    cs_dups = filtered_df[filtered_df['IS_CROSS_SCHEME_DUPLICATE'] == True] if 'IS_CROSS_SCHEME_DUPLICATE' in filtered_df.columns else pd.DataFrame()

    # Summary Metrics for Duplicate Detection
    c_d1, c_d2, c_d3, c_d4 = st.columns(4)
    c_d1.metric("Confirmed Intra-District Pairs", f"{len(intra_dups):,}")
    c_d2.metric("Cross-Scheme Spatial Overlaps", f"{len(cs_dups):,}")
    c_d3.metric("Total Duplicate Risk Flags", f"{len(intra_dups) + len(cs_dups):,}")
    total_val_at_risk = (cs_dups['SANCTION_AMOUNT'].sum() if not cs_dups.empty else 0) + (intra_dups['Sanction Amount A'].sum() if not intra_dups.empty else 0)
    c_d4.metric("Total Value at Risk", format_inr(total_val_at_risk))

    st.markdown("---")

    dup_view_mode = st.radio(
        "Duplicate Category Filter",
        ["All Duplicate Matches", "Confirmed Intra-District Duplicates", "Potential Cross-Scheme Overlaps"],
        horizontal=True
    )

    if dup_view_mode in ["All Duplicate Matches", "Confirmed Intra-District Duplicates"]:
        st.markdown("#### Confirmed Intra-District Duplicate Work Pairs")
        if not intra_dups.empty:
            for _, d_row in intra_dups.head(8).iterrows():
                sim_pct = round(d_row.get('Cosine Similarity', 0.85) * 100, 1)
                st.markdown(f"""
                <div class="dup-card dup-card-confirmed">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div>
                            <span class="badge badge-critical">CONFIRMED DUPLICATE MATCH ({sim_pct}% Similarity)</span>
                            <span style="font-size: 0.85rem; color: #94a3b8; margin-left: 10px;">District: {d_row.get('District', 'N/A')}</span>
                        </div>
                        <span style="font-size: 0.9rem; font-weight: 700; color: #ef4444;">Value: {format_inr(d_row.get('Sanction Amount A', 0))}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 10px; font-size: 0.85rem;">
                        <div style="background: #1e293b; padding: 10px 12px; border-radius: 6px; border: 1px solid #334155;">
                            <div style="color: #38bdf8; font-weight: 700;">Record A ID: {d_row['Work A ID']}</div>
                            <div style="color: #f8fafc; margin-top: 4px;">{d_row.get('Work Description A', 'N/A')}</div>
                            <div style="color: #94a3b8; margin-top: 4px;">Sanctioned Budget: {format_inr(d_row.get('Sanction Amount A', 0))}</div>
                        </div>
                        <div style="background: #1e293b; padding: 10px 12px; border-radius: 6px; border: 1px solid #334155;">
                            <div style="color: #38bdf8; font-weight: 700;">Record B ID: {d_row['Work B ID']}</div>
                            <div style="color: #f8fafc; margin-top: 4px;">{d_row.get('Work Description B', 'N/A')}</div>
                            <div style="color: #94a3b8; margin-top: 4px;">Sanctioned Budget: {format_inr(d_row.get('Sanction Amount B', 0))}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.dataframe(
                intra_dups.style.format({
                    'Sanction Amount A': lambda x: format_inr(x),
                    'Sanction Amount B': lambda x: format_inr(x)
                }),
                use_container_width=True
            )
        else:
            st.info("No confirmed intra-district duplicate work pairs in active selection.")

    if dup_view_mode in ["All Duplicate Matches", "Potential Cross-Scheme Overlaps"]:
        st.markdown("#### Potential Cross-Scheme Double-Funding Overlaps")
        if not cs_dups.empty:
            for _, cs_row in cs_dups.head(8).iterrows():
                dist_m = cs_row.get('CROSS_SCHEME_DISTANCE_METERS', 0.0)
                matched_scheme = cs_row.get('CROSS_SCHEME_MATCH_NAME', 'Parallel Scheme')
                matched_title = cs_row.get('CROSS_SCHEME_MATCH_TITLE', 'Parallel Asset Construction')
                st.markdown(f"""
                <div class="dup-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div>
                            <span class="badge badge-high">POTENTIAL DOUBLE-FUNDING OVERLAP ({dist_m}m Spatial Proximity)</span>
                            <span style="font-size: 0.85rem; color: #94a3b8; margin-left: 10px;">{cs_row.get('DISTRICT_NAME')}, {cs_row.get('STATE_NAME')}</span>
                        </div>
                        <span style="font-size: 0.9rem; font-weight: 700; color: #fdba74;">Sanctioned: {format_inr(cs_row.get('SANCTION_AMOUNT', 0))}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 10px; font-size: 0.85rem;">
                        <div style="background: #1e293b; padding: 10px 12px; border-radius: 6px; border: 1px solid #334155;">
                            <div style="color: #38bdf8; font-weight: 700;">MPLADS Record ID: {cs_row['WORK_ID']}</div>
                            <div style="color: #f8fafc; margin-top: 4px;">{cs_row.get('WORK_DESCRIPTION', 'N/A')}</div>
                            <div style="color: #94a3b8; margin-top: 4px;">Category: {cs_row.get('WORK_CATEGORY')}</div>
                        </div>
                        <div style="background: #1e293b; padding: 10px 12px; border-radius: 6px; border: 1px solid #334155;">
                            <div style="color: #a78bfa; font-weight: 700;">Matched Parallel Program: {matched_scheme}</div>
                            <div style="color: #f8fafc; margin-top: 4px;">{matched_title}</div>
                            <div style="color: #94a3b8; margin-top: 4px;">Spatial Distance: {dist_m} meters away</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.dataframe(
                cs_dups[['WORK_ID', 'STATE_NAME', 'DISTRICT_NAME', 'WORK_CATEGORY', 'SANCTION_AMOUNT', 'CROSS_SCHEME_MATCH_NAME', 'CROSS_SCHEME_DISTANCE_METERS', 'CROSS_SCHEME_MATCH_TITLE', 'RISK_SCORE']].style.format({
                    'SANCTION_AMOUNT': lambda x: format_inr(x),
                    'CROSS_SCHEME_DISTANCE_METERS': '{:.1f} m',
                    'RISK_SCORE': '{:.1f}'
                }),
                use_container_width=True
            )
        else:
            st.info("No cross-scheme spatial overlaps detected in active selection.")

# ==========================================
# TAB 4: WORK RECORD INSPECTOR (Strict Hierarchy)
# ==========================================
with tab_work_detail:
    st.markdown("### Itemized Work Record Inspection")
    render_active_filter_banner()
    
    work_id_options = filtered_df['WORK_ID'].tolist() if not filtered_df.empty else []
    if work_id_options:
        selected_work_id = st.selectbox("Select Target Record for Detailed Case Inspection", work_id_options)
        work_row = filtered_df[filtered_df['WORK_ID'] == selected_work_id].iloc[0]
        explanation = generate_work_explanation(work_row)
        
        # 1. RISK SCORE & 2. RISK LEVEL (Prominent Top Card & Bar)
        r_score = explanation['risk_score']
        r_level = explanation['risk_level']
        badge_cls = "badge-critical" if r_level == 'CRITICAL' else ("badge-high" if r_level == 'HIGH' else ("badge-medium" if r_level == 'MEDIUM' else "badge-low"))
        bar_cls = "risk-bar-fill-critical" if r_level == 'CRITICAL' else ("risk-bar-fill-high" if r_level == 'HIGH' else ("risk-bar-fill-medium" if r_level == 'MEDIUM' else "risk-bar-fill-low"))

        st.markdown(f"""
        <div class="risk-card">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <div style="font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">1. COMPOSITE RISK SCORE</div>
                    <div class="risk-score-display">{r_score} <span style="font-size: 1rem; color: #94a3b8;">/ 100</span></div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; text-align: right; margin-bottom: 4px;">2. RISK LEVEL SEVERITY</div>
                    <span class="badge {badge_cls}" style="font-size: 0.95rem; padding: 6px 14px;">{r_level} RISK</span>
                </div>
            </div>
            <div class="risk-bar-container">
                <div class="{bar_cls}" style="width: {min(100, max(5, r_score))}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 3. RISK REASONS ("Why is this risky?" Clean Chips Section)
        st.markdown("#### 3. Why is this record risky? (Risk Reasons)")
        
        st.markdown("<div style='margin-bottom: 12px;'>", unsafe_allow_html=True)
        for reason_str in explanation['reasons']:
            clean_reason = reason_str.split('. ', 1)[-1] if '. ' in reason_str else reason_str
            chip_class = "reason-chip-critical" if r_level in ['HIGH', 'CRITICAL'] else ""
            st.markdown(f"<div class='reason-chip {chip_class}'>⚠️ {clean_reason}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.info(f"**Recommended Action for Officers:** {explanation['recommended_action']}")

        # 4. DUPLICATE DETECTION / RELATED RECORDS
        st.markdown("#### 4. Duplicate Detection & Related Records")
        is_dup_flag = work_row.get('IS_DUPLICATE_FLAG', False)
        is_cs_flag = work_row.get('IS_CROSS_SCHEME_DUPLICATE', False)

        if is_dup_flag or is_cs_flag:
            if is_cs_flag:
                cs_name = work_row.get('CROSS_SCHEME_MATCH_NAME', 'Parallel Scheme')
                dist_m = work_row.get('CROSS_SCHEME_DISTANCE_METERS', 0.0)
                cs_title = work_row.get('CROSS_SCHEME_MATCH_TITLE', 'Parallel Asset Construction')
                st.markdown(f"""
                <div class="dup-card">
                    <div style="color: #fdba74; font-weight: 700; font-size: 0.9rem;">⚠️ Potential Cross-Scheme Double-Funding Overlap Flagged ({dist_m}m Distance)</div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 6px;">
                        Matched Parallel Program: <strong>{cs_name}</strong> &bull; Asset Title: <em>"{cs_title}"</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            if is_dup_flag:
                st.markdown(f"""
                <div class="dup-card dup-card-confirmed">
                    <div style="color: #fca5a5; font-weight: 700; font-size: 0.9rem;">⚠️ Confirmed Intra-District Duplicate Similarity Flagged</div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 6px;">
                        High text and financial similarity matched with another registered project in {explanation['district']} district.
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✓ No duplicate project matches or cross-scheme spatial overlaps detected for this record.")

        # 5. SUPPORTING DETAILS & METADATA
        st.markdown("#### 5. Supporting Record Details & Disbursement Schedule")
        col_sd1, col_sd2 = st.columns(2)
        with col_sd1:
            st.write(f"**Work Record ID:** `{explanation['work_id']}`")
            st.write(f"**Member of Parliament (MP):** {explanation['mp_name']}")
            st.write(f"**Constituency:** {explanation['constituency']}")
            st.write(f"**State / District:** {explanation['state']} / {explanation['district']}")
            st.write(f"**Work Category:** {explanation['category']}")
            st.write(f"**Work Scope:** {work_row.get('WORK_DESCRIPTION', 'N/A')}")
        with col_sd2:
            st.write(f"**Sanctioned Budget:** {explanation['sanctioned_amount_str']}")
            st.write(f"**Actual Expenditure:** {explanation['expenditure_amount_str']}")
            st.write(f"**Physical Progress:** {explanation['progress_pct_str']}")
            st.write(f"**Work Status:** `{explanation['status']}`")
            st.write(f"**Delay Assessment:** {explanation['delay_info']}")
            st.write(f"**Data Completeness:** `{explanation['data_status_badge']}`")

        st.markdown("---")
        st.markdown("##### Milestone Disbursement Schedule")
        sanc = float(work_row['SANCTION_AMOUNT']) if pd.notna(work_row.get('SANCTION_AMOUNT')) else 0.0
        prog = float(work_row['PROGRESS_PERCENTAGE']) if pd.notna(work_row.get('PROGRESS_PERCENTAGE')) else 0.0
        sanc_dt = str(work_row.get('SANCTION_DATE', 'N/A'))[:10]
        comp_dt = str(work_row.get('COMPLETION_DATE', 'N/A'))[:10]
        
        tranches = [
            {'Tranche': 'Tranche 1 (Commencement)', 'Benchmark': 'Administrative Sanction', 'Share': '20%', 'Amount': format_inr(sanc * 0.20), 'Date': sanc_dt, 'Status': 'Released'},
            {'Tranche': 'Tranche 2 (Interim)', 'Benchmark': '50% Physical Completion', 'Share': '50%', 'Amount': format_inr(sanc * 0.50), 'Date': 'Disbursed' if prog >= 50 else 'Pending', 'Status': 'Released' if prog >= 50 else 'Held Pending Progress'},
            {'Tranche': 'Tranche 3 (Final Release)', 'Benchmark': 'Final Inspection', 'Share': '30%', 'Amount': format_inr(sanc * 0.30), 'Date': comp_dt if work_row['WORK_STATUS'] == 'Completed' else 'Pending', 'Status': 'Released' if work_row['WORK_STATUS'] == 'Completed' else 'Held Pending Completion'}
        ]
        st.dataframe(pd.DataFrame(tranches), use_container_width=True)
    else:
        render_empty_state("No Selection Available", "No work records available under current filter criteria.")

# ==========================================
# TAB 5: HUMAN-IN-THE-LOOP OFFICER REVIEW
# ==========================================
with tab_officer_rev:
    st.markdown("### Officer Verification Portal")
    render_active_filter_banner()
    
    st.markdown("""
    <div class="principle-banner">
        <strong>Verification Workflow:</strong> Risk scores indicate parameter variance relative to category baselines. Authorized authority verification is required before initiating formal inquiry or freezing fund releases.
    </div>
    """, unsafe_allow_html=True)
    
    r_c1, r_c2, r_c3, r_c4 = st.columns(4)
    r_c1.metric("Pending Review Queue", f"{kpis['pending_review_works']:,}")
    r_c2.metric("Verified Works", f"{kpis['verified_works']:,}")
    r_c3.metric("Needs Investigation", f"{kpis['needs_investigation_works']:,}")
    r_c4.metric("False Positives Marked", f"{kpis['false_positive_works']:,}")

    st.markdown("---")
    
    if filtered_df.empty:
        render_empty_state("Queue Empty", "No works match current criteria for officer review.")
    else:
        st.markdown("#### Review Execution Queue")
        
        review_queue_filter = st.radio(
            "Queue Filter Scope",
            ["Pending Review Only", "High & Critical Risk Works Only", "All Works in Filter Scope"],
            horizontal=True
        )
        
        if review_queue_filter == "Pending Review Only":
            queue_df = filtered_df[filtered_df['REVIEW_STATUS'] == 'Pending Review']
        elif review_queue_filter == "High & Critical Risk Works Only":
            queue_df = filtered_df[filtered_df['RISK_LEVEL'].isin(['HIGH', 'CRITICAL'])]
        else:
            queue_df = filtered_df
            
        queue_df = queue_df.sort_values(by='RISK_SCORE', ascending=False)
        
        if queue_df.empty:
            st.success("No pending items require verification under the selected queue scope.")
        else:
            queue_work_ids = queue_df['WORK_ID'].tolist()
            selected_rev_id = st.selectbox("Select Target Work ID for Action", queue_work_ids, key="sb_officer_queue")
            
            target_row = queue_df[queue_df['WORK_ID'] == selected_rev_id].iloc[0]
            exp_diag = generate_work_explanation(target_row)
            
            col_or1, col_or2 = st.columns([1.2, 1])
            
            with col_or1:
                st.markdown(f"##### Work Parameters: {selected_rev_id}")
                st.write(f"**State / District:** {target_row['STATE_NAME']} / {target_row['DISTRICT_NAME']}")
                st.write(f"**MP Name:** {target_row['MP_NAME']}")
                st.write(f"**Category:** {target_row['WORK_CATEGORY']}")
                st.write(f"**Sanctioned Amount:** {exp_diag['sanctioned_amount_str']}")
                st.write(f"**Expenditure Amount:** {exp_diag['expenditure_amount_str']}")
                st.write(f"**Physical Progress:** {exp_diag['progress_pct_str']}")
                st.write(f"**Implementation Status:** {target_row['WORK_STATUS']}")
                st.write(f"**Data Status:** {exp_diag['data_status_badge']}")
                
                st.markdown("##### Why is this record risky?")
                for r_item in exp_diag['reasons']:
                    clean_r = r_item.split('. ', 1)[-1] if '. ' in r_item else r_item
                    st.markdown(f"<div class='reason-chip'>⚠️ {clean_r}</div>", unsafe_allow_html=True)
                    
                st.info(f"Recommended Action: {exp_diag['recommended_action']}")
                
            with col_or2:
                st.markdown("##### Decision Form")
                st.write(f"**Reviewer Role:** `{user_role}`")
                
                curr_st = target_row.get('REVIEW_STATUS', 'Pending Review')
                curr_rem = target_row.get('OFFICER_REMARK', '')
                
                selected_status_val = st.radio(
                    "Set Administrative Status",
                    ["Pending Review", "Verified", "Needs Investigation", "False Positive"],
                    index=["Pending Review", "Verified", "Needs Investigation", "False Positive"].index(curr_st) if curr_st in ["Pending Review", "Verified", "Needs Investigation", "False Positive"] else 0,
                    key=f"radio_officer_{selected_rev_id}"
                )
                
                officer_remark_val = st.text_area(
                    "Inspection Remarks",
                    value=curr_rem,
                    placeholder="Enter audit observations, voucher inspection findings, or justification...",
                    height=120,
                    key=f"ta_officer_{selected_rev_id}"
                )
                
                if st.button("Submit Decision", key=f"btn_officer_{selected_rev_id}", type="primary"):
                    save_officer_review(selected_rev_id, selected_status_val, officer_remark_val, user_role)
                    st.success(f"Review status updated for Work ID '{selected_rev_id}' to '{selected_status_val}'.")
                    st.rerun()

        st.markdown("---")
        st.markdown("#### Verification Audit Trail")
        
        reviewed_items = filtered_df[filtered_df['REVIEW_STATUS'] != 'Pending Review']
        if not reviewed_items.empty:
            display_rev_log = reviewed_items[['WORK_ID', 'DISTRICT_NAME', 'WORK_CATEGORY', 'RISK_SCORE', 'RISK_LEVEL', 'REVIEW_STATUS', 'OFFICER_REMARK', 'REVIEW_TIMESTAMP', 'OFFICER_ROLE']].rename(columns={
                'WORK_ID': 'Work ID',
                'DISTRICT_NAME': 'District',
                'WORK_CATEGORY': 'Category',
                'RISK_SCORE': 'Risk Score',
                'RISK_LEVEL': 'Risk Level',
                'REVIEW_STATUS': 'Status',
                'OFFICER_REMARK': 'Remarks',
                'REVIEW_TIMESTAMP': 'Timestamp',
                'OFFICER_ROLE': 'Reviewing Authority'
            })
            st.dataframe(display_rev_log, use_container_width=True)
        else:
            st.info("No recorded verification decisions in active selection.")

# ==========================================
# TAB 6: DATA ARCHITECTURE & ADVANCED ANALYTICS
# ==========================================
with tab_advanced:
    st.markdown("### Data Architecture & Advanced Forensics")
    render_active_filter_banner()
    
    sub_t1, sub_t2, sub_t3, sub_t4 = st.tabs([
        "System Data Architecture",
        "Policy Guideline Compliance",
        "Predictive Early Warnings",
        "Progress Velocity & Burn-Rate"
    ])

    # SUB-TAB 1: ARCHITECTURE & FLOW
    with sub_t1:
        st.markdown("#### Target Data Pipeline Flowchart")
        st.graphviz_chart("""
        digraph {
            rankdir=LR;
            background="transparent";
            node [shape=box, style="filled,rounded", fillcolor="#1e293b", fontcolor="#f8fafc", fontname="Inter", color="#334155", fontsize=10];
            edge [color="#38bdf8", penwidth=1.5];
            
            A [label="MoSPI Public REST Ingestion Engine\n(mplads.mospi.gov.in)"];
            B [label="Data Cleaning &\nCurrency Normalizer"];
            C [label="Data Quality Validation\n& Null Audit"];
            D [label="Rule-Based Policy\nCompliance Engine"];
            E [label="Isolation Forest\nOutlier Detector"];
            F [label="Composite Risk Scoring\n(0-100 Score)"];
            G [label="Explainable AI\nDiagnostic Cards"];
            H [label="Human Officer\nVerification Portal"];
            I [label="Persisted Audit Log\n& Action Engine"];
            
            A -> B -> C -> D -> E -> F -> G -> H -> I;
        }
        """, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Security & Enterprise Infrastructure Controls")
        st.info("""
        🔒 **Enterprise Infrastructure Controls**:
        - **Deployment Profile**: MeitY-empanelled Cloud / NIC Infrastructure utilizing PostgreSQL enterprise relational storage.
        - **Data Protection**: Enforces HTTPS/TLS data transit encryption, session-gated Role-Based Access Control (RBAC), multi-factor authentication for verifying officers, and immutable transaction audit logging.
        """)

    # SUB-TAB 2: POLICY AUDIT
    with sub_t2:
        st.markdown("#### Scheme Policy & Guideline Compliance Audit")
        st.markdown("""
        Audits work records against official MoSPI MPLADS Guidelines, evaluating MP Tenure Entitlement Ceilings (₹25 Cr for 17th Lok Sabha / ₹10 Cr for 18th Lok Sabha) and administrative sanction completeness.
        """)

        filtered_work_ids = set(filtered_df['WORK_ID'].unique()) if not filtered_df.empty else set()
        filtered_comp = compliance_summary_df[compliance_summary_df['Work ID'].isin(filtered_work_ids)] if (compliance_summary_df is not None and not compliance_summary_df.empty) else pd.DataFrame()

        total_violations = len(filtered_comp)
        ceiling_breaches = len(filtered_comp[filtered_comp['Triggered Policy Rules'].str.contains('Fund Ceiling Breach', na=False)]) if not filtered_comp.empty else 0

        col_c1, col_c2, col_c3 = st.columns(3)
        col_c1.metric("Audited Works", f"{len(filtered_df):,}")
        col_c2.metric("Policy Guideline Flags", f"{total_violations:,}")
        col_c3.metric("Entitlement Ceiling Breaches", f"{ceiling_breaches:,}")

        if not filtered_comp.empty:
            st.markdown("##### Policy Audit Matrix")
            st.dataframe(
                filtered_comp[['Work ID', 'State', 'District', 'MP Name', 'Work Category', 'Sanction Amount', 'Violation Severity', 'Triggered Policy Rules']].style.format({
                    'Sanction Amount': lambda x: format_inr(x)
                }),
                use_container_width=True
            )
        else:
            st.success("All works in active selection comply with MoSPI MPLADS policy guidelines and entitlement ceilings.")

    # SUB-TAB 3: PREDICTIVE EARLY WARNINGS
    with sub_t3:
        st.markdown("#### Predictive Early-Warning Intelligence Engine")
        st.markdown("""
        Evaluates spend-to-progress velocity for ongoing works to project **future cost overruns** and **completion delays** prior to threshold breaches.
        """)
        
        ongoing_df = filtered_df[filtered_df['WORK_STATUS'].isin(['Ongoing', 'Sanctioned / Pending', 'Delayed', 'Incomplete with High Exp'])].copy()
        
        if not ongoing_df.empty:
            early_warn_count = len(ongoing_df[ongoing_df['IS_EARLY_WARNING'] == True]) if 'IS_EARLY_WARNING' in ongoing_df.columns else 0
            high_warn_count = len(ongoing_df[ongoing_df['EARLY_WARNING_LEVEL'] == 'HIGH RISK WARNING']) if 'EARLY_WARNING_LEVEL' in ongoing_df.columns else 0
            avg_overrun_pct = ongoing_df['PROJECTED_OVERRUN_PCT'].mean() if 'PROJECTED_OVERRUN_PCT' in ongoing_df.columns else 0.0
            
            c_ew1, c_ew2, c_ew3, c_ew4 = st.columns(4)
            c_ew1.metric("Ongoing Works Monitored", f"{len(ongoing_df):,}")
            c_ew2.metric("Early Warning Flags", f"{early_warn_count:,}")
            c_ew3.metric("High-Risk Trajectory Warnings", f"{high_warn_count:,}")
            c_ew4.metric("Avg Projected Cost Overrun", f"{avg_overrun_pct:.1f}%")

            st.markdown("---")
            
            if 'IS_EARLY_WARNING' in ongoing_df.columns and 'EARLY_WARNING_SCORE' in ongoing_df.columns:
                early_flagged = ongoing_df[ongoing_df['IS_EARLY_WARNING'] == True].sort_values(by='EARLY_WARNING_SCORE', ascending=False)
            elif 'IS_EARLY_WARNING' in ongoing_df.columns:
                early_flagged = ongoing_df[ongoing_df['IS_EARLY_WARNING'] == True]
            else:
                early_flagged = ongoing_df.iloc[0:0].copy()
            
            if not early_flagged.empty:
                display_ew = early_flagged[['WORK_ID', 'STATE_NAME', 'DISTRICT_NAME', 'WORK_CATEGORY', 'SANCTION_AMOUNT', 'EXPENDITURE_AMOUNT', 'PROGRESS_PERCENTAGE', 'PREDICTED_FINAL_COST', 'PROJECTED_OVERRUN_PCT', 'PROJECTED_DELAY_MONTHS', 'EARLY_WARNING_LEVEL', 'EARLY_WARNING_REASON']].copy()
                st.dataframe(
                    display_ew.style.format({
                        'SANCTION_AMOUNT': lambda x: format_inr(x),
                        'EXPENDITURE_AMOUNT': lambda x: format_inr(x),
                        'PREDICTED_FINAL_COST': lambda x: format_inr(x),
                        'PROGRESS_PERCENTAGE': '{:.1f}%',
                        'PROJECTED_OVERRUN_PCT': '+{:.1f}%',
                        'PROJECTED_DELAY_MONTHS': '{:.1f} mo'
                    }),
                    use_container_width=True
                )
            else:
                st.success("No ongoing works currently project cost overruns or completion delays above early warning thresholds.")
        else:
            render_empty_state("No Data", "No ongoing works match criteria for predictive early-warning analysis.")

    # SUB-TAB 4: BURN-RATE VELOCITY
    with sub_t4:
        st.markdown("#### Progress Milestone & Expenditure Velocity")
        st.markdown("""
        Evaluates physical milestone burn-rate velocity and financial expenditure lead variance across implementation works.
        """)
        
        if 'IS_STAGNANT_SCURVE' in filtered_df.columns:
            stagnant_df = filtered_df[filtered_df['IS_STAGNANT_SCURVE'] == True]
        else:
            stagnant_df = filtered_df.iloc[0:0].copy()
            
        c_sc1, c_sc2, c_sc3 = st.columns(3)
        c_sc1.metric("Audited Works Scope", f"{len(filtered_df):,}")
        c_sc2.metric("Stagnation Flags", f"{len(stagnant_df):,}")
        c_sc3.metric("Avg Physical Progress", f"{filtered_df['PROGRESS_PERCENTAGE'].mean():.1f}%" if not filtered_df.empty else "0.0%")
        
        if not stagnant_df.empty:
            st.warning("Milestone Velocity Lag: Works exhibiting high financial expenditure lead velocity relative to physical completion percentage.")
            display_sc = stagnant_df[['WORK_ID', 'STATE_NAME', 'DISTRICT_NAME', 'WORK_CATEGORY', 'SANCTION_AMOUNT', 'EXPENDITURE_AMOUNT', 'PROGRESS_PERCENTAGE', 'UTILIZATION_PCT', 'RISK_SCORE']].copy()
            st.dataframe(
                display_sc.style.format({
                    'SANCTION_AMOUNT': lambda x: format_inr(x),
                    'EXPENDITURE_AMOUNT': lambda x: format_inr(x),
                    'PROGRESS_PERCENTAGE': '{:.1f}%',
                    'UTILIZATION_PCT': '{:.1f}%',
                    'RISK_SCORE': '{:.1f}'
                }),
                use_container_width=True
            )
        else:
            st.success("Physical milestone progress aligns with financial expenditure burn rates across active selection.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.8rem; padding: 12px 0;">
    MPLADS Risk Intelligence & Duplicate Work Detection System &bull; Ministry of Statistics and Programme Implementation (MoSPI)
</div>
""", unsafe_allow_html=True)
