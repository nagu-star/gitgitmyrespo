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
from src.photo_geofence import audit_photo_geofence_and_hashes
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
    page_title="MPLADS Expenditure & Risk Monitoring Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS Styling (Clean, Minimal, Government Tech Aesthetics)
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
        font-size: 0.9rem;
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
    }
    .kpi-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.2;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 6px;
    }
    
    /* Banners & Cards */
    .principle-banner {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-left: 4px solid #38bdf8;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 20px;
        color: #e2e8f0;
        font-size: 0.9rem;
    }
    
    .explanation-panel {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 20px;
        margin-top: 16px;
    }
    
    /* Risk Badges */
    .badge {
        display: inline-block;
        padding: 3px 8px;
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
    photo_df = audit_photo_geofence_and_hashes(dup_df)
    risk_df = compute_risk_scoring(photo_df)
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
            <h1 class="app-title">MPLADS Expenditure & Risk Monitoring Platform</h1>
            <div class="app-subtitle">Ministry of Statistics and Programme Implementation (MoSPI) &mdash; Data Analytics & Audit Copilot</div>
            <div>
                <span class="gov-tag">Data Ingestion</span>
                <span class="gov-tag">Compliance Audit</span>
                <span class="gov-tag">Isolation Forest</span>
                <span class="gov-tag">XAI Diagnostics</span>
                <span class="gov-tag">Officer Verification</span>
            </div>
        </div>
        <div style="background: #1e293b; padding: 12px 16px; border-radius: 6px; border: 1px solid #334155; min-width: 220px;">
            <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">System Data Status</div>
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
        <div>🔍 <strong>Active Selection Scope:</strong> {scope_str}</div>
        <div style="color: #94a3b8; font-size: 0.78rem;"><strong>{len(filtered_df):,}</strong> works matched</div>
    </div>
    """, unsafe_allow_html=True)

# Compute Filtered KPIs
kpis = compute_kpis(filtered_df)

# Navigation Tabs Structure
tab_overview, tab_risk_mon, tab_work_detail, tab_officer_rev, tab_data_sources, tab_advanced = st.tabs([
    "Overview",
    "Risk Monitoring",
    "Work Inspection",
    "Officer Verification",
    "Data Architecture",
    "Advanced Analytics"
])

# ==========================================
# TAB 1: OVERVIEW DASHBOARD
# ==========================================
with tab_overview:
    st.markdown(f"### Monitoring Overview &mdash; {user_role} View")
    render_active_filter_banner()
    
    st.markdown("""
    <div class="principle-banner">
        <strong>Governance Principle:</strong> AI-based risk models assist in identifying outliers and parameter variances. Official verification and administrative decisions remain with designated authority officers.
    </div>
    """, unsafe_allow_html=True)
    
    if filtered_df.empty:
        render_empty_state("No Works Found", "No work records match the selected filter parameters. Adjust your selections in the sidebar control panel.")
    else:
        # Metrics Section 1: Work & Anomaly Totals
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Monitored Works", f"{kpis['total_works']:,}")
        c2.metric("High & Critical Risk Works", f"{kpis['high_and_critical_risk']:,}")
        c3.metric("Delayed Works", f"{kpis['delayed_works']:,}")
        c4.metric("Anomalous Works", f"{kpis['anomalous_works']:,}")

        # Metrics Section 2: Financial Utilization
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Total Sanctioned", format_inr(kpis['total_sanctioned']))
        c6.metric("Total Expenditure", format_inr(kpis['total_expenditure']), delta=f"{kpis['utilization_rate']}% Utilization")
        c7.metric("Remaining Unspent", format_inr(kpis['remaining_amount']))
        c8.metric("Medium / Low Risk Works", f"{kpis['medium_risk_works']:,} / {kpis['low_risk_works']:,}")

        # Metrics Section 3: Officer Verification Pipeline
        cr1, cr2, cr3, cr4 = st.columns(4)
        cr1.metric("Pending Review Queue", f"{kpis['pending_review_works']:,}")
        cr2.metric("Verified Cases", f"{kpis['verified_works']:,}")
        cr3.metric("Needs Investigation", f"{kpis['needs_investigation_works']:,}")
        cr4.metric("False Positives Marked", f"{kpis['false_positive_works']:,}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Section 1
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("##### State Financial Sanction vs Expenditure")
            state_agg = aggregate_by_state(filtered_df).head(12)
            if not state_agg.empty:
                fig_state = go.Figure()
                fig_state.add_trace(go.Bar(x=state_agg['STATE_NAME'], y=state_agg['Total_Sanctioned']/1e7, name='Sanctioned (Cr)', marker_color='#2563eb'))
                fig_state.add_trace(go.Bar(x=state_agg['STATE_NAME'], y=state_agg['Total_Expenditure']/1e7, name='Expenditure (Cr)', marker_color='#10b981'))
                fig_state.update_layout(barmode='group', xaxis_tickangle=-45, yaxis_title="Amount (₹ Crore)")
                st.plotly_chart(style_plotly_chart(fig_state, 360), use_container_width=True)
            else:
                render_empty_state("No Data", "Insufficient data for state financial comparison.")

        with col_chart2:
            st.markdown("##### Risk Level Distribution")
            if 'RISK_LEVEL' in filtered_df.columns and not filtered_df.empty:
                risk_counts = filtered_df['RISK_LEVEL'].value_counts().reset_index()
                risk_counts.columns = ['Risk Level', 'Count']
                fig_risk = px.pie(
                    risk_counts, values='Count', names='Risk Level', 
                    color='Risk Level', color_discrete_map=PLOTLY_COLOR_DISCRETE,
                    hole=0.45
                )
                st.plotly_chart(style_plotly_chart(fig_risk, 360), use_container_width=True)
            else:
                render_empty_state("No Data", "No risk level classification available.")

        # Charts Section 2
        col_chart3, col_chart4 = st.columns(2)
        with col_chart3:
            st.markdown("##### Financial Allocation by Work Category")
            cat_agg = aggregate_by_category(filtered_df)
            if not cat_agg.empty:
                fig_cat = px.bar(
                    cat_agg, x='Total_Sanctioned', y='WORK_CATEGORY', 
                    orientation='h', color='Utilization_Pct', color_continuous_scale='Blues',
                    labels={'Total_Sanctioned': 'Sanctioned Amount (₹)', 'WORK_CATEGORY': 'Category'}
                )
                st.plotly_chart(style_plotly_chart(fig_cat, 340), use_container_width=True)

        with col_chart4:
            st.markdown("##### Physical Progress vs Financial Utilization Discrepancy")
            if not filtered_df.empty:
                fig_scatter = px.scatter(
                    filtered_df, x='PROGRESS_PERCENTAGE', y='UTILIZATION_PCT',
                    color='RISK_LEVEL', size='SANCTION_AMOUNT',
                    hover_data=['WORK_ID', 'STATE_NAME', 'WORK_CATEGORY'],
                    labels={'PROGRESS_PERCENTAGE': 'Physical Progress (%)', 'UTILIZATION_PCT': 'Financial Utilization (%)'},
                    color_discrete_map=PLOTLY_COLOR_DISCRETE
                )
                fig_scatter.add_shape(type="line", x0=0, y0=100, x1=100, y1=100, line=dict(color="#ef4444", width=1, dash="dash"))
                st.plotly_chart(style_plotly_chart(fig_scatter, 340), use_container_width=True)

# ==========================================
# TAB 2: RISK MONITORING MATRIX
# ==========================================
with tab_risk_mon:
    st.markdown("### Risk Monitoring & Anomaly Matrix")
    render_active_filter_banner()
    st.markdown("""
    Work items ordered by **Risk Score (Highest First)**. Synthesizes Isolation Forest outlier metrics, utilization-to-progress variance, cost escalation factors, and policy guideline compliance.
    """)
    
    if filtered_df.empty:
        render_empty_state("No Works Found", "No work items match the active selection criteria.")
    else:
        risk_mon_df = filtered_df.sort_values(by='RISK_SCORE', ascending=False).copy()
        
        def get_primary_reason(row):
            reasons = row.get('ANOMALY_REASON', '')
            factors = row.get('RISK_FACTORS', '')
            if pd.notna(reasons) and reasons != 'Normal execution pattern':
                return str(reasons).split(';')[0]
            elif pd.notna(factors) and factors != 'Standard implementation profile':
                return str(factors).split(';')[0]
            else:
                return "Normal parameter bounds"

        risk_mon_df['PRIMARY_RISK_REASON'] = risk_mon_df.apply(get_primary_reason, axis=1)
        risk_mon_df['DELAY_INFORMATION'] = risk_mon_df.apply(
            lambda r: f"Delayed ({r.get('PROJECTED_DELAY_MONTHS', 0.0)} mo)" if r.get('WORK_STATUS') == 'Delayed' else "On Schedule",
            axis=1
        )

        display_cols = [
            'WORK_ID', 'DISTRICT_NAME', 'WORK_CATEGORY', 
            'SANCTION_AMOUNT', 'EXPENDITURE_AMOUNT', 'PROGRESS_PERCENTAGE', 
            'DELAY_INFORMATION', 'RISK_SCORE', 'RISK_LEVEL', 
            'PRIMARY_RISK_REASON', 'REVIEW_STATUS'
        ]

        display_df = risk_mon_df[display_cols].rename(columns={
            'WORK_ID': 'Work ID',
            'DISTRICT_NAME': 'District',
            'WORK_CATEGORY': 'Work Category',
            'SANCTION_AMOUNT': 'Sanctioned Amount',
            'EXPENDITURE_AMOUNT': 'Expenditure',
            'PROGRESS_PERCENTAGE': 'Progress (%)',
            'DELAY_INFORMATION': 'Delay Info',
            'RISK_SCORE': 'Risk Score (0-100)',
            'RISK_LEVEL': 'Risk Level',
            'PRIMARY_RISK_REASON': 'Primary Indicator',
            'REVIEW_STATUS': 'Review Status'
        })

        st.dataframe(
            display_df.style.format({
                'Sanctioned Amount': lambda x: format_inr(x),
                'Expenditure': lambda x: format_inr(x),
                'Progress (%)': '{:.1f}%',
                'Risk Score (0-100)': '{:.1f}'
            }),
            use_container_width=True,
            height=520
        )
        st.caption("Select any Work ID in the 'Work Inspection' or 'Officer Verification' tab to view itemized diagnostic reasoning.")

# ==========================================
# TAB 3: WORK INSPECTION & AI DIAGNOSTIC
# ==========================================
with tab_work_detail:
    st.markdown("### Itemized Work Inspection & XAI Diagnostic")
    render_active_filter_banner()
    
    work_id_options = filtered_df['WORK_ID'].tolist() if not filtered_df.empty else []
    if work_id_options:
        selected_work_id = st.selectbox("Select Work ID for Inspection", work_id_options)
        work_row = filtered_df[filtered_df['WORK_ID'] == selected_work_id].iloc[0]
        
        explanation = generate_work_explanation(work_row)
        
        st.markdown(f"""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 14px 18px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span style="font-weight: 700; color: #f8fafc; font-size: 1.05rem;">Inspection Target: {explanation['work_id']}</span>
                <span style="margin-left: 12px; color: #94a3b8; font-size: 0.88rem;">{explanation['category']} &bull; {explanation['state']}, {explanation['district']}</span>
            </div>
            <div>
                <span class="badge badge-info">Data Access: {explanation['data_status_badge']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_inf1, col_inf2 = st.columns([1, 1])
        
        with col_inf1:
            st.markdown("#### 1. Work Record Overview")
            st.write(f"**Work ID:** `{explanation['work_id']}`")
            st.write(f"**Member of Parliament (MP):** {explanation['mp_name']}")
            st.write(f"**Constituency:** {explanation['constituency']}")
            st.write(f"**State / District:** {explanation['state']} / {explanation['district']}")
            st.write(f"**Work Category:** {explanation['category']}")
            st.write(f"**Work Scope:** {work_row.get('WORK_DESCRIPTION', 'N/A')}")
            st.markdown("---")
            st.write(f"**Recommended Amount:** {explanation['recommended_amount_str']}")
            st.write(f"**Sanctioned Amount:** {explanation['sanctioned_amount_str']}")
            st.write(f"**Expenditure Amount:** {explanation['expenditure_amount_str']}")
            st.write(f"**Physical Progress:** {explanation['progress_pct_str']}")
            st.write(f"**Current Status:** `{explanation['status']}`")
            st.write(f"**Delay Assessment:** {explanation['delay_info']}")

        with col_inf2:
            st.markdown("#### 2. Risk Evaluation & Diagnostic Reasoning")
            
            badge_class = "badge-critical" if explanation['risk_level'] == 'CRITICAL' else ("badge-high" if explanation['risk_level'] == 'HIGH' else ("badge-medium" if explanation['risk_level'] == 'MEDIUM' else "badge-low"))
            
            st.markdown(f"""
            <div class="explanation-panel">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 12px;">
                    <span style="font-size:1.15rem; font-weight:800; color:#f8fafc;">Risk Score: {explanation['risk_score']}/100</span>
                    <span class="badge {badge_class}">{explanation['risk_level']} RISK</span>
                </div>
                <div style="color:#38bdf8; font-size:0.85rem; font-weight:700; text-transform:uppercase; margin-bottom:8px;">Diagnostic Reasoning:</div>
            """, unsafe_allow_html=True)
            
            for reason in explanation['reasons']:
                st.markdown(f"<div style='color:#e2e8f0; font-size:0.9rem; margin-bottom:6px;'>&bull; {reason}</div>", unsafe_allow_html=True)
                
            st.markdown(f"""
                <hr style="border-color:#334155; margin: 14px 0 10px 0;">
                <div style="color:#fb923c; font-size:0.85rem;"><strong>Recommended Action:</strong> {explanation['recommended_action']}</div>
                <div style="color:#94a3b8; font-size:0.8rem; margin-top:4px;">{explanation['responsible_ai_notice']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 3. Administrative Verification Action")
        
        curr_status = work_row.get('REVIEW_STATUS', 'Pending Review')
        curr_remark = work_row.get('OFFICER_REMARK', '')
        
        col_ov1, col_ov2 = st.columns([1, 2])
        with col_ov1:
            new_status = st.radio(
                "Verification Determination",
                ["Pending Review", "Verified", "Needs Investigation", "False Positive"],
                index=["Pending Review", "Verified", "Needs Investigation", "False Positive"].index(curr_status) if curr_status in ["Pending Review", "Verified", "Needs Investigation", "False Positive"] else 0,
                key=f"radio_wd_{selected_work_id}"
            )
        with col_ov2:
            new_remark = st.text_area(
                "Officer Remarks & Inspection Log",
                value=curr_remark,
                placeholder="Enter physical site audit observations, voucher verification notes, or administrative justification...",
                key=f"remark_wd_{selected_work_id}"
            )
            if st.button("Save Verification Decision", key=f"btn_wd_{selected_work_id}", type="primary"):
                save_officer_review(selected_work_id, new_status, new_remark, user_role)
                st.success(f"Verification decision saved for Work ID '{selected_work_id}'. Status set to '{new_status}'.")
                st.rerun()

        st.markdown("---")
        st.markdown("#### 4. Milestone & Disbursement Schedule")
        
        sanc = float(work_row['SANCTION_AMOUNT']) if pd.notna(work_row.get('SANCTION_AMOUNT')) else 0.0
        prog = float(work_row['PROGRESS_PERCENTAGE']) if pd.notna(work_row.get('PROGRESS_PERCENTAGE')) else 0.0
        sanc_dt = str(work_row.get('SANCTION_DATE', 'N/A'))[:10]
        comp_dt = str(work_row.get('COMPLETION_DATE', 'N/A'))[:10]
        
        tranches = [
            {
                'Tranche': 'Tranche 1 (Commencement)',
                'Milestone Benchmark': 'Administrative Sanction & Commencement',
                'Share': '20%',
                'Disbursed Amount': format_inr(sanc * 0.20),
                'Date': sanc_dt,
                'Status': 'Released'
            },
            {
                'Tranche': 'Tranche 2 (Interim)',
                'Milestone Benchmark': '50% Physical Completion Benchmark',
                'Share': '50%',
                'Disbursed Amount': format_inr(sanc * 0.50),
                'Date': 'Disbursed' if prog >= 50 else 'Pending Benchmark',
                'Status': 'Released' if prog >= 50 else 'Held Pending Progress'
            },
            {
                'Tranche': 'Tranche 3 (Final Release)',
                'Milestone Benchmark': 'Final Asset Completion & Inspection',
                'Share': '30%',
                'Disbursed Amount': format_inr(sanc * 0.30),
                'Date': comp_dt if work_row['WORK_STATUS'] == 'Completed' else 'Pending Completion',
                'Status': 'Released' if work_row['WORK_STATUS'] == 'Completed' else 'Held Pending Completion'
            }
        ]
        st.dataframe(pd.DataFrame(tranches), use_container_width=True)
    else:
        render_empty_state("No Selection Available", "No work records available under current filter criteria.")

# ==========================================
# TAB 4: HUMAN-IN-THE-LOOP OFFICER REVIEW
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
                
                st.markdown("##### Identified Risk Indicators:")
                for r_item in exp_diag['reasons']:
                    st.write(f"&bull; {r_item}")
                    
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
# TAB 5: DATA ARCHITECTURE & TRANSPARENCY
# ==========================================
with tab_data_sources:
    st.markdown("### Data Architecture & Access Governance")
    render_active_filter_banner()
    
    st.markdown("""
    System architecture transparently delineates **publicly accessible portal data**, **session-gated API endpoints**, and **future integration streams**.
    """)
    
    c_ds1, c_ds2, c_ds3 = st.columns(3)
    
    with c_ds1:
        st.markdown("""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 18px; height: 100%;">
            <div style="color: #38bdf8; font-weight: 700; font-size: 0.95rem; margin-bottom: 8px;">A. Public Ingestion Layer</div>
            <div style="font-size: 0.82rem; color: #cbd5e1;">
                <strong>Source:</strong> MoSPI MPLADS Portal REST APIs (<code>mplads.mospi.gov.in</code>).
            </div>
            <ul style="font-size: 0.8rem; color: #94a3b8; padding-left: 16px; margin-top: 8px;">
                <li>State & District aggregate totals</li>
                <li>MP & Constituency allocations</li>
                <li>Sanctioned amounts & cumulative spend</li>
                <li>Implementation status classifications</li>
            </ul>
            <div class="badge badge-info" style="margin-top: 10px;">Connected in Production Engine</div>
        </div>
        """, unsafe_allow_html=True)

    with c_ds2:
        st.markdown("""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 18px; height: 100%;">
            <div style="color: #f59e0b; font-weight: 700; font-size: 0.95rem; margin-bottom: 8px;">B. Authenticated Government API</div>
            <div style="font-size: 0.82rem; color: #cbd5e1;">
                <strong>Restricted Endpoints:</strong> Role-based access control (RBAC).
            </div>
            <ul style="font-size: 0.8rem; color: #94a3b8; padding-left: 16px; margin-top: 8px;">
                <li>Itemized vendor expenditure vouchers</li>
                <li>Contractor PAN, GST, and banking records</li>
                <li>Detailed technical sanction approval files</li>
                <li>Internal officer workflow logs</li>
            </ul>
            <div class="badge badge-medium" style="margin-top: 10px;">Requires e-SAKSHI OAuth Credentials</div>
        </div>
        """, unsafe_allow_html=True)

    with c_ds3:
        st.markdown("""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 18px; height: 100%;">
            <div style="color: #a78bfa; font-weight: 700; font-size: 0.95rem; margin-bottom: 8px;">C. Cross-Agency Streams</div>
            <div style="font-size: 0.82rem; color: #cbd5e1;">
                <strong>External Data Streams:</strong> GIS & Remote Sensing feeds.
            </div>
            <ul style="font-size: 0.8rem; color: #94a3b8; padding-left: 16px; margin-top: 8px;">
                <li>PMGSY / Jal Jeevan Mission GIS layers</li>
                <li>Satellite & drone progress imaging</li>
                <li>Automated CV milestone verification</li>
            </ul>
            <div class="badge badge-low" style="margin-top: 10px;">Planned Enterprise Integration</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
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
    
    st.markdown("#### Security & Enterprise Deployment Governance")
    st.info("""
    🔒 **Security & Infrastructure Controls**:
    - **Deployment Profile**: Deployment architecture targets MeitY-empanelled Cloud / NIC Infrastructure utilizing PostgreSQL enterprise relational storage.
    - **Data Protection**: Enforces HTTPS/TLS data transit encryption, session-gated Role-Based Access Control (RBAC), multi-factor authentication for verifying officers, and immutable transaction audit logging.
    """)

    st.markdown("---")
    
    st.markdown("#### Functional Module Status Matrix")
    roadmap_df = pd.DataFrame([
        {"System Feature / Component": "MoSPI Public REST Ingestion Engine", "Status": "Active Engine", "Deployment Scope": "Core Pipeline"},
        {"System Feature / Component": "Scheme Guideline Compliance Audit", "Status": "Active Engine", "Deployment Scope": "Rule Engine"},
        {"System Feature / Component": "Isolation Forest Multivariate Outlier Detection", "Status": "Active Engine", "Deployment Scope": "ML Pipeline"},
        {"System Feature / Component": "Composite Risk Score (0-100) & Tiering", "Status": "Active Engine", "Deployment Scope": "Risk Engine"},
        {"System Feature / Component": "Explainable Risk Diagnostic Reasons", "Status": "Active Engine", "Deployment Scope": "XAI Engine"},
        {"System Feature / Component": "Officer Verification & Audit Log Persistence", "Status": "Active Engine", "Deployment Scope": "Governance Portal"},
        {"System Feature / Component": "TF-IDF Text Similarity Duplicate Matrix", "Status": "Active Engine", "Deployment Scope": "Forensics Engine"},
        {"System Feature / Component": "Predictive Trajectory Early Warnings", "Status": "Active Engine", "Deployment Scope": "Forecasting Engine"},
        {"System Feature / Component": "EXIF Geofence Integrity & Photo Variance Verification", "Status": "Active Sandbox", "Deployment Scope": "Spatial Engine"},
        {"System Feature / Component": "Cross-Agency Double-Funding Verification (PMGSY/JJM)", "Status": "Active Sandbox", "Deployment Scope": "Spatial Engine"},
        {"System Feature / Component": "NIC Cloud & Enterprise PostgreSQL Storage Layer", "Status": "Target Architecture", "Deployment Scope": "Infrastructure"}
    ])
    st.dataframe(roadmap_df, use_container_width=True)

# ==========================================
# TAB 6: ADVANCED ANALYTICS & POLICY AUDIT
# ==========================================
with tab_advanced:
    st.markdown("### Advanced Forensic Analytics & Policy Audit")
    render_active_filter_banner()
    
    sub_t1, sub_t2, sub_t3, sub_t4, sub_t5 = st.tabs([
        "Cross-Scheme Audit",
        "Spatial Integrity & S-Curve",
        "Predictive Warnings",
        "Duplicate Detection",
        "Policy Compliance"
    ])
    
    # SUB-TAB 1: CROSS SCHEME
    with sub_t1:
        st.markdown("#### Cross-Scheme Double Funding Spatial Audit")
        st.markdown("""
        Evaluates spatial proximity (<100m) and asset descriptions against parallel central/state scheme databases (PMGSY, Jal Jeevan Mission) to identify potential multi-source funding.
        """)
        
        if 'IS_CROSS_SCHEME_DUPLICATE' in filtered_df.columns:
            cs_matches = filtered_df[filtered_df['IS_CROSS_SCHEME_DUPLICATE'] == True]
        else:
            cs_matches = filtered_df.iloc[0:0].copy()
        
        col_cs1, col_cs2, col_cs3 = st.columns(3)
        col_cs1.metric("Flagged Proximity Overlaps", f"{len(cs_matches):,}")
        col_cs2.metric("Total Value at Risk", format_inr(cs_matches['SANCTION_AMOUNT'].sum()) if not cs_matches.empty else "₹0")
        col_cs3.metric("Avg Proximity Distance", f"{cs_matches['CROSS_SCHEME_DISTANCE_METERS'].mean():.1f} meters" if not cs_matches.empty and 'CROSS_SCHEME_DISTANCE_METERS' in cs_matches.columns else "N/A")
        
        if not cs_matches.empty:
            st.warning("Spatial Overlap Flagged: The works listed below match geographic coordinates and asset scope with parallel scheme records.")
            display_cs = cs_matches[['WORK_ID', 'STATE_NAME', 'DISTRICT_NAME', 'WORK_CATEGORY', 'SANCTION_AMOUNT', 'CROSS_SCHEME_MATCH_NAME', 'CROSS_SCHEME_DISTANCE_METERS', 'CROSS_SCHEME_MATCH_TITLE', 'RISK_SCORE']].copy()
            st.dataframe(
                display_cs.style.format({
                    'SANCTION_AMOUNT': lambda x: format_inr(x),
                    'CROSS_SCHEME_DISTANCE_METERS': '{:.1f} m',
                    'RISK_SCORE': '{:.1f}'
                }),
                use_container_width=True
            )
        else:
            st.success("No cross-scheme spatial overlaps detected in active filter selection.")

    # SUB-TAB 2: PHOTO GEOFENCE & S-CURVE
    with sub_t2:
        st.markdown("#### Spatial Geofence Variance & Progress S-Curve Velocity")
        st.markdown("""
        Cross-references embedded EXIF GPS coordinates from submitted progress photos against official sanctioned location coordinates (>100m threshold) and evaluates milestone burn-rate velocity.
        """)
        
        st.markdown("""
        <div style="background: #1e293b; border: 1px solid #334155; border-left: 4px solid #f59e0b; border-radius: 6px; padding: 10px 14px; margin-bottom: 16px; font-size: 0.83rem; color: #cbd5e1;">
            🔒 <strong>Data Architecture Note:</strong> Public MoSPI REST APIs do not publicly serve raw binary photo attachments. Photo EXIF GPS coordinates and perceptual hashes are evaluated using simulated metadata sandbox streams. Direct binary photo verification requires authenticated e-SAKSHI portal OAuth credentials.
        </div>
        """, unsafe_allow_html=True)
        
        if 'IS_GEO_MISMATCH' in filtered_df.columns:
            geo_mismatches = filtered_df[filtered_df['IS_GEO_MISMATCH'] == True]
        else:
            geo_mismatches = filtered_df.iloc[0:0].copy()
            
        if 'IS_DUPLICATE_PHOTO' in filtered_df.columns:
            dup_photos = filtered_df[filtered_df['IS_DUPLICATE_PHOTO'] == True]
        else:
            dup_photos = filtered_df.iloc[0:0].copy()
        
        col_g1, col_g2, col_g3 = st.columns(3)
        col_g1.metric("EXIF Geofence Mismatches (>100m)", f"{len(geo_mismatches):,}")
        col_g2.metric("Reused Photo Hash Flags", f"{len(dup_photos):,}")
        col_g3.metric("Avg Variance Distance", f"{geo_mismatches['GEOFENCE_DISTANCE_METERS'].mean():.1f} m" if not geo_mismatches.empty and 'GEOFENCE_DISTANCE_METERS' in geo_mismatches.columns else "0 m")
        
        if not geo_mismatches.empty:
            st.error("Geofence Variance Flagged: Submitted photo EXIF coordinates vary by >100m from sanctioned site coordinates.")
            display_geo = geo_mismatches[['WORK_ID', 'STATE_NAME', 'DISTRICT_NAME', 'WORK_CATEGORY', 'GEOFENCE_DISTANCE_METERS', 'LATITUDE', 'LONGITUDE', 'EXIF_LATITUDE', 'EXIF_LONGITUDE']].copy()
            st.dataframe(
                display_geo.style.format({
                    'GEOFENCE_DISTANCE_METERS': '{:.1f} m',
                    'LATITUDE': '{:.6f}',
                    'LONGITUDE': '{:.6f}',
                    'EXIF_LATITUDE': '{:.6f}',
                    'EXIF_LONGITUDE': '{:.6f}'
                }),
                use_container_width=True
            )
        else:
            st.success("All photo EXIF coordinates fall within official geofence boundaries.")

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

    # SUB-TAB 4: DUPLICATE WORK MATRIX
    with sub_t4:
        st.markdown("#### Duplicate Work Verification Matrix")
        st.markdown("""
        Identifies potential redundant or duplicate works using **TF-IDF Vector Space Cosine Similarity** across descriptions, locations, categories, and sanction bounds.
        """)
        
        filtered_work_ids = set(filtered_df['WORK_ID'].unique()) if not filtered_df.empty else set()
        
        if duplicates_matrix is not None and not duplicates_matrix.empty and filtered_work_ids:
            # STRICT FILTER ENFORCEMENT: Both Work A AND Work B must belong to filtered_work_ids
            filtered_dup = duplicates_matrix[
                duplicates_matrix['Work A ID'].isin(filtered_work_ids) & 
                duplicates_matrix['Work B ID'].isin(filtered_work_ids)
            ].copy()
            
            if not filtered_dup.empty:
                st.warning("High text/spatial similarity pairs requiring verification.")
                st.dataframe(
                    filtered_dup.style.format({
                        'Sanction Amount A': lambda x: format_inr(x),
                        'Sanction Amount B': lambda x: format_inr(x)
                    }),
                    use_container_width=True
                )
            else:
                render_empty_state("No Duplicates", "No duplicate work pairs match active selection criteria.")
        else:
            st.info("No potential duplicate works detected matching current criteria.")

    # SUB-TAB 5: POLICY COMPLIANCE AUDIT
    with sub_t5:
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

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.8rem; padding: 12px 0;">
    MPLADS Expenditure & Risk Monitoring Platform &bull; Ministry of Statistics and Programme Implementation (MoSPI)
</div>
""", unsafe_allow_html=True)
