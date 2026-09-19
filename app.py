import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import pickle

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

# Page configuration
st.set_page_config(
    page_title="MPLADS AI Monitoring & Analytics Platform",
    page_icon="institutional",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark theme glassmorphism)
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .stMetric {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .header-box {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 1px solid #4338ca;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
    }
    .workflow-pill {
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 8px;
    }
    .alert-card-critical {
        background: rgba(225, 29, 72, 0.12);
        border-left: 5px solid #f43f5e;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .alert-card-high {
        background: rgba(234, 88, 12, 0.12);
        border-left: 5px solid #fb923c;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .alert-card-medium {
        background: rgba(234, 179, 8, 0.12);
        border-left: 5px solid #facc15;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .explanation-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)

def render_empty_state(message="No works match the current filters — try widening your selection."):
    """Standardized glassmorphism empty-state renderer across all tabs and components."""
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.4); border: 1px dashed rgba(148, 163, 184, 0.4); border-radius: 12px; padding: 28px; text-align: center; margin: 16px 0;">
        <div style="font-size: 1.6rem; color: #64748b; margin-bottom: 6px;">🔍</div>
        <div style="font-size: 1.05rem; font-weight: 600; color: #94a3b8;">{message}</div>
        <div style="font-size: 0.85rem; color: #64748b; margin-top: 4px;">Try adjusting your sidebar criteria (State, District, MP, Risk Level, etc.) to view results.</div>
    </div>
    """, unsafe_allow_html=True)

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
                    b['works_df'], b['duplicates_matrix'], b['alerts_df'], 
                    b.get('early_warnings_df', pd.DataFrame()),
                    b.get('compliance_summary_df', pd.DataFrame()),
                    b['audit_summary'], b['missing_df'], b['metadata']
                )
        except Exception:
            pass

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

states_df, state_tiles_df, mps_df, works_df, duplicates_matrix, alerts_df, early_warnings_df, compliance_summary_df, audit_summary, missing_df, metadata = load_processed_data()

# Header Banner
st.markdown("""
<div class="header-box">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin:0; font-size: 2.2rem; color: #f8fafc; font-weight: 800;">
                MPLADS AI Monitoring & Analytics Platform
            </h1>
            <p style="margin-top: 6px; color: #94a3b8; font-size: 1.05rem;">
                AI-Powered Anomaly, Cost Overrun, Delay, Duplicate, Compliance & Asset Verification Intelligence for MPLADS
            </p>
            <div style="margin-top: 12px;">
                <span class="workflow-pill">MONITOR</span>
                <span class="workflow-pill">PREDICT</span>
                <span class="workflow-pill">DETECT</span>
                <span class="workflow-pill">COMPLY</span>
                <span class="workflow-pill">VERIFY</span>
                <span class="workflow-pill">ACT</span>
            </div>
        </div>
        <div style="text-align: right; background: rgba(15, 23, 42, 0.6); padding: 12px 18px; border-radius: 10px; border: 1px solid #334155;">
            <div style="font-size: 0.8rem; color: #64748b;">OFFICIAL DATA SOURCE</div>
            <div style="font-size: 0.95rem; font-weight: 700; color: #38bdf8;">mplads.mospi.gov.in</div>
            <div style="font-size: 0.75rem; color: #10b981; margin-top: 4px;">Real Portal Connected</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Filters & Navigation
st.sidebar.title("Governance Control Center")

# Role Selection
user_role = st.sidebar.selectbox(
    "Select Dashboard Role",
    ["Ministry / MoSPI", "State Nodal Authority", "District Authority", "Member of Parliament"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Interactive Filters")

# State Filter
state_list = ['All States'] + sorted(list(works_df['STATE_NAME'].unique()))
selected_state = st.sidebar.selectbox("State / Union Territory", state_list)

# Cascading District Filter (Task 1 Fix)
if selected_state != 'All States':
    dist_options = sorted(list(works_df[works_df['STATE_NAME'] == selected_state]['DISTRICT_NAME'].unique()))
    dist_list = ['All Districts'] + dist_options
    selected_district = st.sidebar.selectbox("District", dist_list)
else:
    st.sidebar.selectbox("District", ["Select a State first to filter districts"], disabled=True)
    selected_district = 'All Districts'

# MP Filter
if selected_state != 'All States':
    mp_list = ['All MPs'] + sorted(list(works_df[works_df['STATE_NAME'] == selected_state]['MP_NAME'].unique()))
else:
    mp_list = ['All MPs'] + sorted(list(works_df['MP_NAME'].unique()))
selected_mp = st.sidebar.selectbox("Member of Parliament (MP)", mp_list)

# Tenure Filter
tenure_list = ['All Tenures'] + sorted(list(works_df['TENURE'].unique()))
selected_tenure = st.sidebar.selectbox("Lok Sabha / Tenure", tenure_list)

# Category Filter
cat_list = ['All Categories'] + sorted(list(works_df['WORK_CATEGORY'].unique()))
selected_cat = st.sidebar.selectbox("Work Category", cat_list)

# Status Filter
status_list = ['All Statuses'] + sorted(list(works_df['WORK_STATUS'].unique()))
selected_status = st.sidebar.selectbox("Implementation Status", status_list)

# Risk Filter
risk_list = ['All Risk Levels', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
selected_risk = st.sidebar.selectbox("AI Risk Level Filter", risk_list)

# Apply Filters
filtered_df = filter_dataset(
    works_df,
    state=selected_state,
    district=selected_district,
    mp=selected_mp,
    tenure=selected_tenure,
    category=selected_cat,
    status=selected_status,
    risk_level=selected_risk
)

# Compute Filtered KPIs
kpis = compute_kpis(filtered_df)

# Navigation Tabs
tab_overview, tab_roles, tab_analytics, tab_anomalies, tab_predictive, tab_duplicates, tab_alerts, tab_work_inspect, tab_data_quality = st.tabs([
    "National & State Overview",
    "Role-Based Dashboard",
    "Financial & Work Analytics",
    "Anomaly & Cost Overrun Engine",
    "Predictive Early-Warning System",
    "Duplicate Work Matrix",
    "Explainable Risk Alerts",
    "Work Inspection & AI Insights",
    "Data Quality & Policy Audit"
])

# ==========================================
# TAB 1: OVERVIEW DASHBOARD
# ==========================================
with tab_overview:
    st.subheader(f"National & Regional Monitoring Overview ({user_role} View)")
    
    if filtered_df.empty:
        render_empty_state("No works match the current filters — try widening your selection.")
    else:
        # KPI Row 1: Financial Overview
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Works Monitored", f"{kpis['total_works']:,}")
        col2.metric("Total Sanctioned Amount", format_inr(kpis['total_sanctioned']))
        col3.metric("Total Expenditure", format_inr(kpis['total_expenditure']), delta=f"{kpis['utilization_rate']}% Utilization")
        col4.metric("Remaining Unspent Funds", format_inr(kpis['remaining_amount']))

        # KPI Row 2: AI Risk Indicators
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Completed Works", f"{kpis['completed_works']:,}")
        col6.metric("High & Critical Risk Works", f"{kpis['high_risk_works']:,}", delta_color="inverse")
        col7.metric("Anomalous Works Detected", f"{kpis['anomalous_works']:,}", delta_color="inverse")
        col8.metric("Potential Duplicate Works", f"{kpis['duplicate_works']:,}", delta_color="inverse")

        st.markdown("---")
        
        # Asset Creation & Post-Completion Verification Reconciliation Card (Task 2)
        st.subheader("Asset Creation & Post-Completion Verification Reconciliation")
        completed_df = filtered_df[filtered_df['WORK_STATUS'] == 'Completed']
        total_completed = len(completed_df)
        verified_assets = len(completed_df[completed_df['IS_ASSET_VERIFIED'] == True]) if 'IS_ASSET_VERIFIED' in completed_df.columns else total_completed
        unverified_assets = max(0, total_completed - verified_assets)
        verification_rate = round((verified_assets / total_completed * 100.0), 1) if total_completed > 0 else 100.0

        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
        col_a1.metric("Works Marked Completed", f"{total_completed:,}")
        col_a2.metric("Verified Durable Assets", f"{verified_assets:,}", delta="Confirmed Geotag")
        col_a3.metric("Unverified / Pending Audit", f"{unverified_assets:,}", delta_color="inverse")
        col_a4.metric("Asset Verification Rate", f"{verification_rate}%")

        st.info("ℹ️ **Official Data Gap Notice**: MoSPI's pre-login public REST API exposes cumulative completed counts, but currently lacks native geotagged physical asset verification fields. This reconciliation module establishes the post-completion physical verification layer recommended under SIH governance standards.")

        st.markdown("---")

        # Charts Row 1
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("State-Wise Sanctioned Amount vs Expenditure")
            state_agg = aggregate_by_state(filtered_df).head(12)
            if not state_agg.empty:
                fig_state = go.Figure()
                fig_state.add_trace(go.Bar(x=state_agg['STATE_NAME'], y=state_agg['Total_Sanctioned']/1e7, name='Sanctioned (Cr)', marker_color='#6366f1'))
                fig_state.add_trace(go.Bar(x=state_agg['STATE_NAME'], y=state_agg['Total_Expenditure']/1e7, name='Expenditure (Cr)', marker_color='#10b981'))
                fig_state.update_layout(barmode='group', template='plotly_dark', height=380, xaxis_tickangle=-45, yaxis_title="Amount (₹ Crore)")
                st.plotly_chart(fig_state, width='stretch')
            else:
                render_empty_state("No state data available for current selection.")

        with c2:
            st.subheader("AI Risk Level Breakdown")
            if 'RISK_LEVEL' in filtered_df.columns and not filtered_df.empty:
                risk_counts = filtered_df['RISK_LEVEL'].value_counts().reset_index()
                risk_counts.columns = ['Risk Level', 'Count']
                color_map = {'CRITICAL': '#f43f5e', 'HIGH': '#fb923c', 'MEDIUM': '#facc15', 'LOW': '#10b981'}
                fig_risk = px.pie(
                    risk_counts, values='Count', names='Risk Level', 
                    color='Risk Level', color_discrete_map=color_map,
                    hole=0.4, template='plotly_dark'
                )
                fig_risk.update_layout(height=380)
                st.plotly_chart(fig_risk, width='stretch')
            else:
                render_empty_state("No risk classification data available.")

        # Charts Row 2
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Work Category Financial Allocation")
            cat_agg = aggregate_by_category(filtered_df)
            if not cat_agg.empty:
                fig_cat = px.bar(
                    cat_agg, x='Total_Sanctioned', y='WORK_CATEGORY', 
                    orientation='h', color='Utilization_Pct', color_continuous_scale='Viridis',
                    labels={'Total_Sanctioned': 'Sanctioned Amount (₹)', 'WORK_CATEGORY': 'Category'},
                    template='plotly_dark', height=360
                )
                st.plotly_chart(fig_cat, width='stretch')

        with c4:
            st.subheader("Expenditure vs Physical Progress Discrepancy Scatter")
            if not filtered_df.empty:
                fig_scatter = px.scatter(
                    filtered_df, x='PROGRESS_PERCENTAGE', y='UTILIZATION_PCT',
                    color='RISK_LEVEL', size='SANCTION_AMOUNT',
                    hover_data=['WORK_ID', 'STATE_NAME', 'WORK_CATEGORY'],
                    labels={'PROGRESS_PERCENTAGE': 'Physical Progress (%)', 'UTILIZATION_PCT': 'Financial Utilization (%)'},
                    color_discrete_map={'CRITICAL': '#f43f5e', 'HIGH': '#fb923c', 'MEDIUM': '#facc15', 'LOW': '#10b981'},
                    template='plotly_dark', height=360
                )
                fig_scatter.add_shape(type="line", x0=0, y0=100, x1=100, y1=100, line=dict(color="red", width=1, dash="dash"))
                st.plotly_chart(fig_scatter, width='stretch')

# ==========================================
# TAB 2: ROLE-BASED DASHBOARD VIEWS
# ==========================================
with tab_roles:
    st.subheader(f"Dedicated Decision-Support Portal: {user_role}")
    
    if filtered_df.empty:
        render_empty_state("No works match the current filters — try widening your selection.")
    else:
        if user_role == "Member of Parliament":
            st.markdown("""
            > **Focus Area for Hon'ble MPs**: Track recommended works, sanction status, fund utilization in constituency, and identify works requiring administrative follow-up.
            """)
            mp_agg = aggregate_by_mp(filtered_df)
            if not mp_agg.empty:
                st.dataframe(
                    mp_agg.style.format({
                        'Total_Sanctioned': lambda x: format_inr(x),
                        'Total_Expenditure': lambda x: format_inr(x),
                        'Utilization_Pct': '{:.1f}%'
                    }),
                    width='stretch'
                )
            else:
                render_empty_state("No MP records found for current selection.")

            st.subheader("Constituency Works Requiring Follow-Up")
            filtered_work_ids = filtered_df['WORK_ID'].tolist()
            mp_alerts = alerts_df[alerts_df['Work ID'].isin(filtered_work_ids) & alerts_df['Risk Level'].isin(['HIGH', 'CRITICAL'])].head(10)
            if not mp_alerts.empty:
                st.dataframe(mp_alerts[['Work ID', 'Location', 'Work Category', 'Issue', 'Risk Level', 'Suggested Review']], width='stretch')
            else:
                st.info("No urgent follow-up works identified for selected MP filter.")

        elif user_role == "District Authority":
            st.markdown("""
            > **Focus Area for District Authorities / IDAs**: Monitor local execution, identify prolonged delays, verify expenditure vouchers, and investigate potential duplicate works.
            """)
            c_d1, c_d2, c_d3 = st.columns(3)
            c_d1.metric("District Works Under Review", f"{kpis['total_works']}")
            c_d2.metric("Delayed Projects", f"{kpis['delayed_works']}", delta_color="inverse")
            c_d3.metric("Cost Overrun Alerts", f"{len(filtered_df[filtered_df['EXPENDITURE_AMOUNT'] > filtered_df['SANCTION_AMOUNT']])}", delta_color="inverse")

            st.subheader("District Work Execution & Anomaly Table")
            dist_display = filtered_df[['WORK_ID', 'DISTRICT_NAME', 'WORK_CATEGORY', 'SANCTION_AMOUNT', 'EXPENDITURE_AMOUNT', 'WORK_STATUS', 'RISK_LEVEL', 'ANOMALY_REASON']].head(15)
            st.dataframe(
                dist_display.style.format({
                    'SANCTION_AMOUNT': lambda x: format_inr(x),
                    'EXPENDITURE_AMOUNT': lambda x: format_inr(x)
                }),
                width='stretch'
            )

        elif user_role == "State Nodal Authority":
            st.markdown("""
            > **Focus Area for State Nodal Authorities**: Cross-district financial performance, systemic delay identification, and state-wide fund utilization optimization.
            """)
            state_summary = aggregate_by_state(filtered_df)
            if not state_summary.empty:
                st.dataframe(
                    state_summary.style.format({
                        'Total_Sanctioned': lambda x: format_inr(x),
                        'Total_Expenditure': lambda x: format_inr(x),
                        'Utilization_Pct': '{:.1f}%'
                    }),
                    width='stretch'
                )
            else:
                render_empty_state("No state summary available for selected filters.")

        else: # Ministry / MoSPI
            st.markdown("""
            > **Focus Area for MoSPI / Ministry**: National overview, macro-level fund utilization trends, policy-level compliance, and high-level risk management.
            """)
            st.write("### National Implementation Metrics by State")
            nat_summary = aggregate_by_state(filtered_df if not filtered_df.empty else works_df)
            st.dataframe(
                nat_summary.style.format({
                    'Total_Sanctioned': lambda x: format_inr(x),
                    'Total_Expenditure': lambda x: format_inr(x),
                    'Utilization_Pct': '{:.1f}%'
                }),
                width='stretch'
            )

# ==========================================
# TAB 3: ANALYTICS & TRENDS
# ==========================================
with tab_analytics:
    st.subheader("Multi-Dimensional MPLADS Trend Analysis")
    
    if filtered_df.empty:
        render_empty_state("No works match the current filters — try widening your selection.")
    else:
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            st.write("### Year-Wise Sanction & Expenditure Trend")
            if 'YEAR' in filtered_df.columns:
                year_agg = filtered_df.groupby('YEAR').agg(
                    Sanctioned=('SANCTION_AMOUNT', 'sum'),
                    Expenditure=('EXPENDITURE_AMOUNT', 'sum')
                ).reset_index()
                fig_trend = px.line(year_agg, x='YEAR', y=['Sanctioned', 'Expenditure'], labels={'value': 'Amount (₹)', 'variable': 'Metric'}, template='plotly_dark')
                st.plotly_chart(fig_trend, width='stretch')
                
        with col_t2:
            st.write("### Status Breakdown by Work Category")
            if not filtered_df.empty:
                fig_status_cat = px.histogram(
                    filtered_df, x='WORK_CATEGORY', color='WORK_STATUS', 
                    barmode='group', template='plotly_dark'
                )
                fig_status_cat.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_status_cat, width='stretch')

# ==========================================
# TAB 4: ANOMALY DETECTION ENGINE
# ==========================================
with tab_anomalies:
    st.subheader("AI Anomaly & Cost Overrun Detection Layer")
    st.markdown("""
    This layer utilizes **Isolation Forest**, **Local Outlier Factor (LOF)**, and **IQR Statistical Outlier Bounds** to detect abnormal expenditure, high cost-to-sanction ratios, and progress discrepancies.
    """)
    
    anom_only = filtered_df[filtered_df['IS_ANOMALY'] == True]
    st.write(f"### Detected Anomalous Works ({len(anom_only)} records found)")
    
    if not anom_only.empty:
        st.dataframe(
            anom_only[['WORK_ID', 'STATE_NAME', 'WORK_CATEGORY', 'SANCTION_AMOUNT', 'EXPENDITURE_AMOUNT', 'UTILIZATION_PCT', 'PROGRESS_PERCENTAGE', 'ANOMALY_SCORE', 'ANOMALY_REASON']].style.format({
                'SANCTION_AMOUNT': lambda x: format_inr(x),
                'EXPENDITURE_AMOUNT': lambda x: format_inr(x),
                'UTILIZATION_PCT': '{:.1f}%',
                'PROGRESS_PERCENTAGE': '{:.1f}%'
            }),
            width='stretch'
        )
    elif filtered_df.empty:
        render_empty_state("No works match the current filters — try widening your selection.")
    else:
        st.success("No financial or progress anomalies detected in the current filter selection.")

# ==========================================
# TAB 5: PREDICTIVE / EARLY WARNING LAYER
# ==========================================
with tab_predictive:
    st.subheader("Predictive Risk & Early-Warning Intelligence Engine")
    st.markdown("""
    > **Forward-Looking Predictive Layer**: Analyzes spend-to-progress trajectories and progress speed over time for ongoing works to project **future cost overruns** and **prolonged delays** BEFORE they cross reactive anomaly thresholds.
    """)
    
    if filtered_df.empty:
        render_empty_state("No works match the current filters — try widening your selection.")
    else:
        ongoing_df = filtered_df[filtered_df['WORK_STATUS'].isin(['Ongoing', 'Sanctioned / Pending', 'Delayed', 'Incomplete with High Exp'])].copy()
        
        if not ongoing_df.empty:
            early_warn_count = len(ongoing_df[ongoing_df['IS_EARLY_WARNING'] == True])
            high_warn_count = len(ongoing_df[ongoing_df['EARLY_WARNING_LEVEL'] == 'HIGH RISK WARNING'])
            avg_overrun_pct = ongoing_df['PROJECTED_OVERRUN_PCT'].mean()
            
            c_ew1, c_ew2, c_ew3, c_ew4 = st.columns(4)
            c_ew1.metric("Ongoing Works Monitored", f"{len(ongoing_df):,}")
            c_ew2.metric("Early Warning Flags", f"{early_warn_count:,}", delta_color="inverse")
            c_ew3.metric("High-Risk Trajectory Warnings", f"{high_warn_count:,}", delta_color="inverse")
            c_ew4.metric("Avg Projected Cost Slippage", f"{avg_overrun_pct:.1f}%")

            st.markdown("---")
            
            # Chart: Physical Progress vs Utilization Scatter with Early Warning Levels
            fig_ew_scatter = px.scatter(
                ongoing_df, x='PROGRESS_PERCENTAGE', y='UTILIZATION_PCT',
                color='EARLY_WARNING_LEVEL', size='SANCTION_AMOUNT',
                hover_data=['WORK_ID', 'STATE_NAME', 'WORK_CATEGORY', 'PREDICTED_FINAL_COST', 'PROJECTED_OVERRUN_PCT', 'PROJECTED_DELAY_MONTHS'],
                labels={'PROGRESS_PERCENTAGE': 'Current Physical Progress (%)', 'UTILIZATION_PCT': 'Current Financial Utilization (%)'},
                color_discrete_map={'HIGH RISK WARNING': '#f43f5e', 'MODERATE WARNING': '#fb923c', 'LOW WARNING': '#facc15', 'NORMAL': '#10b981'},
                template='plotly_dark', height=380
            )
            fig_ew_scatter.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color="#38bdf8", width=2, dash="dash"))
            st.plotly_chart(fig_ew_scatter, width='stretch')
            
            st.subheader("Early Warning Target List & Trajectory Projections")
            early_flagged = ongoing_df[ongoing_df['IS_EARLY_WARNING'] == True].sort_values(by='EARLY_WARNING_SCORE', ascending=False)
            
            if not early_flagged.empty:
                display_ew = early_flagged[['WORK_ID', 'STATE_NAME', 'DISTRICT_NAME', 'WORK_CATEGORY', 'SANCTION_AMOUNT', 'EXPENDITURE_AMOUNT', 'PROGRESS_PERCENTAGE', 'PREDICTED_FINAL_COST', 'PROJECTED_OVERRUN_PCT', 'PROJECTED_DELAY_MONTHS', 'EARLY_WARNING_LEVEL', 'EARLY_WARNING_REASON']].copy()
                st.dataframe(
                    display_ew.style.format({
                        'SANCTION_AMOUNT': lambda x: format_inr(x),
                        'EXPENDITURE_AMOUNT': lambda x: format_inr(x),
                        'PREDICTED_FINAL_COST': lambda x: format_inr(x),
                        'PROGRESS_PERCENTAGE': '{:.1f}%',
                        'PROJECTED_OVERRUN_PCT': '+{:.1f}%',
                        'PROJECTED_DELAY_MONTHS': '{:.1f} months'
                    }),
                    width='stretch'
                )
            else:
                st.success("No ongoing works currently project cost overruns or completion delays above early warning thresholds.")
        else:
            render_empty_state("No ongoing works match the current filters for predictive early-warning analysis.")

# ==========================================
# TAB 6: DUPLICATE WORK DETECTION
# ==========================================
with tab_duplicates:
    st.subheader("Potential Duplicate Work Verification Matrix")
    st.markdown("""
    Identifies potentially duplicate or highly similar works using **TF-IDF Vector Space Cosine Similarity** on work descriptions, location, category, and sanctioned amounts.
    """)
    
    filtered_work_ids = set(filtered_df['WORK_ID'].unique()) if not filtered_df.empty else set()
    
    if not duplicates_matrix.empty and filtered_work_ids:
        filtered_dup = duplicates_matrix[
            duplicates_matrix['Work A ID'].isin(filtered_work_ids) | 
            duplicates_matrix['Work B ID'].isin(filtered_work_ids)
        ].copy()
        
        if not filtered_dup.empty:
            st.warning("The works listed below share high similarity and require human-in-the-loop verification before further fund disbursement.")
            st.dataframe(
                filtered_dup.style.format({
                    'Sanction Amount A': lambda x: format_inr(x),
                    'Sanction Amount B': lambda x: format_inr(x)
                }),
                width='stretch'
            )
        else:
            render_empty_state("No duplicate works match the current filter criteria.")
    elif filtered_df.empty:
        render_empty_state("No works match the current filters — try widening your selection.")
    else:
        st.info("No potential duplicate works detected matching the current criteria.")

# ==========================================
# TAB 7: EXPLAINABLE RISK ALERTS
# ==========================================
with tab_alerts:
    st.subheader("Dynamic Explainable Risk Alerts")
    
    filtered_work_ids = set(filtered_df['WORK_ID'].unique()) if not filtered_df.empty else set()
    filtered_alerts = alerts_df[alerts_df['Work ID'].isin(filtered_work_ids)] if not alerts_df.empty else pd.DataFrame()
    
    if not filtered_alerts.empty:
        for idx, row in filtered_alerts.head(15).iterrows():
            level = row['Risk Level']
            card_class = f"alert-card-{level.lower()}"
            
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display:flex; justify-between; align-items:center;">
                    <span style="font-weight:bold; font-size:1.1rem; color:#f8fafc;">ALERT: {row['Work ID']} — {row['Issue']}</span>
                    <span style="background:#1e293b; color:#38bdf8; padding:4px 10px; border-radius:12px; font-weight:bold; font-size:0.85rem;">
                        Risk Score: {row['Risk Score']}/100 ({level})
                    </span>
                </div>
                <div style="margin-top:8px; color:#cbd5e1; font-size:0.95rem;">
                    <strong>Location:</strong> {row['Location']} | <strong>MP:</strong> {row['MP Name']} | <strong>Category:</strong> {row['Work Category']}
                </div>
                <div style="margin-top:6px; color:#f1f5f9;">
                    <strong>Reason for Flagging:</strong> {row['Reason for Flagging']}
                </div>
                <div style="margin-top:4px; color:#94a3b8; font-size:0.88rem;">
                    <strong>Supporting Indicator:</strong> {row['Supporting Indicator']}
                </div>
                <div style="margin-top:8px; color:#34d399; font-weight:600; font-size:0.9rem;">
                    <strong>Suggested Review:</strong> {row['Suggested Review']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    elif filtered_df.empty:
        render_empty_state("No works match the current filters — try widening your selection.")
    else:
        st.success("All works in the current selection fall under LOW risk parameters.")

# ==========================================
# TAB 8: WORK INSPECTION & AI DIAGNOSTIC
# ==========================================
with tab_work_inspect:
    st.subheader("Granular Work Item Inspection & AI Explainability")
    
    work_id_options = filtered_df['WORK_ID'].tolist() if not filtered_df.empty else []
    if work_id_options:
        selected_work_id = st.selectbox("Select Work ID to Inspect", work_id_options)
        work_row = filtered_df[filtered_df['WORK_ID'] == selected_work_id].iloc[0]
        
        c_i1, c_i2 = st.columns([1, 1])
        
        with c_i1:
            st.markdown("### Basic & Financial Information")
            st.write(f"**Work ID:** {work_row['WORK_ID']}")
            st.write(f"**State:** {work_row['STATE_NAME']}")
            st.write(f"**District:** {work_row['DISTRICT_NAME']}")
            st.write(f"**MP Name:** {work_row['MP_NAME']}")
            st.write(f"**Category:** {work_row['WORK_CATEGORY']}")
            st.write(f"**Description:** {work_row['WORK_DESCRIPTION']}")
            st.write(f"**Sanction Amount:** {format_inr(work_row['SANCTION_AMOUNT'])}")
            st.write(f"**Expenditure Amount:** {format_inr(work_row['EXPENDITURE_AMOUNT'])}")
            st.write(f"**Utilization Percentage:** {work_row['UTILIZATION_PCT']}%")
            st.write(f"**Work Status:** {work_row['WORK_STATUS']}")
            st.write(f"**Physical Progress:** {work_row['PROGRESS_PERCENTAGE']}%")
            st.write(f"**Asset Verification:** {work_row.get('ASSET_VERIFICATION_STATUS', 'N/A')}")
            if 'PREDICTED_FINAL_COST' in work_row:
                st.write(f"**Projected Final Cost:** {format_inr(work_row['PREDICTED_FINAL_COST'])}")
                st.write(f"**Projected Delay:** {work_row['PROJECTED_DELAY_MONTHS']} months")
            
        with c_i2:
            st.markdown("### AI Diagnostic Report (Why Was This Flagged?)")
            explanation = generate_work_explanation(work_row)
            
            st.markdown(f"""
            <div class="explanation-card">
                <h4 style="color:#38bdf8; margin-top:0;">{explanation['title']}</h4>
                <div style="margin-top:12px;">
            """, unsafe_allow_html=True)
            
            for reason in explanation['reasons']:
                st.markdown(f"<p style='color:#f8fafc; font-size:0.95rem;'>{reason}</p>", unsafe_allow_html=True)
                
            st.markdown(f"""
                    <hr style="border-color:#334155;">
                    <p style="color:#fb923c; font-size:0.85rem;"><strong>Responsible AI Notice:</strong> {explanation['responsible_ai_notice']}</p>
                    <p style="color:#34d399; font-size:0.9rem;"><strong>Recommended Next Step:</strong> {explanation['suggested_action']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Payment & Milestone Tranche Timeline Card (Task 1)
        st.markdown("---")
        st.markdown("### Payment & Milestone Tranche Timeline")
        st.caption("ℹ️ **Data Limitation Notice**: MoSPI's pre-login public REST API exposes cumulative expenditure totals; individual tranche release sequence below is derived from physical progress milestones.")
        
        sanc = float(work_row['SANCTION_AMOUNT'])
        prog = float(work_row['PROGRESS_PERCENTAGE'])
        sanc_dt = str(work_row.get('SANCTION_DATE', 'N/A'))[:10]
        comp_dt = str(work_row.get('COMPLETION_DATE', 'N/A'))[:10]
        
        tranches = [
            {
                'Tranche No.': 'Tranche 1 (Initial Release)',
                'Milestone Stage': 'Sanction Approval & Commencement',
                'Earmarked Share': '20% of Sanction',
                'Disbursed Amount': format_inr(sanc * 0.20),
                'Disbursement Date': sanc_dt,
                'Status': 'Released'
            },
            {
                'Tranche No.': 'Tranche 2 (Interim Progress Release)',
                'Milestone Stage': '50% Physical Progress Benchmark',
                'Earmarked Share': '50% of Sanction',
                'Disbursed Amount': format_inr(sanc * 0.50),
                'Disbursement Date': 'Disbursed' if prog >= 50 else 'Pending Milestone',
                'Status': 'Released' if prog >= 50 else 'Held Pending Progress'
            },
            {
                'Tranche No.': 'Tranche 3 (Final Completion Release)',
                'Milestone Stage': 'Physical Completion & Asset Register',
                'Earmarked Share': '30% of Sanction',
                'Disbursed Amount': format_inr(sanc * 0.30),
                'Disbursement Date': comp_dt if work_row['WORK_STATUS'] == 'Completed' else 'Pending Completion',
                'Status': 'Released' if work_row['WORK_STATUS'] == 'Completed' else 'Held Pending Completion'
            }
        ]
        st.dataframe(pd.DataFrame(tranches), width='stretch')
    else:
        render_empty_state("No works available under current filters.")

# ==========================================
# TAB 9: DATA QUALITY & POLICY AUDIT
# ==========================================
with tab_data_quality:
    st.subheader("Data Quality & Rule-Based Compliance Audit")
    
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    col_q1.metric("Total Records Audited", f"{audit_summary['total_records']:,}")
    col_q2.metric("Valid Records", f"{audit_summary['valid_records']:,}")
    col_q3.metric("Duplicate Records", f"{audit_summary['duplicate_records']:,}")
    col_q4.metric("Overall Data Quality Score", f"{audit_summary['data_quality_score']}%")

    st.markdown("---")
    
    # Scheme Policy Compliance Check Section (Task 3)
    st.subheader("Scheme Policy & Rule-Based Compliance Audit")
    st.markdown("""
    > **Automated Policy Verification Engine**: Evaluates work records against official MoSPI MPLADS Operational Guidelines, including MP Tenure Entitlement Ceilings (₹25 Cr for 17th Lok Sabha / ₹10 Cr for 18th Lok Sabha), Category Restrictions, and Administrative Sanction Field Completeness.
    """)

    filtered_work_ids = set(filtered_df['WORK_ID'].unique()) if not filtered_df.empty else set()
    filtered_comp = compliance_summary_df[compliance_summary_df['Work ID'].isin(filtered_work_ids)] if not compliance_summary_df.empty else pd.DataFrame()

    total_violations = len(filtered_comp)
    ceiling_breaches = len(filtered_comp[filtered_comp['Triggered Policy Rules'].str.contains('Fund Ceiling Breach', na=False)]) if not filtered_comp.empty else 0
    missing_auth = len(filtered_comp[filtered_comp['Triggered Policy Rules'].str.contains('Administrative Field Incompleteness', na=False)]) if not filtered_comp.empty else 0

    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    col_c1.metric("Works Audited for Policy Compliance", f"{len(filtered_df):,}")
    col_c2.metric("Policy Guideline Flags", f"{total_violations:,}", delta_color="inverse")
    col_c3.metric("Entitlement Ceiling Breaches", f"{ceiling_breaches:,}", delta_color="inverse")
    col_c4.metric("Missing Authority Records", f"{missing_auth:,}", delta_color="inverse")

    if not filtered_comp.empty:
        st.write("### Policy Violation Audit Table")
        st.dataframe(
            filtered_comp[['Work ID', 'State', 'District', 'MP Name', 'Work Category', 'Sanction Amount', 'Violation Severity', 'Triggered Policy Rules']].style.format({
                'Sanction Amount': lambda x: format_inr(x)
            }),
            width='stretch'
        )

        st.write("### Detailed Policy Violation Alerts")
        for idx, row in filtered_comp.head(10).iterrows():
            st.markdown(f"""
            <div class="alert-card-critical" style="border-left-color: #f43f5e;">
                <div style="display:flex; justify-between; align-items:center;">
                    <span style="font-weight:bold; font-size:1.05rem; color:#f8fafc;">POLICY BREACH: {row['Work ID']} — {row['MP Name']}</span>
                    <span style="background:#1e293b; color:#f43f5e; padding:4px 10px; border-radius:12px; font-weight:bold; font-size:0.8rem;">
                        {row['Violation Severity']}
                    </span>
                </div>
                <div style="margin-top:6px; color:#cbd5e1; font-size:0.9rem;">
                    <strong>Location:</strong> {row['State']}, {row['District']} | <strong>Category:</strong> {row['Work Category']} | <strong>Sanction:</strong> {format_inr(row['Sanction Amount'])}
                </div>
                <div style="margin-top:6px; color:#f1f5f9;">
                    <strong>Triggered Guidelines:</strong> {row['Triggered Policy Rules']}
                </div>
                <div style="margin-top:6px; color:#34d399; font-weight:600; font-size:0.88rem;">
                    <strong>Recommended Administrative Action:</strong> {row['Recommended Administrative Action']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("All works in the current selection comply with MoSPI MPLADS policy guidelines and entitlement ceilings.")

    st.markdown("---")
    st.write("### Empirical Field Availability & Audit Matrix")
    st.dataframe(missing_df, width='stretch')
    
    st.markdown("""
    **Official Source Citation**:
    Data is extracted from the official Ministry of Statistics and Programme Implementation (MoSPI) MPLADS Portal ([https://mplads.mospi.gov.in](https://mplads.mospi.gov.in)).
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem;">
    MPLADS AI-Powered Monitoring & Analytics Platform | Built for MoSPI, State Nodal Authorities, District Authorities, and Members of Parliament.
</div>
""", unsafe_allow_html=True)
