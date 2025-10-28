import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards
import numpy as np 
import math 

# ================== 0️⃣ CONFIGURATION ==================
st.set_page_config(
    page_title="Dr Chase Performance Report",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== 1️⃣ DATA LOADING (Dr Chase only) ==================
@st.cache_data
def load_and_enrich_dr_chase_data():
    try:
        # Load Dr_Chase_Leads (Primary Data Source)
        dr = pd.read_csv("Dr_Chase_Leads.csv", encoding='latin-1', low_memory=False)
        # Load O_Plan_Leads (for Closer Name enrichment ONLY)
        oplan = pd.read_csv("O_Plan_Leads.csv", encoding='latin-1', low_memory=False)

        # ================== 2️⃣ DATA CLEANING & ENRICHMENT ==================
        
        # --- Date & Time Conversion (for DR CHASE) ---
        dr['Modified Time'] = pd.to_datetime(dr['Modified Time'], errors='coerce', dayfirst=True)
        date_cols_dr = ["Completion Date", "Assigned date", "Approval date", "Denial Date", "Upload Date"]
        for col in date_cols_dr:
            if col in dr.columns:
                dr[col] = pd.to_datetime(dr[col], errors='coerce', dayfirst=True)

        # --- MCN Standardization ---
        for df_data in [dr, oplan]:
            if 'MCN' in df_data.columns:
                df_data['MCN'] = df_data['MCN'].astype(str).str.strip().str.upper().replace({'NAN': np.nan, '': np.nan}) 
                df_data.dropna(subset=['MCN'], inplace=True) 

        # --- Column Standardization ---
        if 'Chasing Disposition' in dr.columns:
            dr['Chasing Disposition'] = dr['Chasing Disposition'].fillna('N/A - Disposition')

        # Final Dr Chase Column Cleanup (for display simplicity)
        if 'Client' in dr.columns:
            dr['Client'] = dr['Client'].fillna('N/A - Client Missing')
            
        # 🟢 ENRICHMENT LOGIC: Transfer Closer Name to DR CHASE
        
        # 1. Prepare OPlan Closer Data (Deduplicate MCNs - keep first name)
        oplan_closer_map = oplan[['MCN', 'Closer Name']].copy()
        if 'Closer Name' in oplan_closer_map.columns:
            oplan_closer_map['Closer Name'] = oplan_closer_map['Closer Name'].astype(str).str.strip().fillna('N/A - Closer')
            # For MCNs with multiple sales, take the first Closer Name associated with it.
            oplan_closer_map.drop_duplicates(subset='MCN', keep='first', inplace=True)
            
            # 2. Enrich Dr Chase Data with Closer Name
            if 'Closer Name' in dr.columns:
                dr.drop(columns=['Closer Name'], inplace=True, errors='ignore')

            dr = pd.merge(
                dr,
                oplan_closer_map,
                on='MCN',
                how='left'
            )
            # Fill Closer Names for Dr Chase MCNs that don't match OPlan 
            dr['Closer Name'] = dr['Closer Name'].fillna('No OPlan Match') 
        
        # We only return the enriched Dr Chase data
        total_dr_rows = len(dr)
        
        return dr, total_dr_rows
        
    except Exception as e:
        st.error(f"Failed to load data files or process: {e}")
        # Returns empty DataFrame on failure
        return pd.DataFrame(), 0 

# 🔴 تحديث: نستخدم متغير واحد فقط للـ Working Data
working_df, total_dr_rows = load_and_enrich_dr_chase_data()

# ================== 4️⃣ DASHBOARD LAYOUT & TITLE ==================
st.title("📊 Dr Chase Leads Analysis (Standalone)")
st.markdown("---")

# Check if data loaded successfully
if working_df.empty:
    st.warning("Failed to load Dr Chase data. Please check file name and paths.")
    st.stop()

# NEW: Add custom CSS for general font improvements (larger base font, better legibility)
st.markdown(
    """
    <style>
    /* Global font size increase for better visibility */
    html, body {
        font-size: 16px; 
    }
    /* Improve main headers (st.title) */
    .stApp header {
        font-size: 2.5rem;
    }
    /* Custom style for subheaders (st.subheader) - making them bolder/larger */
    h2 {
        font-size: 1.8rem;
        font-weight: 600;
    }
    /* Ensure KPI labels (small text) are legible */
    div[data-testid="stMetricLabel"] > div {
        font-size: 1.1rem;
        font-weight: 500;
    }
    /* Ensure KPI values (big numbers) are prominent */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem; /* Large KPI value size */
        font-weight: bold;
    }
    /* Chart title improvement (Plotly) - handled in Plotly layout below */

    </style>
    """, unsafe_allow_html=True
)

# ================== 5️⃣ SIDEBAR FILTERS ==================
with st.sidebar:
    st.header("⚙️ Data Filters")
    
    # 1. Closer Name Filter
    closer_options = sorted(working_df['Closer Name'].astype(str).unique())
    
    target_closers = ['Aila Patrick', 'Lisa Hanz', 'Athina Henderson', 'Jordan Williams', 'Lauren Bailey', 'Linda Anderson', 'Maeve White', 'Raven Miller', 'Summer Hudson', 'Marcelle David', 'Lily Williams']
    
    default_closers = [c for c in target_closers if c in closer_options]
    if not default_closers and closer_options:
        default_closers = closer_options[:5] 
        
    
    # --- Filter Action Buttons (Closer) ---
    
    def select_default_closers():
        st.session_state['selected_closers_state'] = default_closers
        
    def select_all_closers():
        st.session_state['selected_closers_state'] = closer_options
        
    def clear_all_closers():
        st.session_state['selected_closers_state'] = []

    if 'selected_closers_state' in st.session_state:
        st.session_state['selected_closers_state'] = [
            c for c in st.session_state['selected_closers_state'] if c in closer_options
        ]
    else:
        st.session_state['selected_closers_state'] = closer_options
        
    def update_closer_selection():
        pass

    
    # Buttons for fast action
    col_closer_btn_all, col_closer_btn_default, col_closer_btn_clear = st.columns(3)
    
    with col_closer_btn_all:
        st.button("Select All", on_click=select_all_closers, use_container_width=True, key='closer_select_all_btn')
        
    with col_closer_btn_default:
        st.button("Select Default", on_click=select_default_closers, use_container_width=True, key='closer_select_default_btn')
        
    with col_closer_btn_clear:
        st.button("Clear All", on_click=clear_all_closers, use_container_width=True, key='closer_clear_all_btn')
    
    # --- Filter for Sidebar ---
    selected_closers_sidebar = st.multiselect(
        "🧑‍💼 Closer Name",
        options=closer_options,
        default=st.session_state['selected_closers_state'], 
        key='selected_closers_state',
        on_change=update_closer_selection
    )
    
    # Use expander for primary secondary filters
    with st.expander("⬇️ Advanced Filters: Disposition & Client", expanded=False):
        
        # 2. Chasing Disposition Filter
        disposition_options = sorted(working_df['Chasing Disposition'].unique())
        
        default_dispositions = disposition_options
        selected_dispositions = st.multiselect(
            "🏷️ Chasing Disposition", 
            options=disposition_options,
            default=default_dispositions
        )
        
        # 4. Client Filter
        working_df['Client'] = working_df['Client'].astype(str)
        client_options = sorted(working_df['Client'].unique())
        selected_clients = st.multiselect(
            "💼 Client", 
            options=client_options,
            default=client_options
        )
        
    # 🔴 إزالة فلاتر Opener Status و Assigned To 

    st.markdown("---")
    st.subheader("📚 Dataset Information")
    # 🔴 مؤشرات Dataset Information تعتمد الآن على ملف Dr Chase فقط
    st.metric("Total Records (Initial)", f"{total_dr_rows:,}")
    st.metric("Dr Chase MCNs (Unique)", f"{working_df['MCN'].nunique():,}")
    st.metric("Total Rows in Dashboard", f"{len(working_df):,}")


# ================== 6️⃣ APPLY FILTERS ==================
active_closers = selected_closers_sidebar


filtered_df = working_df.copy()

# Apply the active closer filter
if active_closers:
    filtered_df = filtered_df[filtered_df['Closer Name'].isin(active_closers)]

# Apply other filters
if selected_dispositions:
    filtered_df = filtered_df[filtered_df['Chasing Disposition'].isin(selected_dispositions)]
    
if selected_clients:
    filtered_df = filtered_df[filtered_df['Client'].isin(selected_clients)]


# Recalculate leads count after filtering
total_filtered_leads = len(filtered_df)

# ================== 7️⃣ KPIs (Key Metrics) ==================
st.subheader("Key Performance Indicators (KPIs)")

# Metrics derived only from the DR CHASE data (Enriched)
total_leads = len(working_df)
leads_after_filter = len(filtered_df)
# Records Chased: Since this is Dr Chase data, all filtered records are the "Chased Records"
leads_chased = leads_after_filter

# --- KPI CALCULATION ---
# KPIs based on non-null values in the DR CHASE file
if all(col in filtered_df.columns for col in ['Completion Date', 'Approval date', 'Denial Date', 'Upload Date']):
    
    filtered_completed = filtered_df['Completion Date'].notna().sum()
    filtered_approved = filtered_df['Approval date'].notna().sum()
    filtered_denied = filtered_df['Denial Date'].notna().sum()
    filtered_uploaded = filtered_df['Upload Date'].notna().sum()
else:
    filtered_completed = filtered_approved = filtered_denied = filtered_uploaded = 0

# Calculate percentages (based on Total Filtered Records)
pct_chased = (leads_chased / total_leads * 100) if total_leads > 0 else 0 
pct_completed = (filtered_completed / leads_chased * 100) if leads_chased > 0 else 0
pct_uploaded = (filtered_uploaded / leads_chased * 100) if leads_chased > 0 else 0
pct_approved = (filtered_approved / leads_chased * 100) if leads_chased > 0 else 0
pct_denied = (filtered_denied / leads_chased * 100) if leads_chased > 0 else 0


# --- KPI DISPLAY (6 columns) ---
col1, col2, col5, col6, col3, col4 = st.columns(6)

col1.metric("Total Filtered Records", f"{leads_after_filter:,}", f"out of {total_leads:,}")
col2.metric("Records Chased", f"{leads_chased:,}", f"{pct_chased:.1f}% of Initial") # Note: leads_chased is leads_after_filter

col5.metric("Approvals", f"{filtered_approved:,}", f"{pct_approved:.1f}% of Chased")
col6.metric("Denials", f"{filtered_denied:,}", f"{pct_denied:.1f}% of Chased")

col3.metric("Completed", f"{filtered_completed:,}", f"{pct_completed:.1f}% of Chased")
col4.metric("Uploaded", f"{filtered_uploaded:,}", f"{pct_uploaded:.1f}% of Chased")


# Apply custom styling to the metric cards
style_metric_cards(
    background_color="#121270",
    border_left_color="#f20045",
    box_shadow="3px 3px 10px rgba(0,0,0,0.3)"
)
st.markdown("---")

# ================== 8️⃣ CHARTS: Closer vs Disposition (THE CORE ANALYSIS) ==================

PLOTLY_FONT_SIZE = 14

st.subheader("Distribution Analysis")

# Row 1: Closer and Disposition Counts (Side-by-side)
col_chart_1, col_chart_2 = st.columns(2)

# --- Chart 1: Closer Name Count ---
with col_chart_1:
    closer_count = filtered_df['Closer Name'].value_counts().reset_index()
    closer_count.columns = ["Closer Name", "Count"]
    
    if not closer_count.empty:
        fig1 = px.bar(
            closer_count, 
            x="Closer Name", 
            y="Count", 
            title="Total Leads by Closer Name", 
            text_auto=True,
            template='plotly_white',
            color='Closer Name',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig1.update_layout(
            font=dict(size=PLOTLY_FONT_SIZE),
            title_font=dict(size=PLOTLY_FONT_SIZE + 4)
        )
        fig1.update_xaxes(categoryorder='total descending', tickfont=dict(size=PLOTLY_FONT_SIZE))
        fig1.update_yaxes(tickfont=dict(size=PLOTLY_FONT_SIZE))
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("No data available to display Closer Name Count based on current filters.")
    

# --- Chart 2: Chasing Disposition Count ---
with col_chart_2:
    disposition_count = filtered_df['Chasing Disposition'].value_counts().reset_index()
    disposition_count.columns = ["Chasing Disposition", "Count"]
    
    if not disposition_count.empty:
        fig2 = px.bar(
            disposition_count, 
            x="Chasing Disposition", 
            y="Count", 
            title="Distribution of Chasing Dispositions (Count)", 
            text_auto=True,
            template='plotly_white',
            color='Chasing Disposition',
            color_discrete_sequence=px.colors.qualitative.Pastel # Using Pastel for consistency
        )
        fig2.update_layout(
            font=dict(size=PLOTLY_FONT_SIZE),
            title_font=dict(size=PLOTLY_FONT_SIZE + 4)
        )
        fig2.update_xaxes(categoryorder='total descending', tickfont=dict(size=PLOTLY_FONT_SIZE))
        fig2.update_yaxes(tickfont=dict(size=PLOTLY_FONT_SIZE))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No data available to display Chasing Disposition Count based on current filters.")


# --- Chart 3: Closer -> Disposition Treemap (FULL WIDTH) ---
st.markdown("### Closer Performance Breakdown")
treemap_data = filtered_df.dropna(subset=['Closer Name', 'Chasing Disposition']).copy()
if not treemap_data.empty:
    fig3 = px.treemap(
        treemap_data,
        path=[px.Constant("All Closers"), 'Closer Name', 'Chasing Disposition'],
        title="Closer Performance Breakdown by Chasing Disposition (Treemap)",
        template='plotly_white',
        color='Chasing Disposition',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig3.update_layout(
        margin = dict(t=50, l=25, r=25, b=25),
        font=dict(size=PLOTLY_FONT_SIZE + 2),
        title_font=dict(size=PLOTLY_FONT_SIZE + 4)
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("No data available to display the Treemap based on current filters.")


# --- Chart 4: Client Distribution (FULL WIDTH) ---
st.markdown("### Client Distribution Analysis")
client_count = filtered_df['Client'].value_counts().reset_index()
client_count.columns = ["Client", "Count"]

if not client_count.empty:
    fig4 = px.bar(
        client_count, 
        x="Client", 
        y="Count", 
        title="Distribution of Leads by Client (From Dr Chase)", 
        text_auto=True,
        template='plotly_white',
        color='Client',
        color_discrete_sequence=px.colors.qualitative.Pastel # Using Pastel for consistency
    )
    fig4.update_layout(
        font=dict(size=PLOTLY_FONT_SIZE),
        title_font=dict(size=PLOTLY_FONT_SIZE + 4)
    )
    fig4.update_xaxes(categoryorder='total descending', tickfont=dict(size=PLOTLY_FONT_SIZE))
    fig4.update_yaxes(tickfont=dict(size=PLOTLY_FONT_SIZE))
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.warning("No data available to display Client Distribution based on current filters.")


# --- Chart 5: Opener Status Count (REMOVED) ---

st.markdown("---")

# ================== 9️⃣ DATA TABLE PREVIEW ==================
st.subheader("📋 Filtered Dr Chase Data Preview")
data_preview_cols = ['MCN', 'Closer Name', 'Chasing Disposition', 'Client', 'Dr Chase Lead Number', 
                     'Approval date', 'Denial Date', 'Completion Date', 'Assigned date']

# Filter available columns for display
available_preview_cols = [col for col in data_preview_cols if col in filtered_df.columns]

if not filtered_df.empty:
    st.dataframe(filtered_df[available_preview_cols], use_container_width=True)
else:
    st.info("The filtered data table is empty.")

# ================== 🔟 MISSING DATA WARNING (REMOVED) ==================
