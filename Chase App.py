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
    # 🔴 تحديث: تغيير page_title
    page_title="Closing Analysis on Dr chase Leads",
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
        date_cols_dr = ["Completion Date", "Assigned date", "Approval date", "Denial Date", "Upload Date", "Date of Sale", "Created Time"]
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

        # 🟢 NEW: Clean Dr Specialty Column
        if 'Dr Specialty' in dr.columns:
            dr['Dr Specialty'] = dr['Dr Specialty'].astype(str).str.strip().fillna('N/A Specialty')
        
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
# 🔴 تحديث: تغيير عنوان الداشبورد الرئيسي
st.title("📊 Closing Analysis on Dr chase Leads")
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
total_leads = int(len(working_df)) # Cast to int
leads_after_filter = len(filtered_df)
# Records Chased: Since this is Dr Chase data, all filtered records are the "Chased Records"
leads_chased = int(leads_after_filter) # Cast to int

# --- KPI CALCULATION ---
# KPIs based on non-null values in the DR CHASE file
if all(col in filtered_df.columns for col in ['Completion Date', 'Approval date', 'Denial Date', 'Upload Date']):
    
    # 🔴 FIX: Cast results to int
    filtered_completed = int(filtered_df['Completion Date'].notna().sum())
    filtered_approved = int(filtered_df['Approval date'].notna().sum())
    filtered_denied = int(filtered_df['Denial Date'].notna().sum())
    filtered_uploaded = int(filtered_df['Upload Date'].notna().sum())
else:
    filtered_completed = filtered_approved = filtered_denied = filtered_uploaded = 0

# Calculate percentages (based on Total Filtered Records)
pct_chased = (leads_chased / total_leads * 100) if total_leads > 0 else 0 
pct_completed = (filtered_completed / leads_chased * 100) if leads_chased > 0 else 0
pct_uploaded = (filtered_uploaded / leads_chased * 100) if leads_chased > 0 else 0
pct_approved = (filtered_approved / leads_chased * 100) if leads_chased > 0 else 0
pct_denied = (filtered_denied / leads_chased * 100) if leads_chased > 0 else 0


# --- KPI DISPLAY (5 columns) ---
col2, col5, col6, col3, col4 = st.columns(5) 

# 🔴 FIX: وضع دلتا (Total Leads) لتصحيح شكل البطاقة (using native Python int now)
col2.metric("Records Chased", f"{leads_chased:,}", f"Total: {total_leads:,}") 

col5.metric("Approvals", f"{filtered_approved:,}", f"{pct_approved:.1f}% of Chased")
col6.metric("Denials", f"{filtered_denied:,}", f"{pct_denied:.1f}% of Chased")

col3.metric("Completed", f"{filtered_completed:,}", f"{pct_completed:.1f}% of Chased")
col4.metric("Uploaded", f"{filtered_uploaded:,}", f"{pct_uploaded:.1f}% of Chased")


# Apply custom styling to the metric cards
style_metric_cards(
    background_color="#1F2630", 
    border_left_color="#00C49F", 
    border_radius_px=10,
    border_size_px=2,
    box_shadow="0 4px 12px rgba(0, 196, 159, 0.2)" 
)
st.markdown("---")

# ================== 8️⃣ CHARTS: Closer vs Disposition (THE CORE ANALYSIS) ==================

PLOTLY_FONT_SIZE = 14

st.subheader("Distribution Analysis")

# 🔴 التشخيص: عدد الصفوف النشطة (للتأكد من أن الفلاتر تعمل)
st.info(f"Active Leads (after all filters): {len(filtered_df):,} rows.")


# ---------------------------------------------------------------------------------------
# 🔴 ROW 1: Closer Name Bar Chart + Closer Summary Table (Side-by-Side)
# ---------------------------------------------------------------------------------------
col_closer_chart, col_closer_summary = st.columns([3, 2]) 

# --- Data Prep for Closer Charts/Text ---
closer_count = filtered_df['Closer Name'].value_counts().reset_index()
closer_count.columns = ["Closer Name", "Count"]
total_closer_count = int(closer_count['Count'].sum()) # 🔴 FIX 2: Define total_closer_count here
closer_count['Percentage'] = (closer_count['Count'] / total_closer_count * 100).round(1)

# --- LEFT COLUMN (Chart 1: Total Leads by Closer Name - Bar Chart) ---
with col_closer_chart:
    if not closer_count.empty:
        # 🟢 Chart 1: Bar Chart (Reverted from Pie Chart)
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

# --- RIGHT COLUMN (Closer Info Summary Table) ---
with col_closer_summary:
    st.markdown("### ℹ️ Closer Performance Overview")
    
    # 🟢 عرض جدول أداء كل Closer بالضبط كما طلب المستخدم
    if not closer_count.empty:
        # 🔴 FINAL FIX: Cast columns to native Python types before st.dataframe
        closer_count['Count'] = closer_count['Count'].astype(int)
        closer_count['Percentage'] = closer_count['Percentage'].astype(float)
        
        st.dataframe(
            closer_count,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Closer Name": "Closer",
                "Count": st.column_config.ProgressColumn(
                    "Count",
                    help="Total records for this closer",
                    format="%d",
                    min_value=0,
                    max_value=total_closer_count, # 🔴 FIX: Now defined as int
                ),
                "Percentage": st.column_config.NumberColumn(
                    "Share (%)",
                    format="%.1f%%",
                )
            }
        )
    else:
        st.info("Select closers to view the distribution summary.")

# ---------------------------------------------------------------------------------------
# 🔴 ROW 2: Disposition Bar Chart + Disposition Summary Table (Side-by-Side)
# ---------------------------------------------------------------------------------------
st.markdown("---")
st.markdown("### Status Distribution Breakdown")
col_dispo_chart, col_dispo_table = st.columns([3, 2])

# --- LEFT COLUMN (Chart 2: Chasing Disposition Count - Bar Chart) ---
with col_dispo_chart:
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
            color_discrete_sequence=px.colors.qualitative.Pastel
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


# --- RIGHT COLUMN (Disposition Summary Table - Side-by-Side) ---
with col_dispo_table:
    st.markdown("### 📊 Chasing Disposition Summary Table")
    
    disposition_summary = filtered_df['Chasing Disposition'].value_counts().reset_index(name='Count')
    disposition_summary.columns = ['Disposition', 'Count']
    
    # 🔴 FIX: Convert sum result to int before using in ProgressColumn max_value
    total_records = int(disposition_summary['Count'].sum())
    disposition_summary['Percentage'] = (disposition_summary['Count'] / total_records * 100).round(1).astype(float) # Cast to float
    disposition_summary['Count'] = disposition_summary['Count'].astype(int) # Cast Count

    # عرض ملخص الداتا في جدول (أفضل 10 حالات)
    st.dataframe(
        disposition_summary.head(10), 
        use_container_width=True,
        hide_index=True,
        column_config={
            "Count": st.column_config.ProgressColumn(
                "Count",
                help="Total records for this disposition",
                format="%d",
                min_value=0,
                max_value=total_records,
            ),
            "Percentage": st.column_config.NumberColumn(
                "%",
                format="%.1f%%",
            )
        }
    )


# --- Chart 3: Closer -> Disposition Treemap (FULL WIDTH) ---
st.markdown("### Closer Performance Breakdown")
treemap_data = filtered_df.dropna(subset=['Closer Name', 'Chasing Disposition']).copy()

# 🟢 NEW LOGIC: Calculate percentages for display in the Treemap
treemap_data_agg = treemap_data.groupby(['Closer Name', 'Chasing Disposition']).size().reset_index(name='Count')
closer_totals_agg = treemap_data_agg.groupby('Closer Name')['Count'].sum().reset_index(name='Closer Total')
treemap_data_agg = pd.merge(treemap_data_agg, closer_totals_agg, on='Closer Name', how='left')
treemap_data_agg['Percentage'] = (treemap_data_agg['Count'] / treemap_data_agg['Closer Total'])
# 🔴 FIX: إزالة <br> لعرض النص في سطر واحد
treemap_data_agg['Custom Label'] = treemap_data_agg.apply(
    lambda row: f"{row['Count']:,} ({row['Percentage'] * 100:.1f}%)", axis=1
)

if not treemap_data_agg.empty:
    fig3 = px.treemap(
        treemap_data_agg,
        path=[px.Constant("All Closers"), 'Closer Name', 'Chasing Disposition'],
        values='Count', # Use the pre-calculated count
        title="Closer Performance Breakdown by Chasing Disposition (Treemap)",
        template='plotly_white',
        color='Chasing Disposition',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        
        # 🔴 Set custom data and update traces for labels/hover
        custom_data=['Count', 'Closer Name', 'Chasing Disposition', 'Percentage'],
    )
    
    fig3.update_traces(
        # 🔴 FIX: استخدام textinfo="label" لعرض اسم التصنيف فقط (Dr Denied)
        textinfo="label", 
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Closer %: %{customdata[3]:.1%}<extra></extra>'
    )
    
    fig3.update_layout(
        margin = dict(t=50, l=25, r=25, b=25),
        font=dict(size=PLOTLY_FONT_SIZE + 2),
        title_font=dict(size=PLOTLY_FONT_SIZE + 4)
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("No data available to display the Treemap based on current filters.")


# --- Chart 4: Closer-Specific Disposition Ranking (NEW SECTION) ---
st.markdown("---")
st.subheader("🕵️ Closer-Specific Disposition Ranking")

closer_list = sorted(filtered_df['Closer Name'].unique())

if not closer_list:
    st.info("No closers available for individual breakdown.")
else:
    selected_closer = st.selectbox(
        "Select Closer to Analyze Disposition Ranking:", 
        options=closer_list, 
        key="closer_ranking_select"
    )
    
    # Filter data for the selected closer
    closer_data = filtered_df[filtered_df['Closer Name'] == selected_closer].copy()
    
    if not closer_data.empty:
        # Aggregate dispositions for the selected closer
        closer_dispo_ranking = closer_data['Chasing Disposition'].value_counts().reset_index()
        closer_dispo_ranking.columns = ['Disposition', 'Count']
        
        # 🔴 NEW LOGIC: Calculate Percentage Share for the selected Closer
        total_closer_records_ranking = int(closer_dispo_ranking['Count'].sum()) # Cast to int
        
        # Cast columns to native Python types for JSON compatibility
        closer_dispo_ranking['Percentage'] = (closer_dispo_ranking['Count'] / total_closer_records_ranking * 100).round(1).astype(float) 
        closer_dispo_ranking['Count'] = closer_dispo_ranking['Count'].astype(int) 
        
        # 🟢 Display as Streamlit DataFrame (Table)
        st.dataframe(
            closer_dispo_ranking,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Disposition": "Chasing Disposition",
                "Count": st.column_config.ProgressColumn(
                    "Count",
                    format="%d",
                    min_value=0,
                    max_value=total_closer_records_ranking,
                ),
                "Percentage": st.column_config.NumberColumn(
                    "Share (%)",
                    format="%.1f%%",
                )
            }
        )
        st.info(f"Total filtered records for {selected_closer}: {total_closer_records_ranking:,}")
    else:
        st.info(f"No records found for {selected_closer} under current filters.")


# --- Chart 5: Closer -> Specialty -> Disposition Treemap (REPLACED) ---
st.markdown("---")
st.subheader("🏥 Dr Specialty Performance by Disposition")

# 1. Closer Filter for the chart (New selectbox)
closer_list_all = sorted(filtered_df['Closer Name'].unique())

if not closer_list_all:
    st.info("No data available for this breakdown.")
else:
    closer_filter_5 = st.selectbox(
        "Select Closer to Analyze Specialty Breakdown:", 
        options=closer_list_all, 
        key="specialty_closer_filter"
    )
    
    # Filter data based on selected closer
    specialty_filtered_df = filtered_df[
        filtered_df['Closer Name'] == closer_filter_5
    ].copy()
    
    # 2. Aggregate Data (Specialty vs. Disposition)
    specialty_dispo_count = specialty_filtered_df.groupby(
        ['Dr Specialty', 'Chasing Disposition']
    ).size().reset_index(name='Count')
    
    if not specialty_dispo_count.empty:
        # 3. Create Stacked Bar Chart
        fig_specialty = px.bar(
            specialty_dispo_count,
            x='Dr Specialty',
            y='Count',
            color='Chasing Disposition',
            title=f"Disposition Breakdown by Specialty for {closer_filter_5}",
            template='plotly_white',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_specialty.update_layout(
            xaxis_title="Doctor Specialty",
            yaxis_title="Count of Records",
            barmode='stack',
            font=dict(size=PLOTLY_FONT_SIZE)
        )
        st.plotly_chart(fig_specialty, use_container_width=True)
    else:
        st.info(f"No records found for {closer_filter_5} in this combination.")


# --- Chart 6: Client Distribution (FULL WIDTH) ---
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


# --- Time Series Analysis Section (Chart 7) ---
st.markdown("---")
st.subheader("📈 Key Activity Time Series Analysis")

# 1. Define Available Date Columns
date_cols_analysis = [
    "Created Time", "Date of Sale", "Assigned date", "Completion Date", "Approval date", "Denial Date", "Upload Date"
]
available_date_cols = [c for c in date_cols_analysis if c in working_df.columns]

if not available_date_cols:
    st.info("⚠️ No primary date columns found for time series analysis.")
else:
    # User Selection of Time Column
    time_col = st.selectbox("Select time dimension:", available_date_cols, key="ts_time_col")

    # 🔴 NEW DEBUGGING: Show valid date count
    valid_count = filtered_df[time_col].notna().sum()
    st.markdown(f"**Valid Records for {time_col}: {valid_count:,}**") 
    
    # Filter data for records with valid selected date
    df_ts = filtered_df[filtered_df[time_col].notna()].copy()
    
    if df_ts.empty:
        st.info(f"No valid records to plot using '{time_col}' (Count: {valid_count:,}). Try another dimension.")
    else:
        # Aggregation Frequency Selection
        freq = st.radio("Time aggregation level:", ["Daily", "Weekly", "Monthly"], horizontal=True, key="ts_freq_radio")

        # Grouping/Breakdown Selection
        group_by_col = st.selectbox("Breakdown series by:", ["None", "Closer Name", "Chasing Disposition", "Client"], key="ts_group_select")
        
        # Calculate period (using pandas offset strings)
        freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
        period_offset = freq_map[freq]
        
        # Create the time period column
        df_ts['Period'] = df_ts[time_col].dt.to_period(period_offset).dt.to_timestamp()
        
        if group_by_col == "None":
            ts_data = df_ts.groupby('Period').size().reset_index(name='Count')
            color_col = None
        else:
            ts_data = df_ts.groupby(['Period', group_by_col]).size().reset_index(name='Count')
            color_col = group_by_col
            
        if not ts_data.empty:
            # Generate Plotly Line Chart
            fig_ts = px.line(
                ts_data,
                x='Period',
                y='Count',
                color=color_col,
                title=f"Activity Trend: {time_col} ({freq} Count)",
                template='plotly_white',
                labels={'Count': 'Number of Records', 'Period': 'Time Period'},
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            # Update layout for better time axis visibility
            fig_ts.update_layout(
                xaxis_title=time_col,
                yaxis_title="Record Count",
                hovermode="x unified",
                font=dict(size=PLOTLY_FONT_SIZE)
            )
            
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info("No data available after grouping and time aggregation.")


st.markdown("---")

# ================== 9️⃣ DATA TABLE PREVIEW ==================
st.subheader("📋 Filtered Dr Chase Data Preview")
data_preview_cols = ['MCN', 'Closer Name', 'Chasing Disposition', 'Client', 'Dr Chase Lead Number', 
                     'Approval date', 'Denial Date', 'Completion Date', 'Assigned date', 'Dr Specialty']

# Filter available columns for display
available_preview_cols = [col for col in data_preview_cols if col in filtered_df.columns]

if not filtered_df.empty:
    st.dataframe(filtered_df[available_preview_cols], use_container_width=True)
else:
    st.info("The filtered data table is empty.")

# ================== 🔟 MISSING DATA WARNING (REMOVED) ==================
