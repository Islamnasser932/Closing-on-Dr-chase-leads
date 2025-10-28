import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards
import numpy as np # <--- هذا هو السطر الذي يضمن تعريف 'np'

# ================== 0️⃣ CONFIGURATION ==================
st.set_page_config(
    page_title="Closing Report Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== 1️⃣ DATA LOADING (with appropriate encoding) ==================
@st.cache_data
def load_and_merge_data():
    try:
        # Load Dr_Chase_Leads (using latin-1 as determined previously)
        dr = pd.read_csv("Dr_Chase_Leads.csv", encoding='latin-1', low_memory=False)
        # Load O_Plan_Leads (using latin-1 as determined previously)
        oplan = pd.read_csv("O_Plan_Leads.csv", encoding='latin-1', low_memory=False)

        # ================== 2️⃣ DATA CLEANING & PREP (WITHOUT DEDUP) ==================
        
        # --- Date & Time Conversion ---
        # نحافظ على تحويل التواريخ، لكن لا نستخدمها لإزالة التكرارات
        dr['Modified Time'] = pd.to_datetime(dr['Modified Time'], errors='coerce', dayfirst=True)
        
        # Rename and Convert Date of Sale in OPLAN
        if 'Date of Sale' in oplan.columns:
            oplan.rename(columns={'Date of Sale': 'Sale Date'}, inplace=True)
            oplan['Sale Date'] = pd.to_datetime(oplan['Sale Date'], errors='coerce', dayfirst=True)
        
        date_cols_dr = ["Completion Date", "Assigned date", "Approval date", "Denial Date", "Upload Date"]
        for col in date_cols_dr:
            if col in dr.columns:
                dr[col] = pd.to_datetime(dr[col], errors='coerce', dayfirst=True)

        # --- MCN Standardization and Handling Missing MCNs ---
        for df_data in [dr, oplan]:
            if 'MCN' in df_data.columns:
                df_data['MCN'] = df_data['MCN'].astype(str).str.strip().replace({'nan': np.nan, '': np.nan})
                # لا يمكننا عمل تحليل فعال بدون MCN، لذا نحذف السجلات الناقصة في MCN فقط
                df_data.dropna(subset=['MCN'], inplace=True) 

        # 🔴 تم حذف منطق drop_duplicates بالكامل من DR و OPLAN

        # --- Column Standardization ---
        if 'Closer Name' in oplan.columns:
            oplan['Closer Name'] = oplan['Closer Name'].fillna('N/A - Closer')
        
        if 'Chasing Disposition' in dr.columns:
            dr['Chasing Disposition'] = dr['Chasing Disposition'].fillna('N/A - Disposition')

        if 'Opener Status' in oplan.columns:
            oplan['Opener Status'] = oplan['Opener Status'].fillna('N/A - Status Missing')
        
        # Clean 'Assigned To' (Opener Name)
        if 'Assigned To' in oplan.columns:
            oplan['Assigned To'] = oplan['Assigned To'].fillna('N/A - Assigned Missing')
            oplan['Assigned To'] = oplan['Assigned To'].str.replace('.', ' ', regex=False).str.title()


        # Standardize Client column using Dr Chase version
        if 'Client' in dr.columns:
            dr['Client'] = dr['Client'].fillna('N/A - Client Missing')
        
        # Drop redundant columns before merging
        if 'Client' in oplan.columns:
            oplan.drop(columns=['Client'], inplace=True, errors='ignore')
        if 'Products' in dr.columns:
            dr.drop(columns=['Products'], inplace=True, errors='ignore')


        # --- Selecting Columns for Merge ---
        # 🔴 ملاحظة: الآن، Dr Chase قد يحتوي على تكرارات لنفس الـ MCN.
        # هذا يعني أن كل صف من OPLAN قد يتطابق مع عدة صفوف من DR CHASE.
        dr_cols = ['MCN', 'Dr Chase Lead Number', 'Chasing Disposition', 'Approval date', 'Denial Date', 'Client', 'Completion Date', 'Upload Date']
        oplan_date_col = 'Sale Date' if 'Sale Date' in oplan.columns else 'Date of Sale'
        
        oplan_cols = ['MCN', 'O Plan Lead Number', 'Closer Name', 'Team Leader', 'Products', oplan_date_col, 'Opener Status', 'Assigned To']

        # ================== 3️⃣ CORE MERGE OPERATION (Allowing Duplicates) ==================
        # نستخدم how='left' للحفاظ على جميع سجلات OPLAN ومطابقتها مع جميع سجلات DR CHASE التي تحمل نفس MCN.
        merged_df = pd.merge(
            oplan[oplan_cols],
            dr[dr_cols],
            on='MCN',
            how='left',
            suffixes=('_OPLAN', '_DRCHASE')
        )

        # Fill NaN Chasing Disposition for leads not found in Dr Chase
        merged_df['Chasing Disposition'] = merged_df['Chasing Disposition'].fillna('No Chase Data (OPlan Only)')

        # 🔴 تم حذف drop_duplicates من هنا أيضًا. الآن merged_df يحتوي على جميع السجلات الناتجة عن الـ Merge.
        
        # 🔴 تصحيح: يجب علينا إخبار المستخدم بالعدد الحقيقي للسجلات في ملفات الإدخال
        total_oplan_rows = len(oplan)
        total_dr_rows = len(dr)
        
        # نستخدم merged_df كما هو لعمل التحليلات بناءً على جميع السجلات الناتجة.
        return merged_df, dr, oplan, total_oplan_rows, total_dr_rows
        
    except Exception as e:
        st.error(f"Failed to load data files or process: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0, 0 

# 🔴 تحديث: إضافة المتغيرات الجديدة الناتجة من الدالة
merged_df, dr_df, oplan_df, total_oplan_rows, total_dr_rows = load_and_merge_data()

# ================== 4️⃣ DASHBOARD LAYOUT & TITLE ==================
st.title("📊 Closer Performance Analysis")
st.markdown("---")

# Check if data loaded successfully
if merged_df.empty:
    st.warning("Failed to load or merge data. Please check file names and paths.")
    st.stop()

# 🔴 NEW: Add custom CSS for general font improvements (larger base font, better legibility)
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

# ================== 5️⃣ SIDEBAR FILTERS (IMPROVED COMPACTNESS) ==================
with st.sidebar:
    st.header("⚙️ Data Filters")
    
    # 1. Closer Name Filter
    closer_options = sorted(merged_df['Closer Name'].unique())
    
    target_closers = ['Aila Patrick', 'Lisa Hanz', 'Athina Henderson', 'Jordan Williams', 'Lauren Bailey', 'Linda Anderson', 'Maeve White', 'Raven Miller', 'Summer Hudson', 'Marcelle David', 'Lily Williams']
    
    default_closers = [c for c in target_closers if c in closer_options]
    if not default_closers and closer_options:
        default_closers = closer_options[:5] 
        
    
    # --- Filter Action Buttons (Closer) ---
    col_closer_btn1, col_closer_btn2 = st.columns(2)
    
    # Function to select the default closers
    def select_default_closers():
        st.session_state['selected_closers_state'] = default_closers
        
    # Function to clear all closers
    def clear_all_closers():
        st.session_state['selected_closers_state'] = []

    # Initialize session state with the defined default list
    if 'selected_closers_state' not in st.session_state:
        st.session_state['selected_closers_state'] = default_closers
        
    def update_closer_selection():
        pass

    
    # Buttons for fast action
    with col_closer_btn1:
        st.button("Select Default", on_click=select_default_closers, use_container_width=True)
    with col_closer_btn2:
        st.button("Clear All", on_click=clear_all_closers, use_container_width=True)
    
    # --- Filter for Sidebar ---
    selected_closers_sidebar = st.multiselect(
        "🧑‍💼 Closer Name",
        options=closer_options,
        default=st.session_state['selected_closers_state'],
        key='selected_closers_state',
        on_change=update_closer_selection
    )
    
    # Use expander for primary secondary filters
    with st.expander("⬇️ Advanced Filters: Disposition & Opener", expanded=False):
        
        # 2. Chasing Disposition Filter
        disposition_options = sorted(merged_df['Chasing Disposition'].unique())
        
        # TWEAK: Exclude 'No Chase Data (OPlan Only)' from default selection
        default_dispositions = [disp for disp in disposition_options if disp != 'No Chase Data (OPlan Only)']
        selected_dispositions = st.multiselect(
            "🏷️ Chasing Disposition", 
            options=disposition_options,
            default=default_dispositions
        )
        
        # 3. Opener Status Filter
        opener_options = sorted(merged_df['Opener Status'].unique())
        selected_openers = st.multiselect(
            "🚀 Opener Status", 
            options=opener_options,
            default=opener_options
        )
    
        # 4. Client Filter
        merged_df['Client'] = merged_df['Client'].astype(str)
        client_options = sorted(merged_df['Client'].unique())
        selected_clients = st.multiselect(
            "💼 Client", 
            options=client_options,
            default=client_options
        )
        
    # Assigned To filter
    assigned_to_options = sorted(merged_df['Assigned To'].unique())

    # --- Assigned To Action Buttons ---
    # Initialize state for Assigned To filter
    if 'selected_assigned_to_state' not in st.session_state:
        st.session_state['selected_assigned_to_state'] = assigned_to_options

    def select_all_assigned_to():
        st.session_state['selected_assigned_to_state'] = assigned_to_options
    
    def clear_all_assigned_to():
        st.session_state['selected_assigned_to_state'] = []

    with st.expander("🙋‍♂️ Filter by Assigned To (Opener)", expanded=False):
        col_op_btn1, col_op_btn2 = st.columns(2)
        with col_op_btn1:
            st.button("Select All", key="op_select_all", on_click=select_all_assigned_to, use_container_width=True)
        with col_op_btn2:
            st.button("Clear All", key="op_clear_all", on_click=clear_all_assigned_to, use_container_width=True)

        selected_assigned_to = st.multiselect(
            "Select Opener(s)", # Simplified label inside the expander
            options=assigned_to_options,
            default=st.session_state['selected_assigned_to_state'],
            key='selected_assigned_to_state'
        )


    st.markdown("---")
    st.subheader("📚 Dataset Information")
    # 🔴 تحديث: استخدام المتغيرات الجديدة التي تحمل أعداد الصفوف الأصلية
    st.metric("Total OPlan Records (Initial)", f"{total_oplan_rows:,}")
    st.metric("Total Dr Chase Records (Initial)", f"{total_dr_rows:,}")
    st.metric("Total Merged Records (All Matches)", f"{len(merged_df):,}")


# ================== 6️⃣ APPLY FILTERS ==================
active_closers = selected_closers_sidebar


filtered_df = merged_df.copy()

# Apply the active closer filter
if active_closers:
    filtered_df = filtered_df[filtered_df['Closer Name'].isin(active_closers)]

# Apply other filters
if selected_assigned_to:
    filtered_df = filtered_df[filtered_df['Assigned To'].isin(selected_assigned_to)]

if selected_dispositions:
    filtered_df = filtered_df[filtered_df['Chasing Disposition'].isin(selected_dispositions)]
    
if selected_openers:
    filtered_df = filtered_df[filtered_df['Opener Status'].isin(selected_openers)]

if selected_clients:
    filtered_df = filtered_df[filtered_df['Client'].isin(selected_clients)]


# Recalculate leads count after filtering
total_filtered_leads = len(filtered_df)

# ================== 7️⃣ KPIs (Key Metrics) ==================
st.subheader("Key Performance Indicators (KPIs)")

# Metrics derived from the MERGED (OPlan/Dr Chase) data
total_leads = len(merged_df)
leads_after_filter = len(filtered_df)
# leads_chased: عدد السجلات المفلترة التي لها بيانات متابعة في Dr Chase
leads_chased = filtered_df[filtered_df['Chasing Disposition'] != 'No Chase Data (OPlan Only)'].shape[0]

# --- KPI CALCULATION ---
# 🔴 ملاحظة: بما أننا لم نقم بإزالة التكرارات، فإن حساب KPIs الآن يعتمد على عدد السجلات وليس عدد الـ MCNs الفريدة.
# نحتاج إلى إيجاد الـ MCNs الفريدة في DR CHASE ثم مطابقتها مع السجلات المفلترة في merged_df
if all(col in dr_df.columns for col in ['Completion Date', 'Upload Date', 'Approval date', 'Denial Date']):
    
    # نحدد الـ MCNs الفريدة التي حققت حالة معينة في DR CHASE
    completed_mcns = dr_df.dropna(subset=['Completion Date'])['MCN'].unique()
    uploaded_mcns = dr_df.dropna(subset=['Upload Date'])['MCN'].unique()
    approved_mcns = dr_df.dropna(subset=['Approval date'])['MCN'].unique()
    denied_mcns = dr_df.dropna(subset=['Denial Date'])['MCN'].unique()
    
    # ثم نحسب عدد السجلات في filtered_df التي تتطابق مع هذه الـ MCNs
    # بما أننا لم نحذف التكرارات في filtered_df، فإن السجلات المكررة ستحسب أكثر من مرة إذا كان لها MCN مطابق
    filtered_completed = filtered_df[filtered_df['MCN'].isin(completed_mcns)].shape[0]
    filtered_uploaded = filtered_df[filtered_df['MCN'].isin(uploaded_mcns)].shape[0]
    filtered_approved = filtered_df[filtered_df['MCN'].isin(approved_mcns)].shape[0]
    filtered_denied = filtered_df[filtered_df['MCN'].isin(denied_mcns)].shape[0]
else:
    filtered_completed = 0
    filtered_uploaded = 0
    filtered_approved = 0
    filtered_denied = 0

# Calculate percentages (based on chased leads for status KPIs)
pct_chased = (leads_chased / leads_after_filter * 100) if leads_after_filter > 0 else 0
# 🔴 ملاحظة: إذا كان الهدف هو نسبة حالة السجل من إجمالي السجلات التي تم تتبعها
pct_completed = (filtered_completed / leads_chased * 100) if leads_chased > 0 else 0
pct_uploaded = (filtered_uploaded / leads_chased * 100) if leads_chased > 0 else 0
pct_approved = (filtered_approved / leads_chased * 100) if leads_chased > 0 else 0
pct_denied = (filtered_denied / leads_chased * 100) if leads_chased > 0 else 0

# --- KPI DISPLAY (6 columns) ---
col1, col2, col5, col6, col3, col4 = st.columns(6)

col1.metric("Total Filtered Records", f"{leads_after_filter:,}", f"out of {total_leads:,}")
col2.metric("Records Chased", f"{leads_chased:,}", f"{pct_chased:.1f}% of Filtered")

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
            title="Total Leads by Closer Name (All Records)", # 🔴 تحديث العنوان
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
        color_discrete_sequence=px.colors.qualitative.Pastel
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


# --- Chart 5: Opener Status Count (Full Width) ---
st.subheader("Opener Status Distribution")
opener_count = filtered_df['Opener Status'].value_counts().reset_index()
opener_count.columns = ["Opener Status", "Count"]

if not opener_count.empty:
    fig5 = px.bar(
        opener_count, 
        x="Opener Status", 
        y="Count", 
        title="Leads by Opener Status (Count)", 
        text_auto=True,
        template='plotly_white',
        color='Opener Status',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig5.update_layout(
        font=dict(size=PLOTLY_FONT_SIZE),
        title_font=dict(size=PLOTLY_FONT_SIZE + 4)
    )
    fig5.update_xaxes(categoryorder='total descending', tickfont=dict(size=PLOTLY_FONT_SIZE))
    fig5.update_yaxes(tickfont=dict(size=PLOTLY_FONT_SIZE))
    st.plotly_chart(fig5, use_container_width=True)
else:
    st.info("No data available to display Opener Status Distribution based on current filters.")

st.markdown("---")

# ================== 9️⃣ DATA TABLE PREVIEW ==================
st.subheader("📋 Merged and Filtered Data Preview")
# FINAL FIX: Using the simplified list of columns
data_preview_cols = ['MCN', 'Closer Name', 'Opener Status', 'Chasing Disposition', 'Products', 
                     'Approval date', 'Denial Date', 'Client', 'Assigned To']

if not filtered_df.empty:
    st.dataframe(filtered_df[data_preview_cols], use_container_width=True)
else:
    st.info("The filtered data table is empty.")
