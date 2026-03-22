import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from prophet import Prophet

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="FitPulse ML Dashboard",
    page_icon="💓",
    layout="wide"
)

# -------------------------------------------------------
# GLOBAL DARK THEME CSS
# -------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

* { box-sizing: border-box; }

.stApp {
    background: #060d1a;
    color: #e2e8f0;
    font-family: 'Syne', sans-serif;
}

[data-testid="stSidebar"] {
    background: #080f1e !important;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #38bdf8 !important; }
[data-testid="stSidebar"] .stRadio label { cursor: pointer; }

h1 { color: #f0f9ff !important; font-family: 'Syne', sans-serif; font-weight: 800; letter-spacing: -1px; }
h2 { color: #38bdf8 !important; font-family: 'Syne', sans-serif; font-weight: 700; }
h3 { color: #7dd3fc !important; font-family: 'Syne', sans-serif; }

[data-baseweb="tab-list"] {
    background: #0c1829 !important;
    border-radius: 12px !important;
    padding: 6px !important;
    gap: 4px !important;
    border: 1px solid #1e3a5f !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: #64748b !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 10px 20px !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
[data-baseweb="tab"]:hover {
    background: #1e3a5f !important;
    color: #93c5fd !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: linear-gradient(135deg, #1d4ed8, #0ea5e9) !important;
    color: #fff !important;
    box-shadow: 0 4px 15px rgba(14, 165, 233, 0.35) !important;
}
[data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 24px !important;
}

[data-testid="metric-container"] {
    background: #0c1829 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700;
}

[data-testid="stDataFrame"] {
    border: 1px solid #1e3a5f !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #0ea5e9) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    padding: 10px 28px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(14, 165, 233, 0.5) !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #0c1829 !important;
    border: 2px dashed #1e3a5f !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #38bdf8 !important;
}

.stSuccess { background: #052e16 !important; border: 1px solid #15803d !important; border-radius: 10px !important; color: #86efac !important; }
.stInfo    { background: #082f49 !important; border: 1px solid #0369a1 !important; border-radius: 10px !important; color: #7dd3fc !important; }
.stWarning { background: #1c1000 !important; border: 1px solid #a16207 !important; border-radius: 10px !important; color: #fde047 !important; }

hr { border-color: #1e3a5f !important; }

.section-card {
    background: #0c1829;
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
}
.step-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1d4ed8, #0ea5e9);
    color: white;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 10px;
    letter-spacing: 1px;
}
.page-header {
    background: linear-gradient(135deg, #0c1829 0%, #0f2040 100%);
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 32px 36px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(14,165,233,0.12), transparent 70%);
    border-radius: 50%;
}
.page-header h1 { margin: 0 !important; font-size: 2rem !important; }
.page-header p  { color: #64748b; margin-top: 8px; font-size: 14px; }

.graph-title {
    color: #38bdf8;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 15px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# DARK MATPLOTLIB HELPER
# -------------------------------------------------------
def apply_dark_style(fig, axes_list=None):
    DARK_BG  = "#0c1829"
    GRID_CLR = "#1e3a5f"
    TEXT_CLR = "#94a3b8"
    ACCENT   = "#38bdf8"

    fig.patch.set_facecolor(DARK_BG)

    if axes_list is None:
        axes_list = fig.get_axes()
    if not isinstance(axes_list, (list, np.ndarray)):
        axes_list = [axes_list]

    for ax in axes_list:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TEXT_CLR, labelsize=8)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color(ACCENT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_CLR)
        ax.grid(color=GRID_CLR, alpha=0.5, linewidth=0.6)
    return fig


PALETTE = ["#38bdf8", "#818cf8", "#34d399", "#fb923c", "#f472b6", "#a78bfa"]


# -------------------------------------------------------
# SIDEBAR — PROFESSIONAL WITH EMOJIS
# -------------------------------------------------------
st.sidebar.markdown("""
<div style='text-align:center; padding: 20px 0 12px 0;'>
    <div style='font-size:2.8rem; line-height:1;'>💓</div>
    <div style='font-family: Syne, sans-serif; font-size:1.25rem; font-weight:800;
                color:#f0f9ff; letter-spacing:-0.5px; margin-top:8px;'>FitPulse</div>
    <div style='font-size:10px; color:#475569; letter-spacing:2.5px;
                text-transform:uppercase; margin-top:4px;'>ML Dashboard</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

milestone = st.sidebar.radio(
    "🚀 Select Milestone",
    ("📊  Data Processing Pipeline", "🤖  ML Analytics Pipeline")
)

st.sidebar.markdown("---")

# ── Navigation Section ───────────────────────────────
st.sidebar.markdown("""
<div style='font-size:11px; color:#38bdf8; font-weight:700;
            letter-spacing:2px; text-transform:uppercase; margin-bottom:10px;'>
    🗺️ &nbsp; Navigation
</div>
""", unsafe_allow_html=True)

nav_items = [
    ("📂", "Data Upload",     "Load your CSV dataset"),
    ("🔍", "Null Analysis",   "Detect & visualize missing data"),
    ("🧹", "Preprocessing",   "Clean & interpolate records"),
    ("👁️", "Data Preview",    "Inspect cleaned dataset"),
    ("📊", "EDA",             "Distribution & stats plots"),
]
for icon, title, desc in nav_items:
    st.sidebar.markdown(f"""
    <div style='background:#0c1829; border:1px solid #1e3a5f; border-radius:8px;
                padding:8px 12px; margin-bottom:6px;'>
        <span style='font-size:14px;'>{icon}</span>
        <span style='color:#e2e8f0; font-size:12px; font-weight:600; margin-left:6px;'>{title}</span>
        <div style='color:#475569; font-size:10px; margin-top:2px; margin-left:22px;'>{desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")

# ── ML Modules Section ───────────────────────────────
st.sidebar.markdown("""
<div style='font-size:11px; color:#818cf8; font-weight:700;
            letter-spacing:2px; text-transform:uppercase; margin-bottom:10px;'>
    🤖 &nbsp; ML Modules
</div>
""", unsafe_allow_html=True)

ml_items = [
    ("❤️",  "Heart Rate Processing", "Resample to minute level"),
    ("🔬",  "TSFresh Features",      "Auto time-series extraction"),
    ("🔥",  "Feature Heatmap",       "Normalized feature matrix"),
    ("📈",  "HR Forecasting",        "30-day Prophet prediction"),
    ("🔵",  "KMeans Clustering",     "K=3 user segmentation"),
    ("📉",  "PCA Projection",        "2D cluster visualization"),
    ("🧠",  "t-SNE Comparison",      "KMeans vs DBSCAN view"),
]
for icon, title, desc in ml_items:
    st.sidebar.markdown(f"""
    <div style='background:#0c1829; border:1px solid #1e3a5f; border-radius:8px;
                padding:8px 12px; margin-bottom:6px;'>
        <span style='font-size:14px;'>{icon}</span>
        <span style='color:#e2e8f0; font-size:12px; font-weight:600; margin-left:6px;'>{title}</span>
        <div style='color:#475569; font-size:10px; margin-top:2px; margin-left:22px;'>{desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")

# ── System Status Section ────────────────────────────
st.sidebar.markdown("""
<div style='font-size:11px; color:#34d399; font-weight:700;
            letter-spacing:2px; text-transform:uppercase; margin-bottom:10px;'>
    ⚡ &nbsp; System Status
</div>
""", unsafe_allow_html=True)

status_items = [
    ("🟢", "Pipeline",  "Active"),
    ("🟢", "ML Engine", "Ready"),
    ("🟢", "Prophet",   "Loaded"),
    ("🟢", "TSFresh",   "Loaded"),
    ("🟡", "GPU",       "CPU Mode"),
]
for dot, name, val in status_items:
    st.sidebar.markdown(f"""
    <div style='display:flex; justify-content:space-between; align-items:center;
                padding:5px 8px; margin-bottom:4px;'>
        <span style='font-size:11px; color:#94a3b8;'>{dot} {name}</span>
        <span style='font-size:10px; color:#475569; font-family:monospace;'>{val}</span>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align:center; padding:8px 0; color:#475569; font-size:10px;'>
    🛠️ Built with Streamlit &nbsp;·&nbsp; v2.0<br>
    💡 FitPulse Analytics Platform
</div>
""", unsafe_allow_html=True)


# =====================================================
# MILESTONE 1
# =====================================================
def milestone1():
    if "df" not in st.session_state:
        st.session_state.df = None
    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = None

    st.markdown("""
    <div class='page-header'>
        <h1>💙 Data Processing Pipeline</h1>
        <p>📂 Upload &nbsp;·&nbsp; 🔍 Analyse &nbsp;·&nbsp; 🧹 Clean &nbsp;·&nbsp; 📊 Explore</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📂  Upload",
        "🔍  Null Analysis",
        "🧹  Preprocess",
        "👁️  Preview",
        "📊  EDA"
    ])

    # ── TAB 1 · UPLOAD ────────────────────────────────
    with tab1:
        st.markdown("<div class='step-badge'>STEP 01 — DATA INGESTION</div>", unsafe_allow_html=True)
        st.markdown("## 📂 Upload Dataset")
        st.markdown("<p style='color:#64748b;'>Upload your Fitbit / health CSV dataset to begin the pipeline.</p>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df

            rows, cols  = df.shape
            total_nulls = df.isnull().sum().sum()
            dup_rows    = df.duplicated().sum()

            st.success("✅ Dataset Loaded Successfully!")
            st.markdown("<br>", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📋 Rows",        f"{rows:,}")
            c2.metric("📐 Columns",     cols)
            c3.metric("⚠️ Total Nulls", f"{total_nulls:,}")
            c4.metric("🔁 Duplicates",  f"{dup_rows:,}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🗂️ Raw Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📋 Column Types")
            dtype_df = pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values})
            st.dataframe(dtype_df, use_container_width=True)

    # ── TAB 2 · NULL ANALYSIS ─────────────────────────
    with tab2:
        st.markdown("<div class='step-badge'>STEP 02 — MISSING VALUE ANALYSIS</div>", unsafe_allow_html=True)
        st.markdown("## 🔍 Missing Value Analysis")

        if st.session_state.df is None:
            st.warning("⬆️ Please upload a dataset in the Upload tab first.")
        else:
            df           = st.session_state.df
            null_counts  = df.isnull().sum()
            null_percent = (null_counts / len(df)) * 100

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📋 Null Counts per Column")
            null_df = pd.DataFrame({
                "Column":     null_counts.index,
                "Null Count": null_counts.values,
                "Null %":     null_percent.round(2).values
            }).sort_values("Null Count", ascending=False)
            st.dataframe(null_df, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📊 Missing Data Percentage — Bar Chart")
            bars   = null_percent.sort_values(ascending=False)
            colors = [PALETTE[i % len(PALETTE)] for i in range(len(bars))]

            fig, ax = plt.subplots(figsize=(10, 4))
            apply_dark_style(fig, ax)
            bars.plot(kind="bar", ax=ax, color=colors, edgecolor="none")
            ax.set_ylabel("Missing %", fontsize=9)
            ax.set_xlabel("")
            ax.set_title("Missing Data % per Column", fontsize=11)
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    # ── TAB 3 · PREPROCESS ────────────────────────────
    with tab3:
        st.markdown("<div class='step-badge'>STEP 03 — PREPROCESSING</div>", unsafe_allow_html=True)
        st.markdown("## 🧹 Data Preprocessing")

        if st.session_state.df is None:
            st.warning("⬆️ Please upload a dataset first.")
        else:
            st.markdown("""
            <div class='section-card'>
                <div style='color:#64748b; font-size:13px;'>The preprocessing pipeline applies:</div>
                <ul style='color:#94a3b8; font-size:13px; margin-top:8px;'>
                    <li>🗓️ Date column parsing to <code style='color:#38bdf8;'>datetime</code></li>
                    <li>📈 Per-user interpolation for numeric health metrics</li>
                    <li>↕️ Forward fill + backward fill for remaining nulls</li>
                    <li>🏋️ Workout type null filling → <code style='color:#38bdf8;'>No Workout</code></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            if st.button("▶ Run Preprocessing"):
                df = st.session_state.df.copy()

                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

                numeric_cols = ["Hours_Slept", "Water_Intake (Liters)", "Active_Minutes", "Heart_Rate (bpm)"]
                numeric_cols = [c for c in numeric_cols if c in df.columns]

                if "User_ID" in df.columns and numeric_cols:
                    df[numeric_cols] = df.groupby("User_ID")[numeric_cols].transform(
                        lambda x: x.interpolate()
                    )
                    df[numeric_cols] = df.groupby("User_ID")[numeric_cols].transform(
                        lambda x: x.ffill().bfill()
                    )

                if "Workout_Type" in df.columns:
                    df["Workout_Type"] = df["Workout_Type"].fillna("No Workout")

                st.session_state.cleaned_df = df
                st.success("✅ Preprocessing Completed!")

                st.markdown("<br>", unsafe_allow_html=True)
                before_nulls = st.session_state.df.isnull().sum().sum()
                after_nulls  = df.isnull().sum().sum()
                c1, c2 = st.columns(2)
                c1.metric("🔴 Nulls Before", before_nulls)
                c2.metric("🟢 Nulls After",  after_nulls,
                          delta=int(after_nulls - before_nulls))

    # ── TAB 4 · PREVIEW ───────────────────────────────
    with tab4:
        st.markdown("<div class='step-badge'>STEP 04 — CLEAN DATA PREVIEW</div>", unsafe_allow_html=True)
        st.markdown("## 👁️ Clean Data Preview")

        if st.session_state.cleaned_df is None:
            st.warning("⬆️ Run preprocessing first in the Preprocess tab.")
        else:
            df = st.session_state.cleaned_df

            c1, c2, c3 = st.columns(3)
            c1.metric("📋 Rows",            f"{df.shape[0]:,}")
            c2.metric("📐 Columns",         df.shape[1])
            c3.metric("✅ Remaining Nulls", df.isnull().sum().sum())

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🗂️ Cleaned Dataset")
            st.dataframe(df.head(20), use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📊 Descriptive Statistics")
            st.dataframe(df.describe().round(2), use_container_width=True)

    # ── TAB 5 · EDA ───────────────────────────────────
    with tab5:
        st.markdown("<div class='step-badge'>STEP 05 — EXPLORATORY DATA ANALYSIS</div>", unsafe_allow_html=True)
        st.markdown("## 📊 Exploratory Data Analysis")

        if st.session_state.cleaned_df is None:
            st.warning("⬆️ Run preprocessing first.")
        else:
            if st.button("▶ Run EDA"):
                df = st.session_state.cleaned_df

                numeric_cols = [
                    "Steps_Taken", "Calories_Burned", "Hours_Slept",
                    "Active_Minutes", "Heart_Rate (bpm)", "Stress_Level (1-10)"
                ]
                numeric_cols = [c for c in numeric_cols if c in df.columns]

                # Each plot full-width stacked vertically
                for i, col in enumerate(numeric_cols):
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"#### 📈 Distribution — {col}")

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.set_facecolor("#0c1829")
                    fig.patch.set_facecolor("#0c1829")
                    sns.histplot(
                        df[col], kde=True, ax=ax,
                        color=PALETTE[i % len(PALETTE)],
                        edgecolor="none", alpha=0.75
                    )
                    if ax.lines:
                        ax.lines[-1].set_color("#f0f9ff")
                    ax.set_title(col, color="#38bdf8", fontsize=11, fontweight="bold")
                    ax.tick_params(colors="#64748b", labelsize=8)
                    ax.xaxis.label.set_color("#64748b")
                    ax.yaxis.label.set_color("#64748b")
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#1e3a5f")
                    ax.grid(color="#1e3a5f", alpha=0.5, linewidth=0.6)
                    ax.set_xlabel(col, fontsize=9)
                    ax.set_ylabel("Count", fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)


# =====================================================
# MILESTONE 2
# =====================================================
def milestone2():

    st.markdown("""
    <div class='page-header'>
        <h1>💓 ML Analytics Pipeline</h1>
        <p>🔵 Clustering &nbsp;·&nbsp; 📈 Forecasting &nbsp;·&nbsp; 🔬 Feature Engineering — Real Fitbit Data</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Upload Fitbit Dataset Files")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        daily_file = st.file_uploader("📋 dailyActivity",     type=["csv"])
    with col2:
        steps_file = st.file_uploader("👟 hourlySteps",       type=["csv"])
    with col3:
        int_file   = st.file_uploader("⚡ hourlyIntensities", type=["csv"])
    with col4:
        sleep_file = st.file_uploader("😴 minuteSleep",       type=["csv"])
    with col5:
        hr_file    = st.file_uploader("❤️ heartrate",         type=["csv"])

    if not (daily_file and steps_file and int_file and sleep_file and hr_file):
        st.markdown("""
        <div class='section-card' style='text-align:center; padding:48px;'>
            <div style='font-size:3.5rem; margin-bottom:14px;'>⬆️</div>
            <div style='color:#64748b; font-size:15px;'>
                Upload all <strong style='color:#38bdf8;'>5 Fitbit CSV files</strong> above to unlock the ML pipeline
            </div>
            <div style='color:#475569; font-size:12px; margin-top:8px;'>
                dailyActivity · hourlySteps · hourlyIntensities · minuteSleep · heartrate
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.success("✅ All 5 files uploaded — Pipeline Ready")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── LOAD & PROCESS DATA ────────────────────────────
    daily = pd.read_csv(daily_file)
    sleep = pd.read_csv(sleep_file)
    hr    = pd.read_csv(hr_file)

    daily["ActivityDate"] = pd.to_datetime(daily["ActivityDate"])
    sleep["date"]         = pd.to_datetime(sleep["date"])
    hr["Time"]            = pd.to_datetime(hr["Time"])

    hr_minute = (
        hr.set_index("Time")
        .groupby("Id")["Value"]
        .resample("1min")
        .mean()
        .reset_index()
    )
    hr_minute.columns = ["Id", "Time", "HeartRate"]
    hr_minute.dropna(inplace=True)
    hr_minute["Date"] = hr_minute["Time"].dt.date

    sleep["Date"] = sleep["date"].dt.date
    sleep_daily = (
        sleep.groupby(["Id", "Date"])
        .agg(TotalSleepMinutes=("value", "count"))
        .reset_index()
    )

    master = daily.copy()
    master["Date"] = master["ActivityDate"].dt.date
    master = master.merge(sleep_daily, on=["Id", "Date"], how="left")
    master.fillna(0, inplace=True)

    cluster_cols = ["TotalSteps", "Calories", "VeryActiveMinutes", "SedentaryMinutes", "TotalSleepMinutes"]
    cluster_features = master.groupby("Id")[cluster_cols].mean()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_features)

    OPTIMAL_K = 3
    kmeans    = KMeans(n_clusters=OPTIMAL_K, n_init=10, random_state=42)
    labels    = kmeans.fit_predict(X_scaled)
    cluster_features["Cluster"] = labels

    # ── TABS ──────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊  Master Data",
        "🔵  Clustering",
        "📉  Projections",
        "🧪  TSFresh",
        "📈  Forecasting",
        "📋  Summary"
    ])

    # ── TAB 1 · MASTER DATA ───────────────────────────
    with tab1:
        st.markdown("<div class='step-badge'>OVERVIEW — MASTER DATASET</div>", unsafe_allow_html=True)
        st.markdown("## 📊 Master Dataset")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👥 Users",    daily["Id"].nunique())
        c2.metric("📋 Records",  f"{master.shape[0]:,}")
        c3.metric("📅 Days",     master["Date"].nunique())
        c4.metric("🧮 Features", master.shape[1])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🗂️ Dataset Preview")
        st.dataframe(master.head(15), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Activity Summary Statistics")
        st.dataframe(master[cluster_cols].describe().round(2), use_container_width=True)

    # ── TAB 2 · CLUSTERING ────────────────────────────
    with tab2:
        st.markdown("<div class='step-badge'>STEP 01 — KMEANS CLUSTERING</div>", unsafe_allow_html=True)
        st.markdown("## 🔵 KMeans Clustering")

        # Elbow Curve — full width
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Elbow Curve — Finding Optimal K")
        inertias = []
        K_range  = range(2, 8)
        for k in K_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            km.fit(X_scaled)
            inertias.append(km.inertia_)

        fig1, ax1 = plt.subplots(figsize=(10, 4))
        apply_dark_style(fig1, ax1)
        ax1.plot(list(K_range), inertias, "o-", color=PALETTE[0], markersize=8, linewidth=2.5)
        ax1.fill_between(list(K_range), inertias, alpha=0.12, color=PALETTE[0])
        ax1.set_title("Elbow Curve — Optimal K Selection", fontsize=11)
        ax1.set_xlabel("Number of Clusters (K)", fontsize=9)
        ax1.set_ylabel("Inertia", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        st.info(f"👉 Optimal K selected: **{OPTIMAL_K}** (elbow point)")

        # Cluster Profiles Table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📋 Cluster Profiles Table")
        profile = cluster_features.groupby("Cluster")[cluster_cols].mean().round(2)
        st.dataframe(profile, use_container_width=True)

        # Cluster Bar Chart — full width
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Cluster Mean Feature Values — Bar Chart")
        fig_p, ax_p = plt.subplots(figsize=(10, 5))
        apply_dark_style(fig_p, ax_p)
        x     = np.arange(len(profile.columns))
        width = 0.25
        for i, (idx, row) in enumerate(profile.iterrows()):
            ax_p.bar(x + i * width, row.values, width,
                     label=f"Cluster {idx}",
                     color=PALETTE[i % len(PALETTE)], alpha=0.85)
        ax_p.set_xticks(x + width)
        ax_p.set_xticklabels(profile.columns, rotation=20, ha="right", fontsize=9)
        ax_p.legend(fontsize=9, facecolor="#0c1829", edgecolor="#1e3a5f", labelcolor="#94a3b8")
        ax_p.set_title("Cluster Mean Feature Comparison", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig_p, use_container_width=True)

        # Cluster Interpretation Cards
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🧠 Cluster Interpretation")
        interp_cols = st.columns(OPTIMAL_K)
        for i in range(OPTIMAL_K):
            if i in profile.index:
                row   = profile.loc[i]
                steps = row["TotalSteps"]
                with interp_cols[i]:
                    if steps > 10000:
                        badge = "🏃 Highly Active"
                        color = "#15803d"
                    elif steps > 5000:
                        badge = "🚶 Moderate"
                        color = "#0369a1"
                    else:
                        badge = "🛋️ Sedentary"
                        color = "#a16207"
                    st.markdown(f"""
                    <div style='background:{color}22; border:1px solid {color}; border-radius:12px;
                                padding:18px; text-align:center; margin-top:8px;'>
                        <div style='font-size:1.1rem; font-weight:700; color:#f0f9ff;'>Cluster {i}</div>
                        <div style='color:#94a3b8; font-size:12px; margin:6px 0;'>{steps:,.0f} avg steps/day</div>
                        <div style='background:{color}; color:#fff; border-radius:6px;
                                    padding:4px 10px; font-size:12px; font-weight:700;
                                    display:inline-block; margin-top:4px;'>{badge}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── TAB 3 · PROJECTIONS ───────────────────────────
    with tab3:
        st.markdown("<div class='step-badge'>STEP 02 — DIMENSIONALITY REDUCTION</div>", unsafe_allow_html=True)
        st.markdown("## 📉 Dimensionality Reduction")

        # PCA — full width
        pca   = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📉 PCA — 2D Cluster Projection")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        apply_dark_style(fig2, ax2)
        for c_id in sorted(set(labels)):
            mask = labels == c_id
            ax2.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                c=PALETTE[c_id % len(PALETTE)],
                label=f"Cluster {c_id}", s=80, alpha=0.85,
                edgecolors="#0c1829", linewidths=0.8
            )
        ax2.set_title("PCA Cluster Projection", fontsize=11)
        ax2.set_xlabel("PC1", fontsize=9)
        ax2.set_ylabel("PC2", fontsize=9)
        ax2.legend(fontsize=9, facecolor="#0c1829", edgecolor="#1e3a5f", labelcolor="#94a3b8")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        st.caption(
            f"Variance explained — PC1: {pca.explained_variance_ratio_[0]*100:.1f}%"
            f" · PC2: {pca.explained_variance_ratio_[1]*100:.1f}%"
        )

        # t-SNE KMeans — full width
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🧠 t-SNE — KMeans Cluster View")
        st.info("⏳ Running t-SNE... (may take ~20–30 seconds)")

        tsne_model = TSNE(
            n_components=2, random_state=42,
            perplexity=min(30, len(X_scaled) - 1), max_iter=1000
        )
        X_tsne = tsne_model.fit_transform(X_scaled)

        kmeans2       = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
        kmeans_labels = kmeans2.fit_predict(X_scaled)

        fig_ts1, ax_ts1 = plt.subplots(figsize=(10, 5))
        apply_dark_style(fig_ts1, ax_ts1)
        for c_id in sorted(set(kmeans_labels)):
            mask = kmeans_labels == c_id
            ax_ts1.scatter(
                X_tsne[mask, 0], X_tsne[mask, 1],
                c=PALETTE[c_id % len(PALETTE)],
                label=f"Cluster {c_id}", s=60, alpha=0.85,
                edgecolors="#0c1829", linewidths=0.5
            )
        ax_ts1.set_title(f"t-SNE — KMeans (K={OPTIMAL_K})", fontsize=11)
        ax_ts1.set_xlabel("Dimension 1", fontsize=9)
        ax_ts1.set_ylabel("Dimension 2", fontsize=9)
        ax_ts1.legend(fontsize=9, facecolor="#0c1829", edgecolor="#1e3a5f", labelcolor="#94a3b8")
        plt.tight_layout()
        st.pyplot(fig_ts1, use_container_width=True)

        # t-SNE DBSCAN — full width
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🧠 t-SNE — DBSCAN Cluster View")
        EPS           = 0.5
        dbscan        = DBSCAN(eps=EPS, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)

        fig_ts2, ax_ts2 = plt.subplots(figsize=(10, 5))
        apply_dark_style(fig_ts2, ax_ts2)
        for label in sorted(set(dbscan_labels)):
            mask = dbscan_labels == label
            if label == -1:
                ax_ts2.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                               c="#ef4444", marker="x", s=60,
                               label="Noise", alpha=0.9)
            else:
                ax_ts2.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                               c=PALETTE[label % len(PALETTE)],
                               label=f"Cluster {label}", s=60, alpha=0.85,
                               edgecolors="#0c1829", linewidths=0.5)
        ax_ts2.set_title(f"t-SNE — DBSCAN (eps={EPS})", fontsize=11)
        ax_ts2.set_xlabel("Dimension 1", fontsize=9)
        ax_ts2.set_ylabel("Dimension 2", fontsize=9)
        ax_ts2.legend(fontsize=9, facecolor="#0c1829", edgecolor="#1e3a5f", labelcolor="#94a3b8")
        plt.tight_layout()
        st.pyplot(fig_ts2, use_container_width=True)

        n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise       = list(dbscan_labels).count(-1)
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("🔵 DBSCAN Clusters", n_clusters_db)
        c2.metric("🔴 Noise Points",    n_noise)
        c3.metric("📊 Noise %",         f"{n_noise/len(dbscan_labels)*100:.1f}%")

    # ── TAB 4 · TSFRESH ───────────────────────────────
    with tab4:
        st.markdown("<div class='step-badge'>STEP 03 — TSFRESH FEATURE ENGINEERING</div>", unsafe_allow_html=True)
        st.markdown("## 🧪 TSFresh Feature Extraction")

        ts_hr = hr_minute.rename(columns={"Id": "id", "Time": "time", "HeartRate": "value"})

        features = extract_features(
            ts_hr,
            column_id="id",
            column_sort="time",
            column_value="value",
            default_fc_parameters=MinimalFCParameters()
        )
        features.dropna(axis=1, how="all", inplace=True)

        mm_scaler     = MinMaxScaler()
        features_norm = pd.DataFrame(
            mm_scaler.fit_transform(features),
            columns=features.columns
        )

        c1, c2 = st.columns(2)
        c1.metric("🔬 Extracted Features", features.shape[1])
        c2.metric("👥 Users (rows)",        features.shape[0])

        # Heatmap — full width
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔥 Feature Matrix Heatmap (Normalized 0–1)")
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        apply_dark_style(fig4, ax4)
        sns.heatmap(
            features_norm, ax=ax4, cmap="Blues",
            annot=True, fmt=".2f",
            annot_kws={"size": 7},
            linewidths=0.4, cbar=True,
            linecolor="#1e3a5f"
        )
        ax4.set_title("TSFresh Feature Matrix (Normalized 0–1)", fontsize=11)
        ax4.tick_params(axis='x', labelsize=7, rotation=90)
        ax4.tick_params(axis='y', labelsize=7)
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📋 Raw Feature Values")
        st.dataframe(features.round(3), use_container_width=True)

    # ── TAB 5 · FORECASTING ───────────────────────────
    with tab5:
        st.markdown("<div class='step-badge'>STEP 04 — PROPHET FORECASTING</div>", unsafe_allow_html=True)
        st.markdown("## 📈 Heart Rate Forecast")

        hr_daily_plot = (
            hr_minute.groupby("Date")["HeartRate"]
            .mean()
            .reset_index()
        )
        prophet_hr       = hr_daily_plot.rename(columns={"Date": "ds", "HeartRate": "y"})
        prophet_hr["ds"] = pd.to_datetime(prophet_hr["ds"])

        model_hr    = Prophet()
        model_hr.fit(prophet_hr)
        future_hr   = model_hr.make_future_dataframe(periods=30)
        forecast_hr = model_hr.predict(future_hr)

        # HR Forecast — full width
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### ❤️ Heart Rate Forecast — 30 Days Ahead")
        fig_hr, ax_hr = plt.subplots(figsize=(10, 5))
        apply_dark_style(fig_hr, ax_hr)
        ax_hr.scatter(prophet_hr["ds"], prophet_hr["y"],
                      s=25, alpha=0.75, color=PALETTE[1], label="Actual", zorder=3)
        ax_hr.plot(forecast_hr["ds"], forecast_hr["yhat"],
                   linewidth=2.5, color=PALETTE[0], label="Forecast")
        ax_hr.fill_between(
            forecast_hr["ds"],
            forecast_hr["yhat_lower"],
            forecast_hr["yhat_upper"],
            alpha=0.15, color=PALETTE[0], label="80% CI"
        )
        ax_hr.set_title("Heart Rate Forecast — 30 Day Horizon", fontsize=11)
        ax_hr.set_xlabel("Date", fontsize=9)
        ax_hr.set_ylabel("BPM", fontsize=9)
        ax_hr.legend(fontsize=9, facecolor="#0c1829", edgecolor="#1e3a5f", labelcolor="#94a3b8")
        plt.xticks(rotation=30, fontsize=7)
        plt.tight_layout()
        st.pyplot(fig_hr, use_container_width=True)

        # Prophet Components — full width
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Prophet Decomposition Components")
        fig_comp = model_hr.plot_components(forecast_hr)
        fig_comp.set_size_inches(10, 5)
        fig_comp.patch.set_facecolor("#0c1829")
        for ax_c in fig_comp.get_axes():
            ax_c.set_facecolor("#0c1829")
            ax_c.tick_params(colors="#64748b", labelsize=7)
            ax_c.xaxis.label.set_color("#64748b")
            ax_c.yaxis.label.set_color("#64748b")
            ax_c.title.set_color("#38bdf8")
            for spine in ax_c.spines.values():
                spine.set_edgecolor("#1e3a5f")
            ax_c.grid(color="#1e3a5f", alpha=0.4)
            for line in ax_c.get_lines():
                if line.get_color() in ["#0072B2", "b", "blue"]:
                    line.set_color(PALETTE[0])
        plt.tight_layout()
        st.pyplot(fig_comp, use_container_width=True)

        # Cluster Profile bar chart — full width
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Cluster Profiles")
        profile = cluster_features.groupby("Cluster")[cluster_cols].mean().round(2)
        st.dataframe(profile, use_container_width=True)

        fig_profile, ax_profile = plt.subplots(figsize=(10, 4))
        apply_dark_style(fig_profile, ax_profile)
        x     = np.arange(len(profile.columns))
        width = 0.25
        for i, (idx, row) in enumerate(profile.iterrows()):
            ax_profile.bar(x + i * width, row.values, width,
                           label=f"Cluster {idx}",
                           color=PALETTE[i % len(PALETTE)], alpha=0.85)
        ax_profile.set_xticks(x + width)
        ax_profile.set_xticklabels(profile.columns, rotation=20, ha="right", fontsize=9)
        ax_profile.legend(fontsize=9, facecolor="#0c1829", edgecolor="#1e3a5f", labelcolor="#94a3b8")
        ax_profile.set_title("Cluster Profile Comparison", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig_profile, use_container_width=True)

    # ── TAB 6 · SUMMARY ───────────────────────────────
    with tab6:
        st.markdown("<div class='step-badge'>PIPELINE SUMMARY</div>", unsafe_allow_html=True)
        st.markdown("## 📋 Milestone 2 — Full Summary")

        profile      = cluster_features.groupby("Cluster")[cluster_cols].mean().round(2)
        cluster_dist = dict(cluster_features["Cluster"].value_counts().sort_index())

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👥 Total Users",      cluster_features.shape[0])
        c2.metric("📅 Days Tracked",     "31")
        c3.metric("🔬 TSFresh Features", "Auto-extracted")
        c4.metric("🤖 Clusters Found",   OPTIMAL_K)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("""
            <div class='section-card'>
                <div style='color:#38bdf8; font-weight:700; font-size:14px; margin-bottom:12px;'>✅ Dataset Info</div>
                <div style='color:#94a3b8; font-size:13px; line-height:2;'>
                    📅 Period: March–April 2016<br>
                    👥 Participants: Real Fitbit users<br>
                    📂 Source: Kaggle Fitbit Dataset
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class='section-card'>
                <div style='color:#818cf8; font-weight:700; font-size:14px; margin-bottom:12px;'>🤖 KMeans Results</div>
                <div style='color:#94a3b8; font-size:13px; line-height:2;'>
                    🔵 Clusters: {OPTIMAL_K}<br>
                    📊 Distribution: {cluster_dist}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("""
            <div class='section-card'>
                <div style='color:#34d399; font-weight:700; font-size:14px; margin-bottom:12px;'>📈 Prophet Forecasts</div>
                <div style='color:#94a3b8; font-size:13px; line-height:2;'>
                    ❤️ Heart Rate — 30 day forecast (80% CI)<br>
                    👟 Steps — 30 day forecast (80% CI)<br>
                    😴 Sleep — 30 day forecast (80% CI)
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class='section-card'>
                <div style='color:#fb923c; font-weight:700; font-size:14px; margin-bottom:12px;'>🔬 TSFresh Features</div>
                <div style='color:#94a3b8; font-size:13px; line-height:2;'>
                    ⏱️ Source: Minute-level HR data<br>
                    📐 Parameters: MinimalFCParameters<br>
                    🔢 Normalized: MinMaxScaler (0–1)
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Cluster Profiles Overview")
        st.dataframe(profile, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.success("✅ Milestone 2 Completed Successfully — All modules executed!")


# =====================================================
# APP ROUTER
# =====================================================
if milestone == "📊  Data Processing Pipeline":
    milestone1()
elif milestone == "🤖  ML Analytics Pipeline":
    milestone2()