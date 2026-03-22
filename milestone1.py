import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler, MinMaxScaler




# -------------------------------------------------------
# PAGE CONFIG (ONLY ONCE)
# -------------------------------------------------------

st.set_page_config(
    page_title="FitPulse ML Dashboard",
    page_icon="💓",
    layout="wide"
)
# -------------------------------------------------------
# SIDEBAR MILESTONE SELECTOR
# -------------------------------------------------------

st.sidebar.title("💓 FitPulse Dashboard")

st.sidebar.markdown("---")


# -------------------------------------------------------
# GLOBAL DARK CORPORATE UI
# -------------------------------------------------------

st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#0f172a,#020617);
color:white;
}

/* Sidebar */

[data-testid="stSidebar"]{
background:#020617;
}

[data-testid="stSidebar"] *{
color:white !important;
}

/* Buttons */

button{
background:#2563eb !important;
color:white !important;
border-radius:8px;
}

/* Headers */

h1,h2,h3{
color:#60a5fa;
}

</style>
""", unsafe_allow_html=True)



# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

# -------------------------------------------------------
# SIDEBAR EXECUTED FEATURES
# -------------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Pipeline Features")

st.sidebar.write("📂 **Data Upload**")
st.sidebar.caption("Upload Fitbit and health datasets for analysis")

st.sidebar.write("🔍 **Missing Value Analysis**")
st.sidebar.caption("Detect null values and visualize missing data %")

st.sidebar.write("🧹 **Data Preprocessing**")
st.sidebar.caption("Interpolation, forward fill and cleaning of records")

st.sidebar.write("👀 **Clean Data Preview**")
st.sidebar.caption("View processed dataset after preprocessing")

st.sidebar.write("📊 **Exploratory Data Analysis**")
st.sidebar.caption("Distribution plots for health metrics")

st.sidebar.write("❤️ **Heart Rate Processing**")
st.sidebar.caption("Resampling second-level HR data to minute level")

st.sidebar.write("🔬 **TSFresh Feature Engineering**")
st.sidebar.caption("Automatic time-series feature extraction")

st.sidebar.write("🔥 **Feature Heatmap Visualization**")
st.sidebar.caption("Normalized feature matrix heatmap")

st.sidebar.write("📈 **Heart Rate Forecasting**")
st.sidebar.caption("Future HR prediction using Prophet")

st.sidebar.write("🤖 **User Clustering**")
st.sidebar.caption("KMeans clustering based on activity metrics")

st.sidebar.write("📉 **PCA Visualization**")
st.sidebar.caption("2D dimensionality reduction of clusters")

st.sidebar.write("🧠 **t-SNE Projection**")
st.sidebar.caption("Advanced visualization of user clusters")

st.sidebar.markdown("---")

st.sidebar.success("🚀 ML Pipeline Active")


# =====================================================
# MILESTONE 1 FUNCTION
# =====================================================

def milestone1():

    if "df" not in st.session_state:
        st.session_state.df = None

    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = None

    st.title("💙 Fitness Health Data — Pro Pipeline")

    # STEP 1
    st.header("📂 Step 1 · Upload Dataset")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:

        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

        rows, cols = df.shape
        total_nulls = df.isnull().sum().sum()

        c1,c2,c3 = st.columns(3)

        c1.metric("Rows", rows)
        c2.metric("Columns", cols)
        c3.metric("Total Nulls", total_nulls)

        st.success("Dataset Loaded Successfully!")

    # STEP 2
    if st.session_state.df is not None:

        st.header("🔍 Step 2 · Check Null Values")

        df = st.session_state.df

        null_counts = df.isnull().sum()

        st.dataframe(null_counts)

        st.subheader("📊 Missing Data Percentage")

        null_percent = (null_counts/len(df))*100

        fig,ax = plt.subplots()

        null_percent.sort_values(ascending=False).plot(kind="bar",ax=ax)

        ax.set_ylabel("Missing %")

        st.pyplot(fig)

    # STEP 3
    if st.session_state.df is not None:

        st.header("⚙ Step 3 · Preprocess Data")

        if st.button("Run Preprocessing"):

            df = st.session_state.df.copy()

            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"],errors="coerce")

            numeric_cols = [
            "Hours_Slept",
            "Water_Intake (Liters)",
            "Active_Minutes",
            "Heart_Rate (bpm)"
            ]

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

            st.success("Preprocessing Completed")

    # STEP 4
    if st.session_state.cleaned_df is not None:

        st.header("👁 Step 4 · Preview Cleaned Data")

        df = st.session_state.cleaned_df

        st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

        st.dataframe(df.head(20))

    # STEP 5
    if st.session_state.cleaned_df is not None:

        st.header("📊 Step 5 · EDA")

        if st.button("Run EDA"):

            df = st.session_state.cleaned_df

            numeric_cols = [
            "Steps_Taken",
            "Calories_Burned",
            "Hours_Slept",
            "Active_Minutes",
            "Heart_Rate (bpm)",
            "Stress_Level (1-10)"
            ]

            numeric_cols=[c for c in numeric_cols if c in df.columns]

            fig,axes = plt.subplots(3,2,figsize=(12,8))

            axes = axes.flatten()

            for i,col in enumerate(numeric_cols):

                sns.histplot(df[col],kde=True,ax=axes[i])

                axes[i].set_title(col)

            plt.tight_layout()

            st.pyplot(fig)


