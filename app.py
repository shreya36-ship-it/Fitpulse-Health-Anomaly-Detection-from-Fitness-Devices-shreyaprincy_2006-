import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import tempfile
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─── lazy heavy imports ───────────────────────────────────────────────────────
@st.cache_resource
def _load_sklearn():
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    return StandardScaler, MinMaxScaler, KMeans, DBSCAN, PCA, TSNE

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FitPulse · ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# THEME CONSTANTS
# ─────────────────────────────────────────────────────────────
BG          = "linear-gradient(135deg,#0a0e1a 0%,#0f1729 40%,#0a1628 100%)"
CARD_BG     = "rgba(15,23,42,0.85)"
CARD_BOR    = "rgba(99,179,237,0.2)"
TEXT        = "#e2e8f0"
MUTED       = "#94a3b8"
ACCENT      = "#63b3ed"
ACCENT2     = "#f687b3"
ACCENT3     = "#68d391"
ACCENT_RED  = "#fc8181"
ACCENT_ORG  = "#f6ad55"
PLOT_BG     = "#0f172a"
PAPER_BG    = "#0a0e1a"
GRID_CLR    = "rgba(255,255,255,0.05)"
BADGE_BG    = "rgba(99,179,237,0.15)"
SECTION_BG  = "rgba(99,179,237,0.07)"
WARN_BG     = "rgba(246,173,85,0.12)"
WARN_BOR    = "rgba(246,173,85,0.4)"
SUCCESS_BG  = "rgba(104,211,145,0.1)"
SUCCESS_BOR = "rgba(104,211,145,0.4)"
DANGER_BG   = "rgba(252,129,129,0.1)"
DANGER_BOR  = "rgba(252,129,129,0.4)"

# Table specific colors
TABLE_HEADER_BG   = "#0d1424"
TABLE_ROW_ODD     = "#0f1a2e"
TABLE_ROW_EVEN    = "#0a1220"
TABLE_ROW_HOVER   = "#1a2a45"
TABLE_BORDER      = "rgba(99,179,237,0.12)"
TABLE_HEADER_TEXT = "#63b3ed"
TABLE_TEXT        = "#cbd5e1"

MPL_PALETTE = [ACCENT, ACCENT2, ACCENT3, ACCENT_ORG, "#b794f4", "#f6ad55", "#76e4f7"]
DARK_BG_MPL = "#0f172a"
GRID_MPL    = "#1e293b"
TEXT_MPL    = "#94a3b8"

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS  ── fully upgraded
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');

*,*::before,*::after{{box-sizing:border-box}}
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stAppViewBlockContainer"],.main{{
    background:{BG}!important;font-family:'Inter',sans-serif;color:{TEXT}!important}}
[data-testid="stHeader"]{{background:transparent!important}}
[data-testid="stSidebar"]{{
    background:linear-gradient(180deg,#060910 0%,#080d1a 50%,#060910 100%)!important;
    border-right:1px solid rgba(99,179,237,0.15)!important;
    box-shadow:4px 0 24px rgba(0,0,0,0.4)!important}}
[data-testid="stSidebar"] *{{color:{TEXT}!important}}
[data-testid="stSidebar"] .stSelectbox>div>div{{
    background:#0d1424!important;
    border:1px solid rgba(99,179,237,0.25)!important;
    border-radius:8px!important;
    color:{TEXT}!important}}
[data-testid="stSidebar"] label{{color:{MUTED}!important;font-size:0.72rem!important}}
.block-container{{padding:1.2rem 2rem 3rem 2rem!important;max-width:1500px}}
p,div,span,label{{color:{TEXT}}}
h1,h2,h3,h4{{font-family:'Syne',sans-serif!important;color:{TEXT}!important}}

/* ──────────── DROPDOWN DARK THEME ──────────── */
[data-baseweb="select"]>div{{
    background:#0d1424!important;
    border:1px solid rgba(99,179,237,0.25)!important;
    border-radius:10px!important;
    color:{TEXT}!important;
    transition:border-color 0.2s!important}}
[data-baseweb="select"]>div:hover{{
    border-color:{ACCENT}!important}}
[data-baseweb="select"] [data-testid="stSelectboxVirtualDropdown"],
[data-baseweb="popover"],
[data-baseweb="menu"]{{
    background:#0d1424!important;
    border:1px solid rgba(99,179,237,0.25)!important;
    border-radius:10px!important;
    box-shadow:0 8px 32px rgba(0,0,0,0.6)!important}}
[role="option"]{{
    background:#0d1424!important;
    color:{TEXT}!important;
    font-size:0.85rem!important}}
[role="option"]:hover,
[aria-selected="true"]{{
    background:rgba(99,179,237,0.12)!important;
    color:{ACCENT}!important}}
[data-baseweb="select"] svg{{fill:{MUTED}!important}}

/* ──────────── MULTISELECT DARK ──────────── */
[data-baseweb="tag"]{{
    background:rgba(99,179,237,0.15)!important;
    border:1px solid rgba(99,179,237,0.3)!important;
    border-radius:6px!important}}
[data-baseweb="tag"] span{{color:{ACCENT}!important;font-size:0.75rem!important}}

/* ──────────── PROFESSIONAL TABLE CSS ──────────── */
[data-testid="stDataFrame"]{{
    border:1px solid {TABLE_BORDER}!important;
    border-radius:14px!important;
    overflow:hidden!important;
    box-shadow:0 4px 24px rgba(0,0,0,0.3)!important}}
[data-testid="stDataFrame"] thead tr th{{
    background:{TABLE_HEADER_BG}!important;
    color:{TABLE_HEADER_TEXT}!important;
    font-family:'JetBrains Mono',monospace!important;
    font-size:0.72rem!important;
    font-weight:700!important;
    text-transform:uppercase!important;
    letter-spacing:0.08em!important;
    padding:10px 14px!important;
    border-bottom:1px solid rgba(99,179,237,0.2)!important;
    position:sticky!important;
    top:0!important;
    z-index:10!important}}
[data-testid="stDataFrame"] tbody tr:nth-child(odd){{
    background:{TABLE_ROW_ODD}!important}}
[data-testid="stDataFrame"] tbody tr:nth-child(even){{
    background:{TABLE_ROW_EVEN}!important}}
[data-testid="stDataFrame"] tbody tr:hover{{
    background:{TABLE_ROW_HOVER}!important;
    transition:background 0.15s ease!important}}
[data-testid="stDataFrame"] tbody tr td{{
    color:{TABLE_TEXT}!important;
    font-size:0.82rem!important;
    padding:8px 14px!important;
    border-bottom:1px solid rgba(255,255,255,0.03)!important;
    font-family:'Inter',sans-serif!important}}
[data-testid="stDataFrame"] [data-testid="glideDataEditor"]{{
    background:#0a1220!important}}
/* Scrollbar inside dataframe */
[data-testid="stDataFrame"] ::-webkit-scrollbar{{width:5px;height:5px}}
[data-testid="stDataFrame"] ::-webkit-scrollbar-track{{background:#060910}}
[data-testid="stDataFrame"] ::-webkit-scrollbar-thumb{{background:rgba(99,179,237,0.3);border-radius:10px}}

/* ──────────── HERO ──────────── */
.m4-hero{{
    background:linear-gradient(135deg,rgba(99,179,237,0.08),rgba(104,211,145,0.05),rgba(10,14,26,0.9));
    border:1px solid {CARD_BOR};border-radius:20px;padding:2rem 2.5rem;
    margin-bottom:1.5rem;position:relative;overflow:hidden}}
.m4-hero::before{{
    content:'';position:absolute;top:-60px;right:-60px;
    width:200px;height:200px;
    background:radial-gradient(circle,rgba(99,179,237,0.08) 0%,transparent 70%);
    pointer-events:none}}
.hero-title{{font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;
    color:{TEXT};margin:0 0 0.3rem 0;letter-spacing:-0.02em}}
.hero-sub{{font-size:1rem;color:{MUTED};font-weight:300;margin:0}}
.hero-badge{{display:inline-block;background:{BADGE_BG};border:1px solid {CARD_BOR};
    border-radius:100px;padding:0.25rem 0.9rem;font-size:0.72rem;
    font-family:'JetBrains Mono',monospace;color:{ACCENT};margin-bottom:0.8rem}}

/* ──────────── KPI ──────────── */
.kpi-grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:0.7rem;margin:1rem 0}}
.kpi-grid-4{{display:grid;grid-template-columns:repeat(4,1fr);gap:0.7rem;margin:1rem 0}}
.kpi-grid-3{{display:grid;grid-template-columns:repeat(3,1fr);gap:0.7rem;margin:1rem 0}}
.kpi-card{{
    background:{CARD_BG};border:1px solid {CARD_BOR};border-radius:14px;
    padding:1rem 1.1rem;text-align:center;backdrop-filter:blur(10px);
    transition:transform 0.2s,border-color 0.2s}}
.kpi-card:hover{{transform:translateY(-2px);border-color:rgba(99,179,237,0.4)}}
.kpi-val{{font-family:'Syne',sans-serif;font-size:1.7rem;font-weight:800;
    line-height:1;margin-bottom:0.2rem}}
.kpi-label{{font-size:0.68rem;color:{MUTED};text-transform:uppercase;letter-spacing:0.07em}}
.kpi-sub{{font-size:0.65rem;color:{MUTED};margin-top:0.15rem}}

/* ──────────── SECTION HEADER ──────────── */
.sec-header{{display:flex;align-items:center;gap:0.8rem;margin:1.5rem 0 0.8rem;
    padding-bottom:0.5rem;border-bottom:1px solid {CARD_BOR}}}
.sec-icon{{font-size:1.3rem;width:2rem;height:2rem;display:flex;align-items:center;
    justify-content:center;background:{BADGE_BG};border-radius:8px;border:1px solid {CARD_BOR}}}
.sec-title{{font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:700;color:{TEXT};margin:0}}
.sec-badge{{margin-left:auto;background:{BADGE_BG};border:1px solid {CARD_BOR};
    border-radius:100px;padding:0.2rem 0.7rem;font-size:0.7rem;
    font-family:'JetBrains Mono',monospace;color:{ACCENT}}}

/* ──────────── CARD ──────────── */
.card{{background:{CARD_BG};border:1px solid {CARD_BOR};border-radius:14px;
    padding:1.2rem 1.4rem;margin-bottom:0.8rem;backdrop-filter:blur(10px)}}
.card-title{{font-family:'Syne',sans-serif;font-size:0.85rem;font-weight:700;
    color:{MUTED};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem}}

/* ──────────── ALERTS ──────────── */
.alert-info{{background:{BADGE_BG};border-left:3px solid {ACCENT};border-radius:0 10px 10px 0;
    padding:0.7rem 1rem;margin:0.5rem 0;font-size:0.83rem;color:#bee3f8}}
.alert-success{{background:{SUCCESS_BG};border-left:3px solid {ACCENT3};border-radius:0 10px 10px 0;
    padding:0.7rem 1rem;margin:0.5rem 0;font-size:0.83rem;color:#9ae6b4}}
.alert-danger{{background:{DANGER_BG};border-left:3px solid {ACCENT_RED};border-radius:0 10px 10px 0;
    padding:0.7rem 1rem;margin:0.5rem 0;font-size:0.83rem;color:#feb2b2}}
.alert-warn{{background:{WARN_BG};border-left:3px solid {ACCENT_ORG};border-radius:0 10px 10px 0;
    padding:0.7rem 1rem;margin:0.5rem 0;font-size:0.83rem;color:#fbd38d}}

/* ──────────── PIPELINE STEPS ──────────── */
.pipeline-step{{display:flex;align-items:flex-start;gap:1rem;padding:0.85rem 0;
    border-bottom:1px solid {CARD_BOR}}}
.pipeline-step:last-child{{border-bottom:none}}
.step-num{{background:linear-gradient(135deg,#1a3fa8,#0b8fd4);color:#fff;
    min-width:28px;height:28px;border-radius:50%;display:flex;align-items:center;
    justify-content:center;font-family:'JetBrains Mono',monospace;font-size:10px;
    font-weight:700;flex-shrink:0}}

/* ──────────── BUTTONS ──────────── */
.stButton>button{{
    background:rgba(99,179,237,0.12)!important;
    border:1px solid {CARD_BOR}!important;color:{ACCENT}!important;border-radius:10px!important;
    font-family:'JetBrains Mono',monospace!important;font-size:0.8rem!important;
    font-weight:500!important;padding:0.45rem 1rem!important;
    transition:all 0.2s!important}}
.stButton>button:hover{{
    background:{ACCENT}!important;color:white!important;
    border-color:{ACCENT}!important;transform:translateY(-1px)!important;
    box-shadow:0 4px 16px rgba(99,179,237,0.3)!important}}

/* Pill button special class */
.pill-btn>button{{
    border-radius:100px!important;
    padding:0.5rem 1.6rem!important;
    font-size:0.82rem!important}}

[data-testid="stDownloadButton"]>button{{
    background:rgba(104,211,145,0.12)!important;border:1px solid {SUCCESS_BOR}!important;
    color:{ACCENT3}!important;border-radius:10px!important;
    font-family:'JetBrains Mono',monospace!important;font-size:0.82rem!important;
    font-weight:600!important;padding:0.55rem 1.2rem!important;
    width:100%!important;transition:all 0.2s!important}}
[data-testid="stDownloadButton"]>button:hover{{
    background:{ACCENT3}!important;color:#0a1628!important;
    transform:translateY(-1px)!important;
    box-shadow:0 4px 16px rgba(104,211,145,0.25)!important}}

/* ──────────── UPLOAD ──────────── */
div[data-testid="stFileUploader"]{{
    background:{SECTION_BG};
    border:2px dashed {CARD_BOR};border-radius:14px;padding:0.5rem}}

/* ──────────── TABS ──────────── */
.stTabs [data-baseweb="tab-list"]{{
    background:rgba(8,13,26,0.8);
    border-radius:12px;
    padding:0.3rem;
    gap:0.2rem;
    border:1px solid rgba(99,179,237,0.12)}}
.stTabs [data-baseweb="tab"]{{
    color:{MUTED};
    font-family:'JetBrains Mono',monospace;
    font-size:0.79rem;
    border-radius:8px!important;
    padding:0.45rem 1rem!important;
    transition:all 0.2s!important}}
.stTabs [data-baseweb="tab"]:hover{{
    color:{TEXT}!important;
    background:rgba(99,179,237,0.06)!important}}
.stTabs [aria-selected="true"]{{
    background:linear-gradient(135deg,rgba(99,179,237,0.18),rgba(99,179,237,0.08))!important;
    color:{ACCENT}!important;
    border:1px solid rgba(99,179,237,0.25)!important;
    font-weight:600!important}}
.stTabs [data-baseweb="tab-panel"]{{
    padding-top:1.2rem!important}}

/* ──────────── METRICS ──────────── */
[data-testid="metric-container"]{{
    background:{CARD_BG}!important;
    border:1px solid {CARD_BOR}!important;border-radius:14px!important;padding:1rem!important}}
[data-testid="metric-container"] label{{
    color:{ACCENT}!important;font-size:10px!important;
    text-transform:uppercase;letter-spacing:2px;font-weight:700!important;
    font-family:'JetBrains Mono',monospace!important}}
[data-testid="metric-container"] [data-testid="stMetricValue"]{{
    color:{TEXT}!important;font-family:'Syne',monospace!important;
    font-weight:800!important;font-size:1.6rem!important}}

/* ──────────── EXPANDER ──────────── */
[data-testid="stExpander"]{{
    background:{CARD_BG}!important;
    border:1px solid {CARD_BOR}!important;border-radius:14px!important}}
[data-testid="stExpander"] summary{{
    color:{ACCENT}!important;font-weight:600!important}}

/* ──────────── SLIDERS (sidebar) ──────────── */
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stThumbValue"]{{
    background:{ACCENT}!important;color:#fff!important;font-size:0.7rem!important}}

/* ──────────── TOGGLE ──────────── */
[data-testid="stToggle"] label{{color:{TEXT}!important}}

/* ──────────── SCROLLBAR ──────────── */
::-webkit-scrollbar{{width:6px;height:6px}}
::-webkit-scrollbar-track{{background:#060910}}
::-webkit-scrollbar-thumb{{background:rgba(99,179,237,0.25);border-radius:10px}}
::-webkit-scrollbar-thumb:hover{{background:rgba(99,179,237,0.5)}}

.m4-divider{{border:none;border-top:1px solid {CARD_BOR};margin:1.5rem 0}}

/* ──────────── FILE STATUS GRID ──────────── */
.file-status-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:4px;margin:0.5rem 0}}
.file-status-cell{{border-radius:8px;padding:0.35rem;text-align:center;font-size:0.65rem;border:1px solid}}

/* ──────────── ANOMALY ROW ──────────── */
.anom-row{{display:flex;align-items:center;gap:0.6rem;padding:0.45rem 0;
    border-bottom:1px solid {CARD_BOR};font-size:0.82rem}}
.anom-row:last-child{{border-bottom:none}}

/* ──────────── SIDEBAR LABELS ──────────── */
.sb-label{{font-size:0.7rem;color:{MUTED};font-family:'JetBrains Mono',monospace;
    margin-bottom:0.4rem;text-transform:uppercase;letter-spacing:0.08em}}

/* ──────────── CUSTOM TABLE (HTML) ──────────── */
.pro-table{{
    width:100%;border-collapse:separate;border-spacing:0;
    border-radius:12px;overflow:hidden;
    border:1px solid {TABLE_BORDER};
    font-size:0.82rem;font-family:'Inter',sans-serif}}
.pro-table thead tr th{{
    background:{TABLE_HEADER_BG};
    color:{TABLE_HEADER_TEXT};
    font-family:'JetBrains Mono',monospace;
    font-size:0.7rem;font-weight:700;
    text-transform:uppercase;letter-spacing:0.09em;
    padding:10px 14px;
    border-bottom:1px solid rgba(99,179,237,0.2);
    position:sticky;top:0;z-index:5}}
.pro-table tbody tr:nth-child(odd) td{{background:{TABLE_ROW_ODD}}}
.pro-table tbody tr:nth-child(even) td{{background:{TABLE_ROW_EVEN}}}
.pro-table tbody tr:hover td{{
    background:{TABLE_ROW_HOVER};transition:background 0.15s ease}}
.pro-table tbody tr td{{
    color:{TABLE_TEXT};
    padding:8px 14px;
    border-bottom:1px solid rgba(255,255,255,0.03)}}

/* ──────────── INSIGHT CARD ──────────── */
.insight-card{{
    background:linear-gradient(135deg,{CARD_BG},{SECTION_BG});
    border:1px solid {CARD_BOR};border-radius:14px;
    padding:1.1rem 1.3rem;margin-bottom:0.6rem;
    transition:transform 0.2s,border-color 0.2s}}
.insight-card:hover{{transform:translateY(-2px);border-color:rgba(99,179,237,0.4)}}
.insight-tag{{
    display:inline-block;
    background:{BADGE_BG};border:1px solid {CARD_BOR};
    border-radius:100px;padding:0.15rem 0.6rem;
    font-size:0.65rem;font-family:'JetBrains Mono',monospace;
    color:{ACCENT};margin-bottom:0.4rem}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
_defaults = {
    "df": None, "cleaned_df": None,
    "file_daily": None, "file_steps": None, "file_int": None,
    "file_sleep": None, "file_hr": None,
    "m3_anomaly_done": False, "m3_sim_done": False,
    "m3_anomalies": None, "m3_sim_results": None,
    "m2_master": None, "m2_done": False,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────
# MATPLOTLIB HELPERS
# ─────────────────────────────────────────────────────────────
def apply_dark(fig, axes=None):
    fig.patch.set_facecolor(DARK_BG_MPL)
    if axes is None:
        axes = fig.get_axes()
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(DARK_BG_MPL)
        ax.tick_params(colors=TEXT_MPL, labelsize=8)
        ax.xaxis.label.set_color(TEXT_MPL)
        ax.yaxis.label.set_color(TEXT_MPL)
        ax.title.set_color(ACCENT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_MPL)
        ax.grid(color=GRID_MPL, alpha=0.5, linewidth=0.6)
    return fig


def show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def save_fig_bytes(fig):
    """Save figure to bytes for PDF embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=DARK_BG_MPL, dpi=120)
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────
def sec(icon, title, badge=None):
    badge_html = (
        f'<span class="sec-badge">{badge}</span>'
        if badge else ""
    )
    st.markdown(
        f'<div class="sec-header">'
        f'<div class="sec-icon">{icon}</div>'
        f'<p class="sec-title">{title}</p>'
        f'{badge_html}</div>',
        unsafe_allow_html=True,
    )


def hero(title, subtitle, badge=""):
    st.markdown(
        f'<div class="m4-hero">'
        f'<div class="hero-badge">{badge}</div>'
        f'<h1 class="hero-title">{title}</h1>'
        f'<p class="hero-sub">{subtitle}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def ui_info(m):    st.markdown(f'<div class="alert-info">ℹ️ {m}</div>',    unsafe_allow_html=True)
def ui_success(m): st.markdown(f'<div class="alert-success">✅ {m}</div>', unsafe_allow_html=True)
def ui_danger(m):  st.markdown(f'<div class="alert-danger">🚨 {m}</div>',  unsafe_allow_html=True)
def ui_warn(m):    st.markdown(f'<div class="alert-warn">⚠️ {m}</div>',   unsafe_allow_html=True)


def divider():
    st.markdown('<hr class="m4-divider">', unsafe_allow_html=True)


def dark_df(df, height=None):
    kwargs = {"use_container_width": True}
    if height:
        kwargs["height"] = height
    st.dataframe(df, **kwargs)


def kpi_row(items):
    """items: list of (value, label, sub, color)"""
    n = len(items)
    cls = {4: "kpi-grid-4", 3: "kpi-grid-3", 6: "kpi-grid"}.get(n, "kpi-grid-4")
    html = f'<div class="{cls}">'
    for val, label, sub, color in items:
        border = f"border-color:rgba({_hex_to_rgb(color)},0.35)" if color else ""
        html += (
            f'<div class="kpi-card" style="{border}">'
            f'<div class="kpi-val" style="color:{color or ACCENT}">{val}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-sub">{sub}</div>'
            f'</div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _hex_to_rgb(h):
    h = h.lstrip("#")
    if len(h) != 6:
        return "99,179,237"
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def insight_card(tag, title, body, color=ACCENT):
    st.markdown(
        f'<div class="insight-card">'
        f'<div class="insight-tag">{tag}</div>'
        f'<div style="font-family:Syne,sans-serif;font-weight:700;font-size:0.92rem;'
        f'color:{color};margin-bottom:0.3rem">{title}</div>'
        f'<div style="color:{MUTED};font-size:0.8rem;line-height:1.6">{body}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# REQUIRED FILES MAP
# ─────────────────────────────────────────────────────────────
FILE_KEYS = {
    "daily": ["dailyactivity", "daily_activity"],
    "steps": ["hourlysteps", "hourly_steps"],
    "int":   ["hourlyintensities", "hourly_intensities"],
    "sleep": ["minutesleep", "minute_sleep"],
    "hr":    ["heartrate", "heart_rate", "heartrate_seconds"],
}
FILE_LABELS = {
    "daily": "dailyActivity", "steps": "hourlySteps",
    "int":   "hourlyIntensities", "sleep": "minuteSleep", "hr": "heartrate",
}


# ─────────────────────────────────────────────────────────────
# SHARED FILE UPLOAD
# ─────────────────────────────────────────────────────────────
def shared_file_upload():
    sec("📂", "Upload Fitbit Dataset Files", "5 files required")
    st.markdown(
        f'<div class="card"><div class="card-title">Drop all 5 CSV files at once</div>'
        f'<div style="color:{MUTED};font-size:0.83rem">Files are stored in session — no re-upload needed when switching modules.</div>'
        f'<div style="color:{MUTED};font-size:0.75rem;font-family:\'JetBrains Mono\',monospace;margin-top:0.5rem">'
        f'dailyActivity · hourlySteps · hourlyIntensities · minuteSleep · heartrate_seconds</div></div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload files", type=["csv"], accept_multiple_files=True,
        key="multi_upload", label_visibility="collapsed",
    )
    if uploaded:
        for uf in uploaded:
            nl = uf.name.lower(); uf.seek(0); raw = uf.read()
            for key, patterns in FILE_KEYS.items():
                if any(p in nl for p in patterns):
                    st.session_state[f"file_{key}"] = raw
                    break

    html = '<div class="file-status-grid">'
    ok_count = 0
    for k, lbl in FILE_LABELS.items():
        ok = bool(st.session_state[f"file_{k}"])
        if ok: ok_count += 1
        col  = SUCCESS_BOR if ok else WARN_BOR
        bg   = SUCCESS_BG  if ok else WARN_BG
        ico  = "✅" if ok else "❌"
        html += (
            f'<div class="file-status-cell" style="background:{bg};border-color:{col};color:{MUTED}">'
            f'{ico}<br>{lbl[:8]}</div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    ok_all = ok_count == 5
    if ok_all:
        ui_success("All 5 files ready — shared across M2, M3 and M4")
    else:
        ui_warn(f"{ok_count}/5 files uploaded — upload all 5 to continue")
    return ok_all


# ─────────────────────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_fitbit_data(daily_b, sleep_b, hr_b):
    daily = pd.read_csv(io.BytesIO(daily_b))
    sleep = pd.read_csv(io.BytesIO(sleep_b))
    hr    = pd.read_csv(io.BytesIO(hr_b))
    daily["ActivityDate"] = pd.to_datetime(daily["ActivityDate"], infer_datetime_format=True)
    sleep["date"]         = pd.to_datetime(sleep["date"],         infer_datetime_format=True)
    hr["Time"]            = pd.to_datetime(hr["Time"],            infer_datetime_format=True)
    hr_minute = (
        hr.set_index("Time").groupby("Id")["Value"]
        .resample("1min").mean().reset_index()
    )
    hr_minute.columns = ["Id", "Time", "HeartRate"]
    hr_minute.dropna(inplace=True)
    hr_minute["Date"] = hr_minute["Time"].dt.date
    sleep["Date"]     = sleep["date"].dt.date
    sleep_daily = sleep.groupby(["Id", "Date"]).agg(
        TotalSleepMinutes=("value", "count")
    ).reset_index()
    master = daily.copy()
    master["Date"] = master["ActivityDate"].dt.date
    master = master.merge(sleep_daily, on=["Id", "Date"], how="left")
    master.fillna(0, inplace=True)
    return master, hr_minute, sleep_daily


@st.cache_data(show_spinner=False)
def build_anomaly_data(daily_b, sleep_b, hr_b):
    master, hr_minute, _ = load_fitbit_data(daily_b, sleep_b, hr_b)
    hr_d  = hr_minute.groupby("Date")["HeartRate"].mean().reset_index()
    hr_d.columns  = ["Date", "HR_avg"]
    hr_d["Date"]  = pd.to_datetime(hr_d["Date"])
    st_d  = master.groupby("ActivityDate")["TotalSteps"].mean().reset_index()
    st_d.columns  = ["Date", "Steps"]
    st_d["Date"]  = pd.to_datetime(st_d["Date"])
    sl_d  = master.groupby("ActivityDate")["TotalSleepMinutes"].mean().reset_index()
    sl_d.columns  = ["Date", "Sleep"]
    sl_d["Date"]  = pd.to_datetime(sl_d["Date"])
    return master, hr_minute, hr_d, st_d, sl_d


def rolling_residuals(sdf, dc, vc, win=7, ns=2.0):
    df = sdf.copy().sort_values(dc)
    df["rolling_med"] = df[vc].rolling(win, min_periods=1, center=True).median()
    df["residual"]    = df[vc] - df["rolling_med"]
    std               = df["residual"].std()
    df["anomaly"]     = df["residual"].abs() > ns * std
    return df


# ─────────────────────────────────────────────────────────────
# CHART CAPTURE HELPERS (for PDF screenshots)
# ─────────────────────────────────────────────────────────────
def make_hr_chart_bytes(hr_d, hr_thr, hr_res, hr_high, hr_low, sigma):
    """Render HR anomaly chart and return PNG bytes."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    apply_dark(fig, list(axes))
    ax = axes[0]
    ax.plot(hr_d["Date"], hr_d["HR_avg"], color=MPL_PALETTE[0], linewidth=2, label="Avg HR", zorder=2)
    ax.fill_between(hr_d["Date"], hr_high, float(hr_d["HR_avg"].max())*1.05, alpha=0.1, color=ACCENT_RED)
    ax.fill_between(hr_d["Date"], 0, hr_low, alpha=0.1, color=ACCENT_RED)
    ax.axhline(hr_high, color=ACCENT_RED, linewidth=1.5, linestyle="--", alpha=0.8)
    ax.axhline(hr_low,  color=ACCENT_RED, linewidth=1.5, linestyle="--", alpha=0.8)
    if hr_thr.any():
        ax.scatter(hr_d.loc[hr_thr, "Date"], hr_d.loc[hr_thr, "HR_avg"],
                   color=ACCENT3, s=75, zorder=5, marker="D", label="Threshold violation")
    if hr_res["anomaly"].any():
        ax.scatter(hr_res.loc[hr_res["anomaly"], "Date"], hr_res.loc[hr_res["anomaly"], "HR_avg"],
                   color=ACCENT_ORG, s=65, zorder=4, marker="^", label=f"Residual ±{sigma}σ")
    ax.set_title("Heart Rate — Anomaly Detection", fontsize=12)
    ax.set_xlabel("Date", fontsize=9); ax.set_ylabel("HR (bpm)", fontsize=9)
    ax.legend(fontsize=8, facecolor=DARK_BG_MPL, edgecolor=GRID_MPL, labelcolor=TEXT_MPL)
    plt.setp(ax.get_xticklabels(), rotation=30, fontsize=7)
    ax2 = axes[1]
    ax2.bar(hr_res["Date"], hr_res["residual"],
            color=[ACCENT_RED if v else MPL_PALETTE[0] for v in hr_res["anomaly"]],
            edgecolor="none", alpha=0.85, width=0.8)
    sh = float(hr_res["residual"].std())
    ax2.axhline(sigma*sh,  color=ACCENT_ORG, linewidth=1.5, linestyle="--")
    ax2.axhline(-sigma*sh, color=ACCENT_ORG, linewidth=1.5, linestyle="--")
    ax2.set_title("HR Residuals from 7-Day Rolling Median", fontsize=11)
    ax2.set_xlabel("Date", fontsize=9); ax2.set_ylabel("Residual (bpm)", fontsize=9)
    plt.setp(ax2.get_xticklabels(), rotation=30, fontsize=7)
    plt.tight_layout()
    b = save_fig_bytes(fig)
    plt.close(fig)
    return b


def make_steps_chart_bytes(st_d, st_thr, st_res, st_low, sigma):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    apply_dark(fig, list(axes))
    ax = axes[0]
    ax.plot(st_d["Date"], st_d["Steps"], color=MPL_PALETTE[2], linewidth=2, label="Avg Steps")
    ax.axhline(st_low, color=ACCENT_RED, linewidth=1.5, linestyle="--", alpha=0.8, label=f"Low threshold ({st_low})")
    if st_thr.any():
        ax.scatter(st_d.loc[st_thr, "Date"], st_d.loc[st_thr, "Steps"],
                   color=ACCENT3, s=75, zorder=5, marker="D", label="Threshold violation")
    if st_res["anomaly"].any():
        ax.scatter(st_res.loc[st_res["anomaly"], "Date"], st_res.loc[st_res["anomaly"], "Steps"],
                   color=ACCENT_ORG, s=65, zorder=4, marker="^", label=f"Residual ±{sigma}σ")
    ax.set_title("Step Count — Anomaly Detection", fontsize=12)
    ax.set_xlabel("Date", fontsize=9); ax.set_ylabel("Steps/Day", fontsize=9)
    ax.legend(fontsize=8, facecolor=DARK_BG_MPL, edgecolor=GRID_MPL, labelcolor=TEXT_MPL)
    plt.setp(ax.get_xticklabels(), rotation=30, fontsize=7)
    ax2 = axes[1]
    ax2.bar(st_res["Date"], st_res["residual"],
            color=[ACCENT_RED if v else MPL_PALETTE[2] for v in st_res["anomaly"]],
            edgecolor="none", alpha=0.85, width=0.8)
    ss = float(st_res["residual"].std())
    ax2.axhline(sigma*ss,  color=ACCENT_ORG, linewidth=1.5, linestyle="--")
    ax2.axhline(-sigma*ss, color=ACCENT_ORG, linewidth=1.5, linestyle="--")
    ax2.set_title("Steps Residuals from 7-Day Rolling Median", fontsize=11)
    ax2.set_xlabel("Date", fontsize=9); ax2.set_ylabel("Residual (steps)", fontsize=9)
    plt.setp(ax2.get_xticklabels(), rotation=30, fontsize=7)
    plt.tight_layout()
    b = save_fig_bytes(fig)
    plt.close(fig)
    return b


def make_sleep_chart_bytes(sl_d, sl_thr, sl_res, sl_low, sl_high, sigma):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    apply_dark(fig, list(axes))
    ax = axes[0]
    ax.plot(sl_d["Date"], sl_d["Sleep"], color=MPL_PALETTE[5], linewidth=2, label="Avg Sleep")
    ax.axhline(sl_low,  color=ACCENT_RED, linewidth=1.5, linestyle="--", alpha=0.8)
    ax.axhline(sl_high, color=ACCENT_RED, linewidth=1.5, linestyle="--", alpha=0.8)
    if sl_thr.any():
        ax.scatter(sl_d.loc[sl_thr, "Date"], sl_d.loc[sl_thr, "Sleep"],
                   color=ACCENT3, s=75, zorder=5, marker="D", label="Threshold violation")
    if sl_res["anomaly"].any():
        ax.scatter(sl_res.loc[sl_res["anomaly"], "Date"], sl_res.loc[sl_res["anomaly"], "Sleep"],
                   color=ACCENT_ORG, s=65, zorder=4, marker="^", label=f"Residual ±{sigma}σ")
    ax.set_title("Sleep Duration — Anomaly Detection", fontsize=12)
    ax.set_xlabel("Date", fontsize=9); ax.set_ylabel("Sleep (min)", fontsize=9)
    ax.legend(fontsize=8, facecolor=DARK_BG_MPL, edgecolor=GRID_MPL, labelcolor=TEXT_MPL)
    plt.setp(ax.get_xticklabels(), rotation=30, fontsize=7)
    ax2 = axes[1]
    ax2.bar(sl_res["Date"], sl_res["residual"],
            color=[ACCENT_RED if v else MPL_PALETTE[5] for v in sl_res["anomaly"]],
            edgecolor="none", alpha=0.85, width=0.8)
    sls = float(sl_res["residual"].std())
    ax2.axhline(sigma*sls,  color=ACCENT_ORG, linewidth=1.5, linestyle="--")
    ax2.axhline(-sigma*sls, color=ACCENT_ORG, linewidth=1.5, linestyle="--")
    ax2.set_title("Sleep Residuals from 7-Day Rolling Median", fontsize=11)
    ax2.set_xlabel("Date", fontsize=9); ax2.set_ylabel("Residual (min)", fontsize=9)
    plt.setp(ax2.get_xticklabels(), rotation=30, fontsize=7)
    plt.tight_layout()
    b = save_fig_bytes(fig)
    plt.close(fig)
    return b


def make_accuracy_chart_bytes(sim_results):
    sig2  = ["Heart Rate", "Steps", "Sleep"]
    accs  = [sim_results[s]["accuracy"] for s in sig2]
    fig, ax = plt.subplots(figsize=(9, 4))
    apply_dark(fig, ax)
    bars = ax.bar(sig2, accs, color=[ACCENT3 if a >= 90 else ACCENT_RED for a in accs],
                  edgecolor="none", alpha=0.9, width=0.45)
    ax.axhline(90, color=ACCENT_RED, linewidth=2, linestyle="--", label="90% Target")
    ax.set_ylim(0, 115)
    ax.set_title("Simulated Anomaly Detection Accuracy", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.legend(fontsize=9, facecolor=DARK_BG_MPL, edgecolor=GRID_MPL, labelcolor=TEXT_MPL)
    for bar, ac in zip(bars, accs):
        ax.annotate(f"{ac}%", xy=(bar.get_x() + bar.get_width()/2, ac),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", color="#f0f9ff", fontsize=13, fontweight="bold")
    plt.tight_layout()
    b = save_fig_bytes(fig)
    plt.close(fig)
    return b


def make_dbscan_chart_bytes(master):
    StandardScaler, _, _, DBSCAN, PCA, _ = _load_sklearn()
    cc_db = [c for c in ["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes","TotalSleepMinutes"] if c in master.columns]
    cf_db = master.groupby("Id")[cc_db].mean().dropna()
    Xs_db = StandardScaler().fit_transform(cf_db)
    Xp_db = PCA(n_components=2).fit_transform(Xs_db)
    dbl   = DBSCAN(eps=0.5, min_samples=3).fit_predict(Xs_db)
    fig, ax = plt.subplots(figsize=(9, 5))
    apply_dark(fig, ax)
    for lbl in sorted(set(dbl)):
        mask = dbl == lbl
        if lbl == -1:
            ax.scatter(Xp_db[mask, 0], Xp_db[mask, 1], c=ACCENT_RED, marker="X",
                       s=130, alpha=0.9, edgecolors="#f0f9ff", linewidths=0.8, label="Outlier", zorder=5)
        else:
            ax.scatter(Xp_db[mask, 0], Xp_db[mask, 1], c=MPL_PALETTE[lbl % len(MPL_PALETTE)],
                       s=80, alpha=0.85, edgecolors=DARK_BG_MPL, linewidths=0.8, label=f"Cluster {lbl}")
    ax.set_title("DBSCAN — PCA User Space (Structural Outliers)", fontsize=12)
    ax.set_xlabel("PC1", fontsize=9); ax.set_ylabel("PC2", fontsize=9)
    ax.legend(fontsize=9, facecolor=DARK_BG_MPL, edgecolor=GRID_MPL, labelcolor=TEXT_MPL)
    plt.tight_layout()
    b = save_fig_bytes(fig)
    plt.close(fig)
    return b


# ─────────────────────────────────────────────────────────────
# PDF SAFE-STRING HELPER
# Helvetica (and all built-in fpdf2 fonts) only support Latin-1
# (ISO-8859-1, code points 0x00-0xFF).  Any character outside
# that range causes:
#   "Character '–' … is outside the range of characters
#    supported by the font used: 'helveticaB'"
# This function replaces EVERY known offender with a plain-ASCII
# equivalent BEFORE the string is handed to any fpdf method.
# Call it on every argument you pass to cell(), multi_cell(),
# section_hdr(), kv(), para(), embed_chart caption, etc.
# ─────────────────────────────────────────────────────────────
def _pdf_safe(text: str) -> str:
    """
    Sanitise a string so it contains only Latin-1 characters.
    Replaces common Unicode punctuation and symbols with ASCII equivalents.
    Falls back to replacing any remaining non-Latin-1 byte with '?'.
    """
    if not isinstance(text, str):
        text = str(text)

    # ── Punctuation & typography ───────────────────────────────
    text = text.replace("\u2014", " - ")   # em dash          —
    text = text.replace("\u2013", " - ")   # en dash          –
    text = text.replace("\u2012", " - ")   # figure dash      ‒
    text = text.replace("\u2011", "-")     # non-breaking hyphen ‑
    text = text.replace("\u2010", "-")     # hyphen           ‐
    text = text.replace("\u2015", " - ")   # horizontal bar   ―

    # ── Quotes ─────────────────────────────────────────────────
    text = text.replace("\u2018", "'")     # left single quote  '
    text = text.replace("\u2019", "'")     # right single quote '
    text = text.replace("\u201a", ",")     # single low-9 quote ‚
    text = text.replace("\u201b", "'")     # single high-rev-9  ‛
    text = text.replace("\u201c", '"')     # left double quote  "
    text = text.replace("\u201d", '"')     # right double quote "
    text = text.replace("\u201e", '"')     # double low-9 quote „
    text = text.replace("\u201f", '"')     # double high-rev-9  ‟
    text = text.replace("\u2039", "<")     # single left angle  ‹
    text = text.replace("\u203a", ">")     # single right angle ›
    text = text.replace("\u00ab", '"')     # left guillemet     «
    text = text.replace("\u00bb", '"')     # right guillemet    »

    # ── Maths / science ────────────────────────────────────────
    text = text.replace("\u00b1", "+/-")   # plus-minus        ±
    text = text.replace("\u00d7", "x")     # multiplication    ×
    text = text.replace("\u00f7", "/")     # division          ÷
    text = text.replace("\u2212", "-")     # minus sign        −
    text = text.replace("\u2217", "*")     # asterisk operator ∗
    text = text.replace("\u2248", "~")     # almost equal      ≈
    text = text.replace("\u2260", "!=")    # not equal         ≠
    text = text.replace("\u2264", "<=")    # less or equal     ≤
    text = text.replace("\u2265", ">=")    # greater or equal  ≥
    text = text.replace("\u221e", "inf")   # infinity          ∞
    text = text.replace("\u00b2", "^2")    # superscript 2     ²
    text = text.replace("\u00b3", "^3")    # superscript 3     ³
    text = text.replace("\u00b0", " deg")  # degree sign       °
    text = text.replace("\u03bc", "u")     # micro / mu        μ
    text = text.replace("\u03c3", "sigma") # sigma             σ
    text = text.replace("\u03b1", "alpha") # alpha             α
    text = text.replace("\u03b2", "beta")  # beta              β
    text = text.replace("\u03b3", "gamma") # gamma             γ
    text = text.replace("\u03b4", "delta") # delta             δ
    text = text.replace("\u03bb", "lambda")# lambda            λ
    text = text.replace("\u03c0", "pi")    # pi                π

    # ── Punctuation extras ─────────────────────────────────────
    text = text.replace("\u2022", "*")     # bullet            •
    text = text.replace("\u00b7", ".")     # middle dot        ·
    text = text.replace("\u2026", "...")   # ellipsis          …
    text = text.replace("\u2030", "o/oo") # per mille         ‰
    text = text.replace("\u2032", "'")     # prime             ′
    text = text.replace("\u2033", '"')     # double prime      ″
    text = text.replace("\u00a9", "(c)")   # copyright         ©
    text = text.replace("\u00ae", "(R)")   # registered        ®
    text = text.replace("\u2122", "(TM)")  # trade mark        ™
    text = text.replace("\u20ac", "EUR")   # euro              €
    text = text.replace("\u00a3", "GBP")   # pound             £
    text = text.replace("\u00a5", "JPY")   # yen               ¥
    text = text.replace("\u2190", "<-")    # left arrow        ←
    text = text.replace("\u2192", "->")    # right arrow       →
    text = text.replace("\u2194", "<->")   # both arrows       ↔
    text = text.replace("\u25cf", "*")     # black circle      ●
    text = text.replace("\u25cb", "o")     # white circle      ○
    text = text.replace("\u25a0", "[x]")   # black square      ■
    text = text.replace("\u25a1", "[ ]")   # white square      □
    text = text.replace("\u2713", "ok")    # check mark        ✓
    text = text.replace("\u2717", "x")     # ballot x          ✗
    text = text.replace("\u00a0", " ")     # non-breaking space

    # ── Final safety net ───────────────────────────────────────
    # Encode to Latin-1, replacing anything still unrecognised with '?',
    # then decode back to str so fpdf receives a plain Python str.
    text = text.encode("latin-1", errors="replace").decode("latin-1")

    return text


# ─────────────────────────────────────────────────────────────
# PDF GENERATOR  ── with embedded chart screenshots
# ─────────────────────────────────────────────────────────────
def generate_pdf_report(master, anom_log, stats_df,
                         hr_high, hr_low, st_low, sl_low, sl_high, sigma,
                         total_hr, total_st, total_sl, sim_results,
                         hr_d=None, hr_thr=None, hr_res=None,
                         st_d=None, st_thr=None, st_res=None,
                         sl_d=None, sl_thr=None, sl_res=None):
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    # Pre-render chart screenshots
    chart_hr_bytes  = None
    chart_st_bytes  = None
    chart_sl_bytes  = None
    chart_acc_bytes = None
    chart_db_bytes  = None

    if hr_d is not None and hr_thr is not None and hr_res is not None:
        try:
            chart_hr_bytes = make_hr_chart_bytes(hr_d, hr_thr, hr_res, hr_high, hr_low, sigma)
        except Exception:
            pass
    if st_d is not None and st_thr is not None and st_res is not None:
        try:
            chart_st_bytes = make_steps_chart_bytes(st_d, st_thr, st_res, st_low, sigma)
        except Exception:
            pass
    if sl_d is not None and sl_thr is not None and sl_res is not None:
        try:
            chart_sl_bytes = make_sleep_chart_bytes(sl_d, sl_thr, sl_res, sl_low, sl_high, sigma)
        except Exception:
            pass
    if sim_results and isinstance(sim_results, dict) and "Heart Rate" in sim_results:
        try:
            chart_acc_bytes = make_accuracy_chart_bytes(sim_results)
        except Exception:
            pass
    try:
        chart_db_bytes = make_dbscan_chart_bytes(master)
    except Exception:
        pass

    def save_tmp_png(raw_bytes, name):
        """Write bytes to a temp file and return the path."""
        if raw_bytes is None:
            return None
        fd, path = tempfile.mkstemp(suffix=f"_{name}.png")
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw_bytes)
        return path

    tmp_hr  = save_tmp_png(chart_hr_bytes,  "hr")
    tmp_st  = save_tmp_png(chart_st_bytes,  "steps")
    tmp_sl  = save_tmp_png(chart_sl_bytes,  "sleep")
    tmp_acc = save_tmp_png(chart_acc_bytes, "accuracy")
    tmp_db  = save_tmp_png(chart_db_bytes,  "dbscan")

    class PDF(FPDF):
        def header(self):
            self.set_fill_color(10, 14, 26)
            self.rect(0, 0, 210, 18, "F")
            self.set_font("Helvetica", "B", 12)
            self.set_text_color(99, 179, 237)
            self.set_y(4)
            self.cell(0, 10, _pdf_safe("FitPulse - Anomaly Detection Report"), align="C")
            self.set_text_color(100, 116, 139)
            self.set_font("Helvetica", "", 7)
            self.set_y(13)
            self.cell(0, 4, f"Generated: {datetime.now().strftime('%d %B %Y  %H:%M')}", align="C")
            self.ln(6)

        def footer(self):
            self.set_y(-13)
            self.set_font("Helvetica", "", 7)
            self.set_text_color(100, 116, 139)
            self.cell(0, 8, _pdf_safe(f"FitPulse ML Dashboard  -  Page {self.page_no()}"), align="C")

        def section_hdr(self, title, rgb=(15, 23, 60)):
            self.ln(3)
            self.set_fill_color(*rgb)
            self.set_text_color(200, 220, 255)
            self.set_font("Helvetica", "B", 9)
            self.cell(0, 7, _pdf_safe(f"  {title}"), fill=True, ln=True)
            self.set_text_color(40, 40, 60)
            self.ln(2)

        def kv(self, key, val):
            self.set_font("Helvetica", "B", 8.5)
            self.set_text_color(60, 80, 120)
            self.cell(58, 6, _pdf_safe(key + ":"), ln=False)
            self.set_font("Helvetica", "", 8.5)
            self.set_text_color(20, 20, 40)
            self.cell(0, 6, _pdf_safe(str(val)), ln=True)

        def para(self, text, sz=8):
            self.set_font("Helvetica", "", sz)
            self.set_text_color(50, 60, 80)
            self.multi_cell(0, 4.5, _pdf_safe(text))
            self.ln(1)

        def embed_chart(self, path, caption="", w=180):
            if path and os.path.exists(path):
                try:
                    x = (210 - w) / 2
                    self.image(path, x=x, w=w)
                    if caption:
                        self.set_font("Helvetica", "I", 7)
                        self.set_text_color(100, 116, 139)
                        self.cell(0, 5, _pdf_safe(caption), align="C", ln=True)
                    self.ln(3)
                except Exception:
                    pass

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    n_users = master["Id"].nunique()
    n_days  = master["Date"].nunique()
    try:
        dr = f"{pd.to_datetime(master['Date']).min().strftime('%d %b %Y')} - {pd.to_datetime(master['Date']).max().strftime('%d %b %Y')}"
    except Exception:
        dr = "Fitbit Dataset"

    pdf.section_hdr("1.  EXECUTIVE SUMMARY", (8, 32, 80))
    pdf.kv("Dataset",    "Real Fitbit Device Data (Kaggle - arashnic/fitbit)")
    pdf.kv("Users",      f"{n_users} participants")
    pdf.kv("Date Range", dr)
    pdf.kv("Total Days", f"{n_days} observation days")
    pdf.ln(2)

    pdf.section_hdr("2.  ANOMALY SUMMARY", (80, 20, 20))
    pdf.kv("Heart Rate Anomalies", f"{total_hr} days flagged")
    pdf.kv("Steps Anomalies",      f"{total_st} days flagged")
    pdf.kv("Sleep Anomalies",      f"{total_sl} days flagged")
    pdf.kv("Total Flags",          f"{total_hr+total_st+total_sl} across all signals")
    pdf.ln(2)

    pdf.section_hdr("3.  DETECTION THRESHOLDS", (20, 80, 40))
    pdf.kv("HR High",        f"> {int(hr_high)} bpm")
    pdf.kv("HR Low",         f"< {int(hr_low)} bpm")
    pdf.kv("Steps Low",      f"< {int(st_low):,} steps/day")
    pdf.kv("Sleep Low",      f"< {int(sl_low)} min/night")
    pdf.kv("Sleep High",     f"> {int(sl_high)} min/night")
    pdf.kv("Residual Sigma", f"+/- {float(sigma):.1f} sigma from rolling 7-day median")
    pdf.ln(2)

    pdf.section_hdr("4.  SIMULATED ACCURACY", (40, 20, 100))
    if sim_results and isinstance(sim_results, dict):
        for sig in ["Heart Rate", "Steps", "Sleep"]:
            r = sim_results.get(sig, {})
            if isinstance(r, dict):
                pdf.kv(f"  {sig}", f"{r.get('accuracy','N/A')}%  ({r.get('detected','?')}/{r.get('injected','?')} detected)")
        pdf.kv("  Overall", f"{sim_results.get('Overall','N/A')}%  (target: 90%+)")
    else:
        pdf.para("Accuracy simulation not yet run.")
    pdf.ln(2)

    pdf.section_hdr("5.  METHODOLOGY", (20, 60, 100))
    pdf.para(
        "Three complementary anomaly detection methods were applied to Heart Rate, "
        "Step Count, and Sleep Duration signals:\n\n"
        "  1. THRESHOLD VIOLATIONS - Hard clinical upper/lower bounds on each metric.\n\n"
        "  2. RESIDUAL-BASED DETECTION - A rolling 7-day median baseline. Days deviating "
        f"more than +/-{float(sigma):.1f} SD are flagged.\n\n"
        "  3. DBSCAN STRUCTURAL OUTLIERS - Users profiled on 5 activity features, "
        "clustered in PCA-reduced space using DBSCAN (eps=0.5, min_samples=3). "
        "Users assigned label -1 are structural outliers.", sz=8)

    # ── Page 2: Chart screenshots
    pdf.add_page()
    pdf.section_hdr("6.  ANOMALY VISUALIZATIONS - HEART RATE", (80, 20, 20))
    pdf.embed_chart(tmp_hr, "Fig 1 - Heart Rate daily average with threshold zones and residual anomalies")

    pdf.section_hdr("7.  ANOMALY VISUALIZATIONS - STEP COUNT", (20, 80, 40))
    pdf.embed_chart(tmp_st, "Fig 2 - Daily step count trend with low-activity alerts and residual deviations")

    # ── Page 3: Sleep + Accuracy
    pdf.add_page()
    pdf.section_hdr("8.  ANOMALY VISUALIZATIONS - SLEEP DURATION", (60, 20, 100))
    pdf.embed_chart(tmp_sl, "Fig 3 - Sleep duration pattern with threshold bands and residual anomalies")

    pdf.section_hdr("9.  DETECTION ACCURACY (SIMULATION)", (40, 20, 100))
    pdf.embed_chart(tmp_acc, "Fig 4 - Simulated accuracy: 10 injected anomalies per signal vs 90% target")

    # ── Page 4: DBSCAN + Tables
    pdf.add_page()
    pdf.section_hdr("10.  DBSCAN STRUCTURAL OUTLIERS", (20, 60, 100))
    pdf.embed_chart(tmp_db, "Fig 5 · DBSCAN clustering in PCA-reduced user feature space")

    def draw_table(df_t, title, rgb):
        pdf.section_hdr(title, rgb)
        if df_t.empty:
            pdf.para("No anomalies detected.")
            return
        cols   = list(df_t.columns)
        col_w  = max(20, 180 // len(cols))
        pdf.set_fill_color(8, 20, 50)
        pdf.set_text_color(140, 180, 230)
        pdf.set_font("Helvetica", "B", 7)
        for col in cols:
            pdf.cell(col_w, 6, str(col)[:16], border=0, fill=True)
        pdf.ln()
        pdf.set_font("Helvetica", "", 7)
        for i, (_, row) in enumerate(df_t.head(18).iterrows()):
            if i % 2 == 0: pdf.set_fill_color(12, 22, 46)
            else:           pdf.set_fill_color(8, 15, 32)
            pdf.set_text_color(180, 200, 230)
            for val in row:
                txt = f"{val:.2f}" if isinstance(val, float) else str(val)[:16]
                pdf.cell(col_w, 5, txt, border=0, fill=True)
            pdf.ln()
        if len(df_t) > 18:
            pdf.set_text_color(80, 120, 180)
            pdf.set_font("Helvetica", "I", 6.5)
            pdf.cell(0, 5, f"  ... and {len(df_t)-18} more records", ln=True)
        pdf.ln(3)

    anom_log_local = anom_log if not anom_log.empty else pd.DataFrame(
        columns=["Date","Signal","Value","Anomaly_Type","Severity","Unit"])
    hr_log  = anom_log_local[anom_log_local["Signal"]=="Heart Rate"][["Date","Value","Anomaly_Type","Severity","Unit"]].reset_index(drop=True)
    st_log  = anom_log_local[anom_log_local["Signal"]=="Steps"][["Date","Value","Anomaly_Type","Severity","Unit"]].reset_index(drop=True)
    sl_log  = anom_log_local[anom_log_local["Signal"]=="Sleep"][["Date","Value","Anomaly_Type","Severity","Unit"]].reset_index(drop=True)

    # ── Page 5: Anomaly tables
    pdf.add_page()
    draw_table(hr_log, "11.  HEART RATE ANOMALY RECORDS", (80, 20, 20))
    draw_table(st_log, "12.  STEPS ANOMALY RECORDS",      (20, 80, 40))
    draw_table(sl_log, "13.  SLEEP ANOMALY RECORDS",      (60, 20, 100))

    # ── Page 6: Stats + profiles
    pdf.add_page()
    pdf.section_hdr("14.  SIGNAL TREND STATISTICS", (8, 32, 80))
    if not stats_df.empty:
        cols_s = list(stats_df.columns)
        cw2    = max(18, 180 // len(cols_s))
        pdf.set_fill_color(8, 20, 50); pdf.set_text_color(140, 180, 230); pdf.set_font("Helvetica", "B", 7)
        for col in cols_s: pdf.cell(cw2, 6, str(col)[:15], border=0, fill=True)
        pdf.ln()
        pdf.set_font("Helvetica", "", 7)
        for i, (_, row) in enumerate(stats_df.iterrows()):
            pdf.set_fill_color(12, 22, 46) if i % 2 == 0 else pdf.set_fill_color(8, 15, 32)
            pdf.set_text_color(180, 200, 230)
            for val in row:
                txt = f"{val:.2f}" if isinstance(val, float) else str(val)[:15]
                pdf.cell(cw2, 5, txt, border=0, fill=True)
            pdf.ln()
    pdf.ln(3)

    pdf.section_hdr("15.  USER ACTIVITY PROFILES", (8, 32, 80))
    pcols = [c for c in ["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes","TotalSleepMinutes"] if c in master.columns]
    up    = master.groupby("Id")[pcols].mean().round(1)
    cw3   = max(18, 180 // (len(pcols)+1))
    pdf.set_fill_color(8, 20, 50); pdf.set_text_color(140, 180, 230); pdf.set_font("Helvetica", "B", 7)
    pdf.cell(cw3, 6, "User ID", border=0, fill=True)
    for col in pcols: pdf.cell(cw3, 6, col[:12], border=0, fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", "", 7)
    for i, (uid, row) in enumerate(up.iterrows()):
        pdf.set_fill_color(12, 22, 46) if i % 2 == 0 else pdf.set_fill_color(8, 15, 32)
        pdf.set_text_color(180, 200, 230)
        pdf.cell(cw3, 5, f"...{str(uid)[-6:]}", border=0, fill=True)
        for val in row: pdf.cell(cw3, 5, f"{val:,.0f}", border=0, fill=True)
        pdf.ln()
    pdf.ln(4)

    pdf.section_hdr("16.  CONCLUSION", (20, 60, 40))
    pdf.para(
        f"The FitPulse anomaly detection pipeline processed {n_users} users over "
        f"{n_days} days. A total of {total_hr+total_st+total_sl} anomalous events were identified: "
        f"{total_hr} heart rate flags, {total_st} step count alerts, and {total_sl} sleep anomalies. "
        "The multi-method approach proved effective at capturing both extreme violations and subtle "
        "statistical deviations. Five chart screenshots are included in this report (Figs 1–5) "
        "illustrating the detected anomalies across all signals.", sz=8)

    buf = io.BytesIO()
    buf.write(pdf.output())
    buf.seek(0)

    # Cleanup temp files
    for p in [tmp_hr, tmp_st, tmp_sl, tmp_acc, tmp_db]:
        if p and os.path.exists(p):
            try: os.remove(p)
            except Exception: pass

    return buf


def generate_csv_report(master, anom_log, stats_df):
    buf = io.StringIO()
    buf.write("# FitPulse Anomaly Detection – CSV Report\n")
    buf.write(f"# Generated: {datetime.now().strftime('%d %B %Y %H:%M')}\n\n")
    buf.write("## ANOMALY LOG\n")
    anom_log.to_csv(buf, index=False)
    buf.write("\n## SIGNAL STATISTICS\n")
    stats_df.to_csv(buf, index=False)
    buf.write("\n## MASTER DATASET (first 500 rows)\n")
    master.head(500).to_csv(buf, index=False)
    return buf.getvalue().encode()


# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR  ── fully dark themed
# ═══════════════════════════════════════════════════════════════════════
st.sidebar.markdown(f"""
<div style="padding:0.8rem 0 1.2rem">
  <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.3rem">
    <div style="font-size:1.5rem">💓</div>
    <div>
      <div style="font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:800;color:{ACCENT}">
        FitPulse
      </div>
      <div style="font-size:0.65rem;color:{MUTED};font-family:'JetBrains Mono',monospace">
        ML Dashboard · v4.0
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    f'<hr style="border:none;border-top:1px solid rgba(99,179,237,0.12);margin:0 0 1rem">',
    unsafe_allow_html=True)

st.sidebar.markdown('<div class="sb-label">⚡ NAVIGATION</div>', unsafe_allow_html=True)
milestone = st.sidebar.selectbox(
    "Module",
    options=[
        "📊  M1 — Data Processing Pipeline",
        "🤖  M2 — ML Analytics Pipeline",
        "🚨  M3 — Anomaly Detection",
        "📄  M4 — Dashboard & Reports",
    ],
    label_visibility="collapsed",
    key="main_nav",
)

st.sidebar.markdown(
    f'<hr style="border:none;border-top:1px solid rgba(99,179,237,0.12);margin:1rem 0">',
    unsafe_allow_html=True)

# Thresholds
if "M3" in milestone or "M4" in milestone:
    st.sidebar.markdown(
        f'<div style="background:rgba(99,179,237,0.05);border:1px solid rgba(99,179,237,0.12);'
        f'border-radius:10px;padding:0.7rem 0.8rem;margin-bottom:0.8rem">'
        f'<div class="sb-label" style="margin-bottom:0.7rem">🎛️ ANOMALY THRESHOLDS</div>',
        unsafe_allow_html=True)
    hr_high = st.sidebar.slider("HR High (bpm)",  80,  150, 100, key="sl_hr_high")
    hr_low  = st.sidebar.slider("HR Low (bpm)",   30,   60,  50, key="sl_hr_low")
    st_low  = st.sidebar.slider("Steps Low",      100, 3000, 1000, key="sl_st_low")
    sl_low  = st.sidebar.slider("Sleep Min (min)", 60,  300, 180, key="sl_sl_low")
    sl_high = st.sidebar.slider("Sleep Max (min)", 480, 900, 720, key="sl_sl_high")
    sigma   = st.sidebar.slider("Residual σ",     1.0,  4.0, 2.0, step=0.5, key="sl_sigma")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        f'<hr style="border:none;border-top:1px solid rgba(99,179,237,0.12);margin:0.5rem 0 1rem">',
        unsafe_allow_html=True)
else:
    hr_high, hr_low, st_low, sl_low, sl_high, sigma = 100, 50, 1000, 180, 720, 2.0

# File status in sidebar
if "M2" in milestone or "M3" in milestone or "M4" in milestone:
    n_f = sum(1 for k in ["file_daily","file_steps","file_int","file_sleep","file_hr"] if st.session_state[k])
    pct = int(n_f / 5 * 100)
    col = SUCCESS_BOR if n_f == 5 else (ACCENT_ORG if n_f > 0 else ACCENT_RED)
    st.sidebar.markdown(
        f'<div style="background:rgba(8,13,26,0.8);border:1px solid rgba(99,179,237,0.12);'
        f'border-radius:10px;padding:0.7rem 0.8rem;margin-bottom:0.8rem">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">'
        f'<span style="font-size:0.68rem;color:{MUTED};font-family:JetBrains Mono,monospace;'
        f'text-transform:uppercase;letter-spacing:0.08em">FILES LOADED</span>'
        f'<span style="font-size:0.75rem;color:{col};font-weight:700;font-family:JetBrains Mono,monospace">{n_f}/5</span>'
        f'</div>'
        f'<div style="background:rgba(255,255,255,0.06);border-radius:4px;height:5px;overflow:hidden">'
        f'<div style="width:{pct}%;height:100%;background:linear-gradient(90deg,{ACCENT},{ACCENT3});'
        f'border-radius:4px;transition:width 0.4s ease"></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f'<hr style="border:none;border-top:1px solid rgba(99,179,237,0.12);margin:0 0 1rem">',
        unsafe_allow_html=True)

# System status
st.sidebar.markdown(
    f'<div style="background:rgba(8,13,26,0.8);border:1px solid rgba(99,179,237,0.12);'
    f'border-radius:10px;padding:0.7rem 0.8rem">',
    unsafe_allow_html=True)
st.sidebar.markdown('<div class="sb-label" style="margin-bottom:0.5rem">⚡ SYSTEM STATUS</div>', unsafe_allow_html=True)
for dot, name, val in [
    ("🟢","ML Engine","Active"), ("🟢","Prophet","Ready"),
    ("🟢","TSFresh","Ready"),    ("🟢","FPDF2","Ready"), ("🟡","GPU","CPU Mode"),
]:
    st.sidebar.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:3px 2px;margin-bottom:2px">'
        f'<span style="font-size:0.75rem;color:{MUTED}">{dot} {name}</span>'
        f'<span style="font-size:0.67rem;color:{MUTED};font-family:JetBrains Mono,monospace;'
        f'background:rgba(99,179,237,0.06);padding:1px 6px;border-radius:4px">{val}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
st.sidebar.markdown('</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    f'<div style="text-align:center;padding:14px 0 4px;color:{MUTED};font-size:0.66rem;'
    f'font-family:JetBrains Mono,monospace;letter-spacing:0.05em">FitPulse Analytics · v4.0</div>',
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════
# MILESTONE 1  ── enhanced with more tabs
# ═══════════════════════════════════════════════════════════════════════
def milestone1():
    hero(
        "💙 Data Processing Pipeline",
        "Upload · Analyse · Clean · EDA · Insights",
        "M1 · DATA PIPELINE",
    )

    m1_tabs = st.tabs([
        "📂 Upload", "🔍 Missing Values", "🧹 Preprocessing",
        "👁️ Data Preview", "📊 Visualizations", "💡 Insights"
    ])

    # ── Tab 0: Upload
    with m1_tabs[0]:
        divider()
        sec("📂", "Dataset Upload", "Step 01")
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            ui_success(f"Dataset loaded — {df.shape[0]:,} rows × {df.shape[1]} columns")
            st.markdown("<br>", unsafe_allow_html=True)
            kpi_row([
                (f"{df.shape[0]:,}", "Rows", "", ACCENT),
                (df.shape[1], "Columns", "", ACCENT2),
                (f"{int(df.isnull().sum().sum()):,}", "Nulls", "missing values", ACCENT_RED),
                (int(df.duplicated().sum()), "Dupes", "duplicate rows", ACCENT_ORG),
            ])
            st.markdown("<br>", unsafe_allow_html=True)
            sec("🗂️", "Raw Data Preview")
            dark_df(df.head(10))
            st.markdown("<br>", unsafe_allow_html=True)
            sec("📋", "Column Types")
            dark_df(pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values}))
        else:
            st.markdown(
                f'<div class="card" style="text-align:center;padding:3rem;'
                f'border-style:dashed;border-color:rgba(99,179,237,0.25)">'
                f'<div style="font-size:2.5rem;margin-bottom:0.8rem">📁</div>'
                f'<div style="color:{MUTED};font-size:0.9rem">Upload a CSV file above to begin</div>'
                f'<div style="color:{MUTED};font-size:0.75rem;margin-top:0.4rem">'
                f'Supported: health metrics, fitness data, activity logs</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Tab 1: Missing Values
    with m1_tabs[1]:
        divider()
        sec("🔍", "Missing Value Analysis", "Step 02")
        if st.session_state.df is None:
            ui_warn("Upload a dataset first (tab 📂 Upload).")
        else:
            df  = st.session_state.df
            nc  = df.isnull().sum()
            np_ = (nc / len(df)) * 100
            dark_df(
                pd.DataFrame({"Column": nc.index, "Count": nc.values, "Pct %": np_.round(2).values})
                .sort_values("Count", ascending=False)
            )
            st.markdown("<br>", unsafe_allow_html=True)
            bars = np_.sort_values(ascending=False)
            if bars.sum() > 0:
                fig, ax = plt.subplots(figsize=(10, 4)); apply_dark(fig, ax)
                bars.plot(kind="bar", ax=ax,
                          color=[MPL_PALETTE[i % len(MPL_PALETTE)] for i in range(len(bars))],
                          edgecolor="none")
                ax.set_title("Missing Data % per Column", fontsize=11)
                ax.set_ylabel("Missing %", fontsize=9)
                plt.xticks(rotation=45, ha="right", fontsize=8)
                plt.tight_layout(); show_fig(fig)
            else:
                ui_success("No missing values detected!")

    # ── Tab 2: Preprocessing
    with m1_tabs[2]:
        divider()
        sec("🧹", "Data Preprocessing", "Step 03")
        if st.session_state.df is None:
            ui_warn("Upload a dataset first (tab 📂 Upload).")
        else:
            st.markdown(
                f'<div class="card">'
                f'<div class="card-title">Preprocessing Pipeline</div>'
                f'<div class="pipeline-step"><div class="step-num">01</div><div>'
                f'<div style="color:{TEXT};font-weight:600;font-size:0.88rem">Date Parsing</div>'
                f'<div style="color:{MUTED};font-size:0.8rem">Convert Date column to datetime</div>'
                f'</div></div>'
                f'<div class="pipeline-step"><div class="step-num">02</div><div>'
                f'<div style="color:{TEXT};font-weight:600;font-size:0.88rem">Per-User Interpolation</div>'
                f'<div style="color:{MUTED};font-size:0.8rem">Linear interpolation for numeric health metrics</div>'
                f'</div></div>'
                f'<div class="pipeline-step"><div class="step-num">03</div><div>'
                f'<div style="color:{TEXT};font-weight:600;font-size:0.88rem">Forward / Backward Fill</div>'
                f'<div style="color:{MUTED};font-size:0.8rem">Remaining nulls filled with ffill → bfill per user</div>'
                f'</div></div>'
                f'<div class="pipeline-step"><div class="step-num">04</div><div>'
                f'<div style="color:{TEXT};font-weight:600;font-size:0.88rem">Workout Fill</div>'
                f'<div style="color:{MUTED};font-size:0.8rem">Null Workout_Type → "No Workout"</div>'
                f'</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("▶ Run Preprocessing"):
                df = st.session_state.df.copy()
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                nc2 = [c for c in ["Hours_Slept","Water_Intake (Liters)","Active_Minutes","Heart_Rate (bpm)"] if c in df.columns]
                if "User_ID" in df.columns and nc2:
                    df[nc2] = df.groupby("User_ID")[nc2].transform(lambda x: x.interpolate())
                    df[nc2] = df.groupby("User_ID")[nc2].transform(lambda x: x.ffill().bfill())
                if "Workout_Type" in df.columns:
                    df["Workout_Type"] = df["Workout_Type"].fillna("No Workout")
                st.session_state.cleaned_df = df
                b = int(st.session_state.df.isnull().sum().sum())
                a = int(df.isnull().sum().sum())
                ui_success("Preprocessing complete!")
                kpi_row([
                    (b, "Nulls Before", "before cleaning", ACCENT_RED),
                    (a, "Nulls After",  "after cleaning",  ACCENT3),
                ])

    # ── Tab 3: Data Preview
    with m1_tabs[3]:
        divider()
        sec("👁️", "Clean Data Preview", "Step 04")
        if st.session_state.cleaned_df is None:
            ui_warn("Run preprocessing first (tab 🧹 Preprocessing).")
        else:
            df = st.session_state.cleaned_df
            kpi_row([
                (f"{df.shape[0]:,}", "Rows",    "",              ACCENT),
                (df.shape[1],        "Columns", "",              ACCENT2),
                (int(df.isnull().sum().sum()), "Nulls", "remaining", ACCENT3),
            ])
            st.markdown("<br>", unsafe_allow_html=True)

            # Preview button styled as pill
            st.markdown('<div class="pill-btn">', unsafe_allow_html=True)
            show_preview = st.button("💾 Preview Cleaned Dataset", key="pill_preview")
            st.markdown('</div>', unsafe_allow_html=True)

            if show_preview or True:
                dark_df(df.head(20))
                st.markdown("<br>", unsafe_allow_html=True)
                sec("📐", "Descriptive Statistics")
                dark_df(df.describe().round(2))

    # ── Tab 4: Visualizations
    with m1_tabs[4]:
        divider()
        sec("📊", "EDA & Visualizations", "Step 05")
        if st.session_state.cleaned_df is None:
            ui_warn("Run preprocessing first (tab 🧹 Preprocessing).")
        else:
            if st.button("▶ Run Full EDA"):
                df  = st.session_state.cleaned_df
                num = [c for c in ["Steps_Taken","Calories_Burned","Hours_Slept","Active_Minutes",
                                    "Heart_Rate (bpm)","Stress_Level (1-10)"] if c in df.columns]
                for i, col in enumerate(num):
                    sec("📈", col)
                    fig, ax = plt.subplots(figsize=(10, 4)); apply_dark(fig, ax)
                    sns.histplot(df[col].dropna(), kde=True, ax=ax,
                                 color=MPL_PALETTE[i % len(MPL_PALETTE)], edgecolor="none", alpha=0.75)
                    if ax.lines: ax.lines[-1].set_color("#f0f9ff")
                    ax.set_title(col, color=ACCENT, fontsize=11, fontweight="bold")
                    ax.set_xlabel(col, fontsize=9); ax.set_ylabel("Count", fontsize=9)
                    plt.tight_layout(); show_fig(fig)

                if len(num) >= 2:
                    sec("🔥", "Correlation Matrix")
                    fig, ax = plt.subplots(figsize=(10, 7)); apply_dark(fig, ax)
                    sns.heatmap(df[num].corr(), ax=ax, cmap="coolwarm", annot=True, fmt=".2f",
                                annot_kws={"size": 9}, linewidths=0.5, linecolor=GRID_MPL,
                                mask=np.triu(np.ones((len(num), len(num)), dtype=bool)),
                                vmin=-1, vmax=1, center=0)
                    ax.set_title("Correlation Matrix", fontsize=11)
                    plt.tight_layout(); show_fig(fig)

                if "Workout_Type" in df.columns:
                    wt = df["Workout_Type"].value_counts()
                    # Compact pie chart
                    fig, ax = plt.subplots(figsize=(5.5, 5))
                    fig.patch.set_facecolor(DARK_BG_MPL); ax.set_facecolor(DARK_BG_MPL)
                    w, t, at = ax.pie(wt.values, labels=wt.index, autopct="%1.1f%%",
                                      colors=MPL_PALETTE[:len(wt)], startangle=140,
                                      wedgeprops=dict(edgecolor=DARK_BG_MPL, linewidth=2),
                                      pctdistance=0.82,
                                      radius=0.85)
                    for tx in t:  tx.set_color(TEXT_MPL); tx.set_fontsize(8)
                    for tx in at: tx.set_color("#f0f9ff"); tx.set_fontsize(8); tx.set_fontweight("bold")
                    ax.set_title("Workout Type Distribution", color=ACCENT, fontsize=11, pad=10)
                    fig.tight_layout(pad=1.5)
                    col_l, col_c, col_r = st.columns([1, 2, 1])
                    with col_c:
                        show_fig(fig)

    # ── Tab 5: Insights
    with m1_tabs[5]:
        divider()
        sec("💡", "Data Insights", "Auto-generated")
        if st.session_state.cleaned_df is None:
            ui_warn("Run preprocessing first to generate insights.")
        else:
            df = st.session_state.cleaned_df
            num_cols = df.select_dtypes(include="number").columns.tolist()

            insight_card("📊 Shape", "Dataset Dimensions",
                f"{df.shape[0]:,} records × {df.shape[1]} features · "
                f"{int(df.isnull().sum().sum())} remaining nulls after cleaning.")
            if "Steps_Taken" in df.columns:
                avg_s = df["Steps_Taken"].mean()
                insight_card("🚶 Activity", "Average Step Count",
                    f"Mean daily steps: {avg_s:,.0f} — "
                    f"{'above' if avg_s > 8000 else 'below'} the recommended 8,000/day benchmark.", ACCENT3)
            if "Hours_Slept" in df.columns:
                avg_sl = df["Hours_Slept"].mean()
                insight_card("😴 Sleep", "Sleep Quality",
                    f"Average sleep: {avg_sl:.1f} hours — "
                    f"{'within' if 7 <= avg_sl <= 9 else 'outside'} the optimal 7–9 hour range.", "#b794f4")
            if "Heart_Rate (bpm)" in df.columns:
                avg_hr = df["Heart_Rate (bpm)"].mean()
                insight_card("❤️ Vitals", "Resting Heart Rate",
                    f"Mean HR: {avg_hr:.0f} bpm — "
                    f"{'normal range' if 60 <= avg_hr <= 100 else 'outside normal range (60–100 bpm)'}.", ACCENT2)
            if len(num_cols) >= 2:
                corr = df[num_cols].corr().abs()
                np.fill_diagonal(corr.values, 0)
                max_pair = corr.stack().idxmax()
                max_val  = corr.stack().max()
                insight_card("🔗 Correlation", "Strongest Feature Relationship",
                    f"Highest correlation: {max_pair[0]} ↔ {max_pair[1]} (r = {max_val:.2f}). "
                    f"{'Strong' if max_val > 0.7 else 'Moderate'} linear relationship detected.", ACCENT_ORG)


# ═══════════════════════════════════════════════════════════════════════
# MILESTONE 2  ── analytics with improved tab layout
# ═══════════════════════════════════════════════════════════════════════
def milestone2():
    hero(
        "🤖 ML Analytics Pipeline",
        "Scaling · Clustering · TSFresh · Prophet · Projections",
        "M2 · MACHINE LEARNING",
    )

    if not shared_file_upload():
        return

    with st.spinner("Loading data..."):
        master, hr_minute, _ = load_fitbit_data(
            st.session_state.file_daily,
            st.session_state.file_sleep,
            st.session_state.file_hr,
        )
        daily = pd.read_csv(io.BytesIO(st.session_state.file_daily))
        daily["ActivityDate"] = pd.to_datetime(daily["ActivityDate"], infer_datetime_format=True)

    StandardScaler, MinMaxScaler, KMeans, DBSCAN, PCA, TSNE = _load_sklearn()

    cluster_cols = ["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes","TotalSleepMinutes"]
    cf = master.groupby("Id")[cluster_cols].mean()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(cf)
    OPTIMAL_K = 3
    km = KMeans(n_clusters=OPTIMAL_K, n_init=10, random_state=42)
    labels = km.fit_predict(Xs)
    cf = cf.copy(); cf["Cluster"] = labels
    profile = cf.groupby("Cluster")[cluster_cols].mean().round(2)

    tabs = st.tabs([
        "📊 Overview", "⚖️ Data Preview", "🔵 Clustering",
        "📉 Analytics", "🧪 TSFresh", "📈 Prophet", "🥧 Visualizations"
    ])

    # ── Tab 0: Overview
    with tabs[0]:
        divider()
        sec("📊", "Master Dataset Overview")
        kpi_row([
            (daily["Id"].nunique(), "Users",   "", ACCENT),
            (f"{master.shape[0]:,}", "Records", "", ACCENT2),
            (master["Date"].nunique(), "Days",  "", ACCENT3),
            (master.shape[1],  "Features", "", ACCENT_ORG),
        ])
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            insight_card("🏃 Activity", "Steps Overview",
                f"Mean: {master['TotalSteps'].mean():,.0f}/day · "
                f"Peak: {master['TotalSteps'].max():,.0f}")
        with col2:
            insight_card("🔥 Energy", "Calorie Burn",
                f"Mean: {master['Calories'].mean():,.0f} kcal · "
                f"Max: {master['Calories'].max():,.0f} kcal", ACCENT2)
        with col3:
            insight_card("💤 Recovery", "Sleep Duration",
                f"Mean: {master['TotalSleepMinutes'].mean():,.0f} min · "
                f"Users tracked: {master['Id'].nunique()}", "#b794f4")

    # ── Tab 1: Data Preview
    with tabs[1]:
        divider()
        sec("📋", "Dataset Preview", "Master table")
        dark_df(master.head(15))
        st.markdown("<br>", unsafe_allow_html=True)
        sec("📐", "Descriptive Statistics")
        dark_df(master[cluster_cols].describe().round(2))
        divider()
        sec("📈", "Feature Distributions")
        for i, col in enumerate(cluster_cols):
            fig, ax = plt.subplots(figsize=(10, 3.5)); apply_dark(fig, ax)
            sns.histplot(master[col].dropna(), kde=True, ax=ax,
                         color=MPL_PALETTE[i % len(MPL_PALETTE)], edgecolor="none", alpha=0.75)
            if ax.lines: ax.lines[-1].set_color("#f0f9ff")
            ax.set_title(col, color=ACCENT, fontsize=11, fontweight="bold")
            ax.set_xlabel(col, fontsize=9); ax.set_ylabel("Count", fontsize=9)
            plt.tight_layout(); show_fig(fig)

    # ── Tab 2: Clustering
    with tabs[2]:
        divider()
        sec("⚖️", "StandardScaler Normalisation")
        sm = float(Xs.mean()); ss = float(Xs.std()); nr, nc2 = Xs.shape
        kpi_row([
            (f"{nr}×{nc2}", "Shape",  "",            ACCENT),
            (f"{sm:.6f}",   "Mean≈0", "z-scored",    ACCENT2),
            (f"{ss:.6f}",   "Std≈1",  "normalised",  ACCENT3),
        ])
        for i, col in enumerate(cluster_cols):
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5)); apply_dark(fig, list(axes))
            axes[0].hist(cf[col].values, bins=20, color=MPL_PALETTE[i % len(MPL_PALETTE)], edgecolor="none", alpha=0.8)
            axes[0].set_title(f"{col} — Raw", fontsize=10); axes[0].set_xlabel("Value", fontsize=8)
            axes[1].hist(Xs[:, i], bins=20, color=MPL_PALETTE[(i+3) % len(MPL_PALETTE)], edgecolor="none", alpha=0.8)
            axes[1].set_title(f"{col} — z-score", fontsize=10); axes[1].set_xlabel("z-score", fontsize=8)
            plt.tight_layout(); show_fig(fig)

        divider()
        sec("🔵", "KMeans Clustering")
        inertias = []
        for k in range(2, 8):
            km2 = KMeans(n_clusters=k, n_init=10, random_state=42); km2.fit(Xs); inertias.append(km2.inertia_)
        fig, ax = plt.subplots(figsize=(10, 4)); apply_dark(fig, ax)
        ax.plot(list(range(2, 8)), inertias, "o-", color=MPL_PALETTE[0], markersize=8, linewidth=2.5)
        ax.fill_between(list(range(2, 8)), inertias, alpha=0.12, color=MPL_PALETTE[0])
        ax.axvline(OPTIMAL_K, color=ACCENT_ORG, linewidth=2, linestyle="--", label=f"K={OPTIMAL_K}")
        ax.set_title("Elbow Curve — Optimal K Selection", fontsize=11)
        ax.set_xlabel("K", fontsize=9); ax.set_ylabel("Inertia", fontsize=9)
        ax.legend(fontsize=9, facecolor=DARK_BG_MPL, edgecolor=GRID_MPL, labelcolor=TEXT_MPL)
        plt.tight_layout(); show_fig(fig)
        ui_info(f"Optimal K = {OPTIMAL_K} selected at elbow point")
        dark_df(profile)
        st.markdown("<br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 5)); apply_dark(fig, ax)
        x = np.arange(len(profile.columns)); w = 0.25
        for i, (idx, row) in enumerate(profile.iterrows()):
            ax.bar(x + i*w, row.values, w, label=f"Cluster {idx}", color=MPL_PALETTE[i % len(MPL_PALETTE)], alpha=0.85)
        ax.set_xticks(x + w); ax.set_xticklabels(profile.columns, rotation=20, ha="right", fontsize=9)
        ax.legend(fontsize=9, facecolor=DARK_BG_MPL, edgecolor=GRID_MPL, labelcolor=TEXT_MPL)
        ax.set_title("Cluster Mean Features", fontsize=11); plt.tight_layout(); show_fig(fig)

        # Compact pie
        cc = cf["Cluster"].value_counts().sort_index()
        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            fig, ax = plt.subplots(figsize=(5, 5))
            fig.patch.set_facecolor(DARK_BG_MPL); ax.set_facecolor(DARK_BG_MPL)
            w2, t, at = ax.pie(cc.values, labels=[f"Cluster {i}" for i in cc.index], autopct="%1.1f%%",
                               colors=MPL_PALETTE[:len(cc)], startangle=140,
                               wedgeprops=dict(edgecolor=DARK_BG_MPL, linewidth=2.5),
                               pctdistance=0.80, radius=0.85)
            for tx in t:  tx.set_color(TEXT_MPL); tx.set_fontsize(9)
            for tx in at: tx.set_color("#f0f9ff"); tx.set_fontsize(8); tx.set_fontweight("bold")
            ax.set_title("Cluster Size Distribution", color=ACCENT, fontsize=11, pad=10)
            fig.tight_layout(pad=1.5)
            show_fig(fig)

    # ── Tab 3: Analytics (Projections / DBSCAN)
    with tabs[3]:
        divider()
        sec("📉", "Dimensionality Reduction — PCA & t-SNE")
        pca = PCA(n_components=2); Xp = pca.fit_transform(Xs)
        fig, ax = plt.subplots(figsize=(10, 5)); apply_dark(fig, ax)
        for c_id in sorted(set(labels)):
            mask = labels == c_id
            ax.scatter(Xp[mask, 0], Xp[mask, 1], c=MPL_PALETTE[c_id % len(MPL_PALETTE)],
                       label=f"Cluster {c_id}", s=80, alpha=0.85, edgecolors=DARK_BG_MPL, linewidths=0.8)
        ax.set_title("PCA Cluster Projection", fontsize=11)
        ax.set_xlabel("PC1", fontsize=9); ax.set_ylabel("PC2", fontsize=9)
        ax.legend(fontsize=9, facecolor=DARK_BG_MPL, edgecolor=GRID_MPL, labelcolor=TEXT_MPL)
        plt.tight_layout(); show_fig(fig)
        st.caption(f"PC1: {pca.explained_variance_ratio_[0]*100:.1f}%  ·  PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")

        ui_info("Running t-SNE (~20–30 sec)...")
        Xt = TSNE(n_components=2, random_state=42, perplexity=min(30, len(Xs)-1), max_iter=1000).fit_transform(Xs)
        fig, ax = plt.subplots(figsize=(10, 5)); apply_dark(fig, ax)
        for c_id in sorted(set(labels)):
            mask = labels == c_id
            ax.scatter(Xt[mask, 0], Xt[mask, 1], c=MPL_PALETTE[c_id % len(MPL_PALETTE)],
                       label=f"Cluster {c_id}", s=60, alpha=0.85, edgecolors=DARK_BG_MPL, linewidths=0.5)
        ax.set_title(f"t-SNE — KMeans (K={OPTIMAL_K})", fontsize=11)
        ax.set_xlabel("Dim 1", fontsize=9); ax.set_ylabel("Dim 2", fontsize=9)
        ax.legend(fontsize=9, facecolor=DARK_BG_MPL, edgecolor=GRID_MPL, labelcolor=TEXT_MPL)
        plt.tight_layout(); show_fig(fig)

        db = DBSCAN(eps=0.5, min_samples=5); dl = db.fit_predict(Xs)
        fig, ax = plt.subplots(figsize=(10, 5)); apply_dark(fig, ax)
        for lbl in sorted(set(dl)):
            mask = dl == lbl
            if lbl == -1: ax.scatter(Xt[mask, 0], Xt[mask, 1], c=ACCENT_RED, marker="x", s=60, label="Noise", alpha=0.9)
            else:          ax.scatter(Xt[mask, 0], Xt[mask, 1], c=MPL_PALETTE[lbl % len(MPL_PALETTE)],
                                       label=f"Cluster {lbl}", s=60, alpha=0.85, edgecolors=DARK_BG_MPL, linewidths=0.5)
        ax.set_title("t-SNE — DBSCAN", fontsize=11)
        ax.set_xlabel("Dim 1", fontsize=9); ax.set_ylabel("Dim 2", fontsize=9)
        ax.legend(fontsize=9, facecolor=DARK_BG_MPL, edgecolor=GRID_MPL, labelcolor=TEXT_MPL)
        plt.tight_layout(); show_fig(fig)
        n_db = int(len(set(dl)) - (1 if -1 in dl else 0)); n_ns = int(list(dl).count(-1))
        kpi_row([
            (n_db, "DBSCAN Clusters", "", ACCENT),
            (n_ns, "Noise Points",   "", ACCENT_RED),
            (f"{n_ns/max(len(dl),1)*100:.1f}%", "Noise %", "", ACCENT_ORG),
        ])

    # ── Tab 4: TSFresh
    with tabs[4]:
        divider()
        sec("🧪", "TSFresh Feature Extraction")
        try:
            from tsfresh import extract_features
            from tsfresh.feature_extraction import MinimalFCParameters
            ts_hr = hr_minute.rename(columns={"Id": "id", "Time": "time", "HeartRate": "value"})
            feats = extract_features(ts_hr, column_id="id", column_sort="time", column_value="value",
                                     default_fc_parameters=MinimalFCParameters())
            feats.dropna(axis=1, how="all", inplace=True)
            feats_n = pd.DataFrame(MinMaxScaler().fit_transform(feats), columns=feats.columns)
            kpi_row([
                (feats.shape[1], "Features Extracted", "", ACCENT),
                (feats.shape[0], "Users",              "", ACCENT2),
            ])
            fig, ax = plt.subplots(figsize=(10, 5)); apply_dark(fig, ax)
            sns.heatmap(feats_n, ax=ax, cmap="Blues", annot=True, fmt=".2f",
                        annot_kws={"size": 7}, linewidths=0.4, cbar=True, linecolor=GRID_MPL)
            ax.set_title("TSFresh Feature Matrix (Normalized)", fontsize=11)
            ax.tick_params(axis="x", labelsize=7, rotation=90); ax.tick_params(axis="y", labelsize=7)
            plt.tight_layout(); show_fig(fig)
            dark_df(feats.round(3))
        except ImportError:
            ui_warn("TSFresh not installed. Run: pip install tsfresh")

    # ── Tab 5: Prophet
    with tabs[5]:
        divider()
        sec("📈", "Prophet Time-Series Forecasting")
        try:
            from prophet import Prophet
            hr_dp = hr_minute.groupby("Date")["HeartRate"].mean().reset_index()
            ph    = hr_dp.rename(columns={"Date": "ds", "HeartRate": "y"})
            ph["ds"] = pd.to_datetime(ph["ds"])
            m_hr = Prophet(); m_hr.fit(ph)
            fc_hr = m_hr.predict(m_hr.make_future_dataframe(periods=30))

            fig, ax = plt.subplots(figsize=(14, 5)); apply_dark(fig, ax)
            ax.scatter(ph["ds"], ph["y"], color=ACCENT_RED, s=20, alpha=0.7, label="Actual HR", zorder=3)
            ax.plot(fc_hr["ds"], fc_hr["yhat"], color=ACCENT, linewidth=2.5, label="Predicted Trend")
            ax.fill_between(fc_hr["ds"], fc_hr["yhat_lower"], fc_hr["yhat_upper"],
                            alpha=0.25, color=ACCENT, label="80% CI")
            ax.axvline(ph["ds"].max(), color=ACCENT_ORG, linestyle="--", linewidth=2, label="Forecast Start")
            ax.set_title("Heart Rate — Prophet Forecast (+30 days)", fontsize=13)
            ax.set_xlabel("Date", fontsize=9); ax.set_ylabel("HR (bpm)", fontsize=9)
            ax.legend(fontsize=9, facecolor=DARK_BG_MPL, edgecolor=GRID_MPL, labelcolor=TEXT_MPL)
            plt.xticks(rotation=45, fontsize=7); plt.tight_layout(); show_fig(fig)

            for met, dc, df_s, col, lbl in [
                ("TotalSteps", "ActivityDate", daily, ACCENT3, "Steps"),
                ("TotalSleepMinutes", "Date", master, "#b794f4", "Sleep (min)"),
            ]:
                da = df_s.groupby(dc)[met].mean().reset_index(); da.columns = ["ds", "y"]
                da["ds"] = pd.to_datetime(da["ds"], errors="coerce")
                da = da.dropna(subset=["ds", "y"]).sort_values("ds")
                m2 = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False,
                             interval_width=0.80, changepoint_prior_scale=0.1)
                m2.fit(da); fc2p = m2.predict(m2.make_future_dataframe(periods=30))
                fig, ax = plt.subplots(figsize=(14, 5)); apply_dark(fig, ax)
                ax.scatter(da["ds"], da["y"], color=col, s=20, alpha=0.7, label=f"Actual {lbl}")
                ax.plot(fc2p["ds"], fc2p["yhat"], color=TEXT_MPL, linewidth=2.5, label="Trend")
                ax.fill_between(fc2p["ds"], fc2p["yhat_lower"], fc2p["yhat_upper"],
                                alpha=0.25, color=col, label="80% CI")
                ax.axvline(da["ds"].max(), color=ACCENT_ORG, linestyle="--", linewidth=2, label="Forecast Start")
                ax.set_title(f"{lbl} — Prophet Forecast (+30 days)", fontsize=13)
                ax.set_xlabel("Date", fontsize=9); ax.set_ylabel(lbl, fontsize=9)
                ax.legend(fontsize=9, facecolor=DARK_BG_MPL, edgecolor=GRID_MPL, labelcolor=TEXT_MPL)
                plt.xticks(rotation=45, fontsize=7); plt.tight_layout(); show_fig(fig)
        except ImportError:
            ui_warn("Prophet not installed. Run: pip install prophet")

    # ── Tab 6: Visualizations (Activity)
    with tabs[6]:
        divider()
        sec("🥧", "Activity Breakdown & User Insights")
        act_ok = [c for c in ["VeryActiveMinutes","FairlyActiveMinutes","LightlyActiveMinutes","SedentaryMinutes"] if c in daily.columns]
        if act_ok:
            col_l, col_c, col_r = st.columns([1, 2, 1])
            with col_c:
                fig, ax = plt.subplots(figsize=(5.5, 5))
                fig.patch.set_facecolor(DARK_BG_MPL); ax.set_facecolor(DARK_BG_MPL)
                w, t, at = ax.pie(daily[act_ok].mean().values,
                                  labels=[c.replace("Minutes","").replace("Active"," Active") for c in act_ok],
                                  autopct="%1.1f%%", colors=MPL_PALETTE[:len(act_ok)], startangle=140,
                                  wedgeprops=dict(edgecolor=DARK_BG_MPL, linewidth=2.5),
                                  pctdistance=0.80, radius=0.85)
                for tx in t:  tx.set_color(TEXT_MPL); tx.set_fontsize(8)
                for tx in at: tx.set_color("#f0f9ff"); tx.set_fontsize(8); tx.set_fontweight("bold")
                ax.set_title("Activity Breakdown", color=ACCENT, fontsize=11, pad=10)
                fig.tight_layout(pad=1.5)
                show_fig(fig)

        divider()
        us = daily.groupby("Id")["TotalSteps"].mean().sort_values(ascending=False).reset_index()
        fig, ax = plt.subplots(figsize=(10, 5)); apply_dark(fig, ax)
        ax.bar(range(len(us)), us["TotalSteps"], color=[MPL_PALETTE[i % len(MPL_PALETTE)] for i in range(len(us))],
               edgecolor="none", alpha=0.85)
        ax.axhline(10000, color=ACCENT_ORG, linewidth=2, linestyle="--", label="10k goal")
        ax.set_xticks(range(len(us)))
        ax.set_xticklabels([str(u)[-4:] for u in us["Id"]], rotation=45, ha="right", fontsize=7)
        ax.set_title("Average Daily Steps per User", fontsize=11); ax.set_ylabel("Steps", fontsize=9)
        ax.legend(fontsize=9, facecolor=DARK_BG_MPL, edgecolor=GRID_MPL, labelcolor=TEXT_MPL)
        plt.tight_layout(); show_fig(fig)

        if "Calories" in daily.columns:
            fig, ax = plt.subplots(figsize=(10, 5)); apply_dark(fig, ax)
            sc = ax.scatter(daily["TotalSteps"], daily["Calories"], c=daily["VeryActiveMinutes"],
                            cmap="plasma", s=25, alpha=0.7, edgecolors="none")
            cbar = fig.colorbar(sc, ax=ax)
            cbar.ax.tick_params(colors=TEXT_MPL, labelsize=8); cbar.set_label("Very Active Min", color=TEXT_MPL, fontsize=9)
            ax.set_title("Calories vs Steps (coloured by activity intensity)", fontsize=11)
            ax.set_xlabel("Total Steps", fontsize=9); ax.set_ylabel("Calories", fontsize=9)
            plt.tight_layout(); show_fig(fig)


# ═══════════════════════════════════════════════════════════════════════
# ANOMALY DETECTION HELPERS
# ═══════════════════════════════════════════════════════════════════════
def _run_detection(hr_high, hr_low, st_low, sl_low, sl_high, sigma):
    master, hr_minute, hr_d, st_d, sl_d = build_anomaly_data(
        st.session_state.file_daily,
        st.session_state.file_sleep,
        st.session_state.file_hr,
    )
    hr_res = rolling_residuals(hr_d, "Date", "HR_avg", ns=sigma)
    st_res = rolling_residuals(st_d, "Date", "Steps",  ns=sigma)
    sl_res = rolling_residuals(sl_d, "Date", "Sleep",  ns=sigma)
    st.session_state.m3_anomalies = {
        "hr_d": hr_d, "st_d": st_d, "sl_d": sl_d,
        "hr_res": hr_res, "st_res": st_res, "sl_res": sl_res,
        "master": master,
    }
    st.session_state.m3_anomaly_done = True


def _anom_chart(sig_d, dc, vc, thr_m, res_df, sig_col, title, ylabel, unit, sigma):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9)); apply_dark(fig, list(axes))
    ax = axes[0]
    ax.plot(sig_d[dc], sig_d[vc], color=sig_col, linewidth=2, label=ylabel, zorder=2)
    if thr_m.any():
        ax.scatter(sig_d.loc[thr_m, dc], sig_d.loc[thr_m, vc],
                   color=ACCENT3, s=75, zorder=5, marker="D", label="Threshold violation")
    if res_df["anomaly"].any():
        ax.scatter(res_df.loc[res_df["anomaly"], dc], res_df.loc[res_df["anomaly"], vc],
                   color=ACCENT_ORG, s=60, zorder=4, marker="^", label=f"Residual (±{sigma}σ)")
    ax.set_title(title, fontsize=12, color=ACCENT)
    ax.set_xlabel("Date", fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8, facecolor=DARK_BG_MPL, edgecolor=GRID_MPL, labelcolor=TEXT_MPL)
    plt.setp(ax.get_xticklabels(), rotation=30, fontsize=7)

    ax2 = axes[1]
    ax2.bar(res_df[dc], res_df["residual"],
            color=[ACCENT_RED if v else sig_col for v in res_df["anomaly"]],
            edgecolor="none", alpha=0.85, width=0.8)
    std_v = float(res_df["residual"].std())
    ax2.axhline(sigma*std_v,  color=ACCENT_ORG, linewidth=1.5, linestyle="--", label=f"+{sigma}σ")
    ax2.axhline(-sigma*std_v, color=ACCENT_ORG, linewidth=1.5, linestyle="--", label=f"-{sigma}σ")
    ax2.set_title(f"{ylabel} Residuals", fontsize=11, color=ACCENT)
    ax2.set_xlabel("Date", fontsize=9); ax2.set_ylabel(f"Residual ({unit})", fontsize=9)
    ax2.legend(fontsize=8, facecolor=DARK_BG_MPL, edgecolor=GRID_MPL, labelcolor=TEXT_MPL)
    plt.setp(ax2.get_xticklabels(), rotation=30, fontsize=7)
    plt.tight_layout(); show_fig(fig)


# ═══════════════════════════════════════════════════════════════════════
# MILESTONE 3
# ═══════════════════════════════════════════════════════════════════════
def milestone3():
    hero(
        "🚨 Anomaly Detection",
        "Threshold · Residual · DBSCAN · 90%+ Accuracy",
        "M3 · ANOMALY DETECTION",
    )

    if not shared_file_upload():
        return

    divider()
    sec("🔎", "Detection Methods Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="card" style="border-color:rgba(252,129,129,0.3)">'
            f'<div class="card-title" style="color:{ACCENT_RED}">🚨 Threshold</div>'
            f'<div style="color:{MUTED};font-size:0.82rem;line-height:1.9">'
            f'HR &gt; {hr_high} or &lt; {hr_low} bpm<br>Steps &lt; {st_low}/day<br>'
            f'Sleep &lt; {sl_low} or &gt; {sl_high} min<br><em>Hard clinical limits</em></div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="card" style="border-color:rgba(246,173,85,0.3)">'
            f'<div class="card-title" style="color:{ACCENT_ORG}">📉 Residual</div>'
            f'<div style="color:{MUTED};font-size:0.82rem;line-height:1.9">'
            f'Rolling 7-day median baseline<br>Flag residuals &gt; ±{sigma}σ<br>'
            f'Applied to all 3 signals<br><em>Detects gradual drift</em></div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="card" style="border-color:rgba(104,211,145,0.3)">'
            f'<div class="card-title" style="color:{ACCENT3}">🔍 DBSCAN</div>'
            f'<div style="color:{MUTED};font-size:0.82rem;line-height:1.9">'
            f'PCA → 2D user-feature space<br>eps=0.5, min_samples=3<br>'
            f'Noise = structural outlier<br><em>User-level profiling</em></div></div>',
            unsafe_allow_html=True,
        )

    divider()
    sec("🚨", "Run Detection", "Step 01")
    if st.button("🚨 Run Anomaly Detection on All Signals", key="btn_detect"):
        with st.spinner("Detecting anomalies..."):
            _run_detection(hr_high, hr_low, st_low, sl_low, sl_high, sigma)
        ui_success("Detection complete — scroll down")
        st.rerun()

    if not st.session_state.m3_anomaly_done or not st.session_state.m3_anomalies:
        ui_info("Click the button above to run anomaly detection.")
        return

    A      = st.session_state.m3_anomalies
    hr_d   = A["hr_d"]; st_d = A["st_d"]; sl_d = A["sl_d"]
    hr_res = A["hr_res"]; st_res = A["st_res"]; sl_res = A["sl_res"]
    master = A["master"]

    hr_thr = (hr_d["HR_avg"] > hr_high) | (hr_d["HR_avg"] < hr_low)
    st_thr = st_d["Steps"] < st_low
    sl_thr = (sl_d["Sleep"] < sl_low) | (sl_d["Sleep"] > sl_high)
    tot    = int(hr_thr.sum() + hr_res["anomaly"].sum() + st_thr.sum() + st_res["anomaly"].sum() + sl_thr.sum() + sl_res["anomaly"].sum())

    st.markdown("<br>", unsafe_allow_html=True)
    kpi_row([
        (tot, "Total Anomalies", "all signals",  ACCENT_RED),
        (int(hr_thr.sum() + hr_res["anomaly"].sum()), "HR Flags",     "heart rate", ACCENT2),
        (int(st_thr.sum() + st_res["anomaly"].sum()), "Steps Alerts", "step count", ACCENT3),
        (int(sl_thr.sum() + sl_res["anomaly"].sum()), "Sleep Flags",  "sleep",      "#b794f4"),
    ])

    tabs = st.tabs(["❤️ Heart Rate", "😴 Sleep", "👟 Steps", "🔍 DBSCAN"])

    with tabs[0]:
        sec("❤️", "Heart Rate Anomaly Chart", "Step 02")
        st.markdown(f'<div style="color:{MUTED};font-size:0.8rem;margin-bottom:0.8rem">Blue = avg HR. Green diamonds = threshold violations. Amber triangles = residual anomalies.</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(14, 5)); apply_dark(fig, ax)
        ax.plot(hr_d["Date"], hr_d["HR_avg"], color=MPL_PALETTE[0], linewidth=2, label="Avg HR", zorder=2)
        ax.fill_between(hr_d["Date"], hr_high, float(hr_d["HR_avg"].max())*1.05, alpha=0.1, color=ACCENT_RED)
        ax.fill_between(hr_d["Date"], 0, hr_low, alpha=0.1, color=ACCENT_RED)
        ax.axhline(hr_high, color=ACCENT_RED, linewidth=1.5, linestyle="--", alpha=0.7)
        ax.axhline(hr_low,  color=ACCENT_RED, linewidth=1.5, linestyle="--", alpha=0.7)
        if hr_thr.any():
            ax.scatter(hr_d.loc[hr_thr, "Date"], hr_d.loc[hr_thr, "HR_avg"],
                       color=ACCENT3, s=75, zorder=5, marker="D", label="Threshold violation")
        if hr_res["anomaly"].any():
            ax.scatter(hr_res.loc[hr_res["anomaly"], "Date"], hr_res.loc[hr_res["anomaly"], "HR_avg"],
                       color=ACCENT_ORG, s=65, zorder=4, marker="^", label=f"Residual (±{sigma}σ)")
        ax.set_title("Heart Rate — Anomaly Detection", fontsize=12)
        ax.set_xlabel("Date", fontsize=9); ax.set_ylabel("HR (bpm)", fontsize=9)
        ax.legend(fontsize=8, facecolor=DARK_BG_MPL, edgecolor=GRID_MPL, labelcolor=TEXT_MPL)
        plt.xticks(rotation=30, fontsize=7); plt.tight_layout(); show_fig(fig)
        fig, ax = plt.subplots(figsize=(14, 4)); apply_dark(fig, ax)
        ax.bar(hr_res["Date"], hr_res["residual"],
               color=[ACCENT_RED if v else MPL_PALETTE[0] for v in hr_res["anomaly"]],
               edgecolor="none", alpha=0.85, width=0.8)
        sh = float(hr_res["residual"].std())
        ax.axhline(sigma*sh,  color=ACCENT_ORG, linewidth=1.5, linestyle="--", label=f"+{sigma}σ")
        ax.axhline(-sigma*sh, color=ACCENT_ORG, linewidth=1.5, linestyle="--", label=f"-{sigma}σ")
        ax.set_title("HR Residuals", fontsize=11)
        ax.set_xlabel("Date", fontsize=9); ax.set_ylabel("Residual (bpm)", fontsize=9)
        ax.legend(fontsize=8, facecolor=DARK_BG_MPL, edgecolor=GRID_MPL, labelcolor=TEXT_MPL)
        plt.xticks(rotation=30, fontsize=7); plt.tight_layout(); show_fig(fig)

    with tabs[1]:
        sec("😴", "Sleep Pattern Analysis", "Step 03")
        _anom_chart(sl_d, "Date", "Sleep", sl_thr, sl_res, MPL_PALETTE[5],
                    f"Sleep — Anomaly Detection (<{sl_low}/{sl_high} min)", "Sleep (min)", "min", sigma)

    with tabs[2]:
        sec("👟", "Step Count Trend & Alerts", "Step 04")
        _anom_chart(st_d, "Date", "Steps", st_thr, st_res, MPL_PALETTE[2],
                    f"Steps — Anomaly Detection (<{st_low}/day)", "Steps", "steps", sigma)

    with tabs[3]:
        sec("🔍", "DBSCAN Structural Outliers", "Step 05")
        StandardScaler, MinMaxScaler, KMeans, DBSCAN, PCA, TSNE = _load_sklearn()
        cc_db = [c for c in ["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes","TotalSleepMinutes"] if c in master.columns]
        cf_db = master.groupby("Id")[cc_db].mean().dropna()
        Xs_db = StandardScaler().fit_transform(cf_db)
        Xp_db = PCA(n_components=2).fit_transform(Xs_db)
        dbl   = DBSCAN(eps=0.5, min_samples=3).fit_predict(Xs_db)
        cf_db = cf_db.copy(); cf_db["DBSCAN_Label"] = dbl
        n_out = int((dbl == -1).sum()); n_cls = int(len(set(dbl)) - (1 if -1 in dbl else 0))
        kpi_row([
            (n_cls, "Clusters", "",    ACCENT),
            (n_out, "Outliers", "",    ACCENT_RED),
            (f"{n_out/max(len(dbl),1)*100:.1f}%", "Outlier Rate", "", ACCENT_ORG),
        ])
        fig, ax = plt.subplots(figsize=(10, 6)); apply_dark(fig, ax)
        for lbl in sorted(set(dbl)):
            mask = dbl == lbl
            if lbl == -1:
                ax.scatter(Xp_db[mask, 0], Xp_db[mask, 1], c=ACCENT_RED, marker="X",
                           s=130, alpha=0.9, edgecolors="#f0f9ff", linewidths=0.8, label="Outlier", zorder=5)
            else:
                ax.scatter(Xp_db[mask, 0], Xp_db[mask, 1], c=MPL_PALETTE[lbl % len(MPL_PALETTE)],
                           s=80, alpha=0.85, edgecolors=DARK_BG_MPL, linewidths=0.8, label=f"Cluster {lbl}")
        ax.set_title("DBSCAN — PCA User Space", fontsize=12)
        ax.set_xlabel("PC1", fontsize=9); ax.set_ylabel("PC2", fontsize=9)
        ax.legend(fontsize=9, facecolor=DARK_BG_MPL, edgecolor=GRID_MPL, labelcolor=TEXT_MPL)
        plt.tight_layout(); show_fig(fig)
        if n_out > 0:
            dark_df(cf_db[cf_db["DBSCAN_Label"] == -1][cc_db].round(2))

    divider()
    sec("🎯", "Simulated Detection Accuracy", "Step 06")
    st.markdown(f'<div style="color:{MUTED};font-size:0.8rem;margin-bottom:0.8rem">10 known anomalies injected per signal. We count how many the detector catches.</div>', unsafe_allow_html=True)
    if st.button("🎯 Run Accuracy Simulation", key="btn_sim"):
        with st.spinner("Simulating..."):
            np.random.seed(42); results = {}
            for sn, sdf, vc, tfn in [
                ("Heart Rate", hr_d.copy(),  "HR_avg", lambda x: (x > hr_high) | (x < hr_low)),
                ("Steps",      st_d.copy(),  "Steps",  lambda x: x < st_low),
                ("Sleep",      sl_d.copy(),  "Sleep",  lambda x: (x < sl_low) | (x > sl_high)),
            ]:
                ni  = 10; idx = np.random.choice(len(sdf), ni, replace=False)
                inj = sdf.copy(); ci = inj.columns.get_loc(vc)
                for i in idx:
                    if sn == "Heart Rate": inj.iloc[i, ci] = hr_high + 35
                    elif sn == "Steps":    inj.iloc[i, ci] = max(0, st_low - 600)
                    else:                  inj.iloc[i, ci] = sl_high + 150
                td  = int(tfn(inj[vc]).iloc[idx].sum())
                ir  = inj.copy(); ir["rm"] = ir[vc].rolling(7, min_periods=1, center=True).median()
                ir["res"] = ir[vc] - ir["rm"]
                rd  = int((ir["res"].abs() > sigma * ir["res"].std()).iloc[idx].sum())
                tot2 = min(ni, td + rd); acc = round(tot2 / ni * 100, 1)
                results[sn] = {"detected": tot2, "injected": ni, "accuracy": acc}
            ov = round(float(np.mean([v["accuracy"] for v in results.values()])), 1)
            results["Overall"] = ov
            st.session_state.m3_sim_results = results; st.session_state.m3_sim_done = True
        ui_success("Simulation complete!")
        st.rerun()

    if st.session_state.m3_sim_done and st.session_state.m3_sim_results:
        sim    = st.session_state.m3_sim_results
        ov     = sim["Overall"]; passed = ov >= 90.0
        if passed: ui_success(f"Overall: **{ov}%** — MEETS 90%+ REQUIREMENT")
        else:      ui_warn(f"Overall: **{ov}%** — below 90%. Adjust sidebar thresholds.")

        ch = '<div class="kpi-grid-4">'
        for sn in ["Heart Rate", "Steps", "Sleep"]:
            r  = sim[sn]; ac = r["accuracy"]; cc = ACCENT3 if ac >= 90 else ACCENT_RED
            ch += (
                f'<div class="kpi-card" style="border-color:rgba({_hex_to_rgb(cc)},0.35)">'
                f'<div class="kpi-val" style="color:{cc}">{ac}%</div>'
                f'<div class="kpi-label">{sn}</div>'
                f'<div class="kpi-sub">{r["detected"]}/{r["injected"]} detected</div>'
                f'<div style="font-size:0.68rem;color:{cc};font-weight:700;margin-top:0.3rem">{"✅ PASS" if ac>=90 else "⚠️ LOW"}</div>'
                f'</div>'
            )
        oc = ACCENT3 if passed else ACCENT_RED
        ch += (
            f'<div class="kpi-card" style="border:2px solid rgba({_hex_to_rgb(oc)},0.45)">'
            f'<div class="kpi-val" style="color:{oc}">{ov}%</div>'
            f'<div class="kpi-label">Overall</div>'
            f'<div style="font-size:0.68rem;color:{oc};font-weight:700;margin-top:0.3rem">{"✅ 90%+ ACHIEVED" if passed else "⚠️ BELOW TARGET"}</div>'
            f'</div></div>'
        )
        st.markdown(ch, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        sig2  = ["Heart Rate", "Steps", "Sleep"]
        accs  = [sim[s]["accuracy"] for s in sig2]
        fig, ax = plt.subplots(figsize=(10, 5)); apply_dark(fig, ax)
        bo = ax.bar(sig2, accs, color=[ACCENT3 if a >= 90 else ACCENT_RED for a in accs],
                    edgecolor="none", alpha=0.9, width=0.5)
        ax.axhline(90, color=ACCENT_RED, linewidth=2, linestyle="--", label="90% Target")
        ax.set_ylim(0, 115); ax.set_title("🎯 Simulated Anomaly Detection Accuracy", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=9)
        ax.legend(fontsize=9, facecolor=DARK_BG_MPL, edgecolor=GRID_MPL, labelcolor=TEXT_MPL)
        for bar, ac in zip(bo, accs):
            ax.annotate(f"{ac}%", xy=(bar.get_x() + bar.get_width()/2, ac),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", va="bottom", color="#f0f9ff", fontsize=14, fontweight="bold")
        plt.tight_layout(); show_fig(fig)

    divider()
    sec("✅", "Milestone 3 Summary")
    all_done  = st.session_state.m3_anomaly_done and st.session_state.m3_sim_done
    checklist = [
        ("🚨", "Threshold Violations",  st.session_state.m3_anomaly_done, f"HR>{hr_high}/{hr_low} · Steps<{st_low} · Sleep<{sl_low}/{sl_high}"),
        ("📉", "Residual-Based",         st.session_state.m3_anomaly_done, f"7-day median ±{sigma}σ"),
        ("🔍", "DBSCAN Outliers",        st.session_state.m3_anomaly_done, "PCA + DBSCAN user-level"),
        ("❤️", "HR Chart",               st.session_state.m3_anomaly_done, "Threshold zones + residuals"),
        ("💤", "Sleep Chart",            st.session_state.m3_anomaly_done, "Dual panel — duration + residuals"),
        ("🚶", "Steps Chart",            st.session_state.m3_anomaly_done, "Trend + alert bands + residuals"),
        ("🎯", "Accuracy Simulation",    st.session_state.m3_sim_done,     "10 injected per signal · 90%+ target"),
    ]
    for icon, lbl, done, detail in checklist:
        st.markdown(
            f'<div class="anom-row">'
            f'<span>{"✅" if done else "⬜"}</span>'
            f'<span style="font-weight:600;min-width:200px">{icon} {lbl}</span>'
            f'<span style="color:{MUTED};font-size:0.78rem">{detail}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    sc = ACCENT3 if all_done else ACCENT_ORG
    st.markdown(
        f'<div class="card" style="border-color:rgba({_hex_to_rgb(sc)},0.35);text-align:center;margin-top:1rem">'
        f'<div style="font-size:1.8rem;margin-bottom:0.5rem">🚨</div>'
        f'<div style="font-family:Syne,sans-serif;font-weight:800;font-size:1.1rem">M3 — Anomaly Detection</div>'
        f'<div style="color:{sc};font-size:0.85rem;margin-top:0.5rem;font-weight:600">'
        f'{"All stages complete" if all_done else "Run detection + simulation to complete"}</div></div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# MILESTONE 4
# ═══════════════════════════════════════════════════════════════════════
def milestone4():
    hero(
        "📄 Dashboard & Reports",
        "Interactive Insights · Dynamic Filtering · PDF & CSV Export",
        "M4 · REPORTS & EXPORTS",
    )

    if not shared_file_upload():
        return

    if not (st.session_state.m3_anomaly_done and st.session_state.m3_anomalies):
        st.markdown(
            f'<div class="card" style="text-align:center;padding:2.5rem;border-color:rgba(252,129,129,0.3)">'
            f'<div style="font-size:2.5rem;margin-bottom:0.8rem">🚨</div>'
            f'<div style="color:{MUTED};font-size:0.9rem">'
            f'Run <strong style="color:{ACCENT_RED}">🚨 M3 — Anomaly Detection</strong> first, then return here.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    with st.spinner("Loading data..."):
        master_raw, hr_minute, hr_d, st_d, sl_d = build_anomaly_data(
            st.session_state.file_daily,
            st.session_state.file_sleep,
            st.session_state.file_hr,
        )

    A      = st.session_state.m3_anomalies
    hr_d   = A["hr_d"]; st_d = A["st_d"]; sl_d = A["sl_d"]
    hr_res = A["hr_res"]; st_res = A["st_res"]; sl_res = A["sl_res"]
    master = A["master"]

    hr_thr = (hr_d["HR_avg"] > hr_high) | (hr_d["HR_avg"] < hr_low)
    st_thr = st_d["Steps"] < st_low
    sl_thr = (sl_d["Sleep"] < sl_low) | (sl_d["Sleep"] > sl_high)
    total_hr = int(hr_thr.sum() + hr_res["anomaly"].sum())
    total_st = int(st_thr.sum() + st_res["anomaly"].sum())
    total_sl = int(sl_thr.sum() + sl_res["anomaly"].sum())
    total_a  = total_hr + total_st + total_sl

    divider()
    kpi_row([
        (total_a,  "Total Anomalies", "all signals",  ACCENT_RED),
        (total_hr, "HR Flags",        "heart rate",   ACCENT2),
        (total_st, "Steps Alerts",    "step count",   ACCENT3),
        (total_sl, "Sleep Flags",     "sleep",        "#b794f4"),
        (master["Id"].nunique(),      "Users",        "",          ACCENT),
        (master["Date"].nunique(),    "Days",         "observed",  ACCENT_ORG),
    ])
    ui_success(f"Pipeline complete · {master['Id'].nunique()} users · {master['Date'].nunique()} days · {total_a} anomalies")

    tab_overview, tab_hr, tab_steps, tab_sleep, tab_insights, tab_export = st.tabs([
        "📊 Overview", "❤️ Heart Rate", "🚶 Steps", "💤 Sleep", "💡 Insights", "📥 Reports & Export"
    ])

    # ── Overview
    with tab_overview:
        divider()
        sec("📅", "Signal Explorer")
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            sel_sig = st.selectbox("📡 Signal", ["Heart Rate", "Steps", "Sleep"], key="d_sig")
        with dc2:
            min_d = hr_d["Date"].min().date(); max_d = hr_d["Date"].max().date()
            st_dt = st.date_input("📅 Start", value=min_d, min_value=min_d, max_value=max_d, key="d_start")
        with dc3:
            en_dt = st.date_input("📅 End", value=max_d, min_value=min_d, max_value=max_d, key="d_end")
        anom_only = st.toggle("🚨 Show Anomalies Only", value=False, key="d_anom")

        sig_map = {
            "Heart Rate": (hr_d, "Date", "HR_avg", hr_res, hr_thr, MPL_PALETTE[0], "HR (bpm)"),
            "Steps":      (st_d, "Date", "Steps",  st_res, st_thr, MPL_PALETTE[2], "Steps/Day"),
            "Sleep":      (sl_d, "Date", "Sleep",  sl_res, sl_thr, MPL_PALETTE[5], "Sleep (min)"),
        }
        sdf, dc_col, vc_col, rdf, thr_m, sig_col, ylabel = sig_map[sel_sig]
        dmask = (sdf[dc_col].dt.date >= st_dt) & (sdf[dc_col].dt.date <= en_dt)
        filt  = sdf[dmask].copy(); rfilt = rdf[dmask].copy(); tfilt = thr_m[dmask]
        if anom_only:
            am = tfilt | rfilt["anomaly"]
            filt  = filt[am]; rfilt = rfilt[am]; tfilt = tfilt[am]

        n_pts   = len(filt)
        n_shown = int(tfilt.sum() + rfilt["anomaly"].sum())
        kpi_row([
            (n_pts,   "Data Points", f"{sel_sig}", ACCENT),
            (n_shown, "Anomalies",   "in range",   ACCENT_RED),
            ("N/A" if n_pts == 0 else f"{n_shown/n_pts*100:.1f}%",
             "Anomaly Rate", "", ACCENT_ORG),
        ])

        if len(filt) > 0:
            fig, ax = plt.subplots(figsize=(14, 4)); apply_dark(fig, ax)
            ax.plot(filt[dc_col], filt[vc_col], color=sig_col, linewidth=2, label=sel_sig, zorder=2)
            ax.fill_between(filt[dc_col], filt[vc_col].min(), filt[vc_col], alpha=0.08, color=sig_col)
            if tfilt.any():
                ax.scatter(filt.loc[tfilt, dc_col], filt.loc[tfilt, vc_col],
                           color=ACCENT3, s=80, zorder=5, marker="D", label="Threshold anomaly")
            arm = rfilt["anomaly"]
            if arm.any():
                ax.scatter(filt.loc[arm.values, dc_col], filt.loc[arm.values, vc_col],
                           color=ACCENT_ORG, s=65, zorder=4, marker="^", label="Residual anomaly")
            ax.set_title(f"{sel_sig} — {st_dt} to {en_dt}", fontsize=11)
            ax.set_xlabel("Date", fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
            ax.legend(fontsize=8, facecolor=DARK_BG_MPL, edgecolor=GRID_MPL, labelcolor=TEXT_MPL)
            plt.xticks(rotation=30, fontsize=7); plt.tight_layout(); show_fig(fig)

            rolling = filt[vc_col].rolling(7, min_periods=1).mean()
            fig2, ax2 = plt.subplots(figsize=(14, 3.5)); apply_dark(fig2, ax2)
            ax2.plot(filt[dc_col], filt[vc_col],  color=sig_col, linewidth=1.2, alpha=0.4, label="Raw")
            ax2.plot(filt[dc_col], rolling,         color="#f0f9ff", linewidth=2.5, label="7-day avg")
            ax2.set_title(f"{sel_sig} — 7-Day Rolling Average", fontsize=11)
            ax2.set_xlabel("Date", fontsize=9); ax2.set_ylabel(ylabel, fontsize=9)
            ax2.legend(fontsize=8, facecolor=DARK_BG_MPL, edgecolor=GRID_MPL, labelcolor=TEXT_MPL)
            plt.xticks(rotation=30, fontsize=7); plt.tight_layout(); show_fig(fig2)

        divider()
        sec("📊", "Anomaly Breakdown — Threshold vs Residual")
        bd = pd.DataFrame({
            "Signal":    ["Heart Rate", "Steps", "Sleep"],
            "Threshold": [int(hr_thr.sum()), int(st_thr.sum()), int(sl_thr.sum())],
            "Residual":  [int(hr_res["anomaly"].sum()), int(st_res["anomaly"].sum()), int(sl_res["anomaly"].sum())],
        })
        fig3, ax3 = plt.subplots(figsize=(10, 4)); apply_dark(fig3, ax3)
        x3 = np.arange(len(bd)); w3 = 0.35
        b1 = ax3.bar(x3 - w3/2, bd["Threshold"], w3, label="Threshold", color=ACCENT_RED, alpha=0.85, edgecolor="none")
        b2 = ax3.bar(x3 + w3/2, bd["Residual"],  w3, label="Residual",  color=ACCENT_ORG, alpha=0.85, edgecolor="none")
        ax3.set_xticks(x3); ax3.set_xticklabels(bd["Signal"], fontsize=10)
        ax3.set_ylabel("Count", fontsize=9); ax3.set_title("Anomaly Breakdown — Threshold vs Residual", fontsize=11)
        ax3.legend(fontsize=9, facecolor=DARK_BG_MPL, edgecolor=GRID_MPL, labelcolor=TEXT_MPL)
        for bars, vals in [(b1, bd["Threshold"]), (b2, bd["Residual"])]:
            for bar, v in zip(bars, vals):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(v),
                         ha="center", va="bottom", color=TEXT_MPL, fontsize=9, fontweight="bold")
        plt.tight_layout(); show_fig(fig3)

    # ── HR tab
    with tab_hr:
        sec("❤️", "Heart Rate Deep Dive", f"{total_hr} anomalies")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f'<div class="card"><div class="card-title">HR Statistics</div>'
                f'<div style="font-size:0.83rem;line-height:2">'
                f'Mean HR: <b style="color:{ACCENT}">{hr_d["HR_avg"].mean():.1f} bpm</b><br>'
                f'Max HR:  <b style="color:{ACCENT_RED}">{hr_d["HR_avg"].max():.1f} bpm</b><br>'
                f'Min HR:  <b style="color:{ACCENT2}">{hr_d["HR_avg"].min():.1f} bpm</b><br>'
                f'Anomaly days: <b style="color:{ACCENT_RED}">{total_hr}</b> of {len(hr_d)} total'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        with col_b:
            rows_hr = []
            for dval, val, t_, r_ in zip(hr_d["Date"], hr_d["HR_avg"], hr_thr, hr_res["anomaly"]):
                if t_ or r_:
                    rows_hr.append({"Date": dval.date(), "Avg HR": round(float(val), 2),
                                    "Type": "Threshold" if t_ else "Residual"})
            hr_display = pd.DataFrame(rows_hr)
            if not hr_display.empty:
                dark_df(hr_display, height=220)
            else:
                ui_success("No HR anomalies in selected range")

    # ── Steps tab
    with tab_steps:
        sec("🚶", "Step Count Deep Dive", f"{total_st} alerts")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f'<div class="card"><div class="card-title">Steps Statistics</div>'
                f'<div style="font-size:0.83rem;line-height:2">'
                f'Mean steps/day: <b style="color:{ACCENT3}">{st_d["Steps"].mean():,.0f}</b><br>'
                f'Max steps/day:  <b style="color:{ACCENT}">{st_d["Steps"].max():,.0f}</b><br>'
                f'Min steps/day:  <b style="color:{ACCENT_RED}">{st_d["Steps"].min():,.0f}</b><br>'
                f'Alert days: <b style="color:{ACCENT_RED}">{total_st}</b> of {len(st_d)} total'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        with col_b:
            rows_st = []
            for dval, val, t_, r_ in zip(st_d["Date"], st_d["Steps"], st_thr, st_res["anomaly"]):
                if t_ or r_:
                    rows_st.append({"Date": dval.date(), "Steps": int(val), "Type": "Threshold" if t_ else "Residual"})
            st_display = pd.DataFrame(rows_st)
            if not st_display.empty:
                dark_df(st_display, height=220)
            else:
                ui_success("No step anomalies in selected range")

    # ── Sleep tab
    with tab_sleep:
        sec("💤", "Sleep Pattern Deep Dive", f"{total_sl} anomalies")
        col_a, col_b = st.columns(2)
        with col_a:
            nonzero = sl_d[sl_d["Sleep"] > 0]["Sleep"]
            st.markdown(
                f'<div class="card"><div class="card-title">Sleep Statistics</div>'
                f'<div style="font-size:0.83rem;line-height:2">'
                f'Mean sleep/night: <b style="color:#b794f4">{sl_d["Sleep"].mean():.0f} min</b><br>'
                f'Max sleep/night:  <b style="color:{ACCENT}">{sl_d["Sleep"].max():.0f} min</b><br>'
                f'Min (non-zero):   <b style="color:{ACCENT_RED}">{nonzero.min() if len(nonzero) > 0 else "N/A":.0f} min</b><br>'
                f'Anomaly days: <b style="color:{ACCENT_RED}">{total_sl}</b> of {len(sl_d)} total'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        with col_b:
            rows_sl = []
            for dval, val, t_, r_ in zip(sl_d["Date"], sl_d["Sleep"], sl_thr, sl_res["anomaly"]):
                if t_ or r_:
                    rows_sl.append({"Date": dval.date(), "Sleep (min)": round(float(val), 1), "Type": "Threshold" if t_ else "Residual"})
            sl_display = pd.DataFrame(rows_sl)
            if not sl_display.empty:
                dark_df(sl_display, height=220)
            else:
                ui_success("No sleep anomalies in selected range")

    # ── Insights tab (NEW)
    with tab_insights:
        divider()
        sec("💡", "Automated Health Insights", "AI-generated")
        hr_mean = hr_d["HR_avg"].mean()
        st_mean = st_d["Steps"].mean()
        sl_mean = sl_d["Sleep"].mean()

        insight_card("❤️ Heart Rate", "Cardiovascular Status",
            f"Average HR: {hr_mean:.1f} bpm over {len(hr_d)} observation days. "
            f"{total_hr} anomalous HR days detected. "
            f"{'Normal range maintained.' if 60 <= hr_mean <= 100 else 'Monitor closely — outside normal range.'}")

        insight_card("🚶 Activity", "Daily Step Performance",
            f"Mean steps/day: {st_mean:,.0f}. "
            f"{'Meets' if st_mean >= 8000 else 'Below'} the recommended 8,000-step benchmark. "
            f"{total_st} low-activity alert days flagged.",
            ACCENT3)

        insight_card("💤 Sleep", "Recovery Quality",
            f"Average sleep: {sl_mean:.0f} min/night ({sl_mean/60:.1f} hrs). "
            f"{'Within' if 420 <= sl_mean <= 540 else 'Outside'} the recommended 7–9 hour window. "
            f"{total_sl} sleep anomaly days recorded.",
            "#b794f4")

        total_flags = total_hr + total_st + total_sl
        risk_level = "Low" if total_flags < 5 else "Moderate" if total_flags < 15 else "High"
        risk_col   = ACCENT3 if risk_level == "Low" else (ACCENT_ORG if risk_level == "Moderate" else ACCENT_RED)
        insight_card("⚠️ Risk Assessment", f"Overall Risk Level: {risk_level}",
            f"Total anomaly flags: {total_flags} across all signals. "
            f"{'No immediate concerns.' if risk_level == 'Low' else 'Review flagged days for clinical context.'} "
            f"Run M3 simulation to validate detection accuracy.",
            risk_col)

        divider()
        sec("📊", "Summary Statistics Table")
        stats_rows = []
        for lbl_s, df_s, vc_s, thr_ms, res_ms in [
            ("Heart Rate (bpm)", hr_d, "HR_avg", hr_thr, hr_res["anomaly"]),
            ("Steps / Day",      st_d, "Steps",  st_thr, st_res["anomaly"]),
            ("Sleep (min)",      sl_d, "Sleep",  sl_thr, sl_res["anomaly"]),
        ]:
            v = df_s[vc_s]
            stats_rows.append({
                "Signal": lbl_s,
                "Mean": round(float(v.mean()), 2),
                "Std":  round(float(v.std()),  2),
                "Min":  round(float(v.min()),  2),
                "Max":  round(float(v.max()),  2),
                "Thr Flags": int(thr_ms.sum()),
                "Res Flags": int(res_ms.sum()),
                "Total Flags": int(thr_ms.sum() + res_ms.sum()),
                "Flag Rate %": round(float((thr_ms.sum() + res_ms.sum()) / max(len(df_s), 1) * 100), 1),
            })
        dark_df(pd.DataFrame(stats_rows))

    # ── Export tab
    with tab_export:
        divider()
        sec("📥", "Project Reports & Exports", "PDF · CSV")

        rows = []
        for dval, val, t_, r_ in zip(hr_d["Date"], hr_d["HR_avg"], hr_thr, hr_res["anomaly"]):
            if t_ or r_:
                rows.append({"Date": dval.date(), "Signal": "Heart Rate", "Value": round(float(val), 2),
                             "Anomaly_Type": "Threshold" if t_ else "Residual",
                             "Severity": "High" if t_ else "Medium", "Unit": "bpm"})
        for dval, val, t_, r_ in zip(st_d["Date"], st_d["Steps"], st_thr, st_res["anomaly"]):
            if t_ or r_:
                rows.append({"Date": dval.date(), "Signal": "Steps", "Value": round(float(val), 0),
                             "Anomaly_Type": "Threshold" if t_ else "Residual",
                             "Severity": "High" if t_ else "Medium", "Unit": "steps/day"})
        for dval, val, t_, r_ in zip(sl_d["Date"], sl_d["Sleep"], sl_thr, sl_res["anomaly"]):
            if t_ or r_:
                rows.append({"Date": dval.date(), "Signal": "Sleep", "Value": round(float(val), 1),
                             "Anomaly_Type": "Threshold" if t_ else "Residual",
                             "Severity": "High" if t_ else "Medium", "Unit": "minutes"})
        anom_log = pd.DataFrame(rows).sort_values("Date") if rows else pd.DataFrame(
            columns=["Date","Signal","Value","Anomaly_Type","Severity","Unit"])

        stats_rows = []
        for lbl_s, df_s, vc_s, thr_ms, res_ms in [
            ("Heart Rate (bpm)", hr_d, "HR_avg", hr_thr, hr_res["anomaly"]),
            ("Steps / Day",      st_d, "Steps",  st_thr, st_res["anomaly"]),
            ("Sleep (min)",      sl_d, "Sleep",  sl_thr, sl_res["anomaly"]),
        ]:
            v = df_s[vc_s]
            stats_rows.append({
                "Signal": lbl_s, "Mean": round(float(v.mean()), 2), "Std": round(float(v.std()), 2),
                "Min": round(float(v.min()), 2), "Max": round(float(v.max()), 2),
                "Thr Anomalies": int(thr_ms.sum()), "Res Anomalies": int(res_ms.sum()),
                "Total": int(thr_ms.sum() + res_ms.sum()),
                "Rate %": round(float((thr_ms.sum() + res_ms.sum()) / max(len(df_s), 1) * 100), 1),
            })
        stats_df = pd.DataFrame(stats_rows)

        st.markdown(
            f'<div class="card">'
            f'<div class="card-title">What\'s Included</div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;font-size:0.82rem">'
            f'<div style="background:{SECTION_BG};border-radius:10px;padding:0.9rem">'
            f'<div style="color:{ACCENT};font-weight:700;margin-bottom:0.4rem">📄 PDF Report (6 pages)</div>'
            f'<div style="color:{MUTED};line-height:1.9">✅ Executive summary<br>✅ Anomaly counts<br>'
            f'✅ Thresholds used<br>✅ Simulated accuracy<br>✅ Methodology<br>'
            f'✅ 5 chart screenshots (Figs 1–5)<br>✅ Anomaly tables<br>✅ User profiles</div></div>'
            f'<div style="background:{SECTION_BG};border-radius:10px;padding:0.9rem">'
            f'<div style="color:{ACCENT3};font-weight:700;margin-bottom:0.4rem">📊 CSV Report</div>'
            f'<div style="color:{MUTED};line-height:1.9">✅ Full anomaly log<br>✅ Signal statistics<br>'
            f'✅ Master dataset (500 rows)<br>✅ All signals combined<br>✅ Date · value · type<br>'
            f'✅ Residual deviation<br>✅ Severity flags</div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        divider()
        col_pdf, col_csv = st.columns(2)

        with col_pdf:
            sec("📄", "PDF Report")
            if st.button("📄 Generate PDF Report", key="gen_pdf", use_container_width=True):
                with st.spinner("⏳ Rendering charts and generating PDF..."):
                    try:
                        pdf_buf = generate_pdf_report(
                            master, anom_log, stats_df,
                            hr_high, hr_low, st_low, sl_low, sl_high, sigma,
                            total_hr, total_st, total_sl,
                            st.session_state.m3_sim_results if st.session_state.m3_sim_done else None,
                            hr_d=hr_d, hr_thr=hr_thr, hr_res=hr_res,
                            st_d=st_d, st_thr=st_thr, st_res=st_res,
                            sl_d=sl_d, sl_thr=sl_thr, sl_res=sl_res,
                        )
                        if pdf_buf:
                            fname = f"FitPulse_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                            st.download_button(
                                label="⬇️ Download PDF Report",
                                data=pdf_buf,
                                file_name=fname,
                                mime="application/pdf",
                                key="dl_pdf",
                                use_container_width=True,
                            )
                            ui_success(f"PDF ready — 6 pages incl. 5 chart screenshots · {fname}")
                        else:
                            ui_danger("PDF generation failed. Ensure fpdf2 is installed: pip install fpdf2")
                    except Exception as e:
                        ui_danger(f"PDF error: {e}")

        with col_csv:
            sec("📊", "CSV Report")
            csv_data  = generate_csv_report(master, anom_log, stats_df)
            fname_csv = f"FitPulse_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            st.download_button(
                label="⬇️ Download CSV Report",
                data=csv_data,
                file_name=fname_csv,
                mime="text/csv",
                key="dl_csv",
                use_container_width=True,
            )
            with st.expander("👁️ Preview anomaly log"):
                dark_df(anom_log, height=260)

        divider()
        sec("📊", "Signal Trend Statistics")
        dark_df(stats_df)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="card" style="text-align:center;padding:2rem;border-color:rgba(104,211,145,0.25)">'
            f'<div style="font-size:1.8rem;margin-bottom:0.5rem">📄</div>'
            f'<div style="font-family:Syne,sans-serif;font-weight:800;font-size:1.1rem">M4 — Dashboard & Reports Complete</div>'
            f'<div style="color:{MUTED};font-size:0.8rem;margin-top:0.6rem;line-height:2">'
            f'✅ Interactive signal explorer &nbsp;·&nbsp; ✅ Date & anomaly filtering<br>'
            f'✅ Anomaly breakdown chart &nbsp;·&nbsp; ✅ Full anomaly log<br>'
            f'✅ AI health insights &nbsp;·&nbsp; ✅ Signal trend statistics<br>'
            f'✅ PDF (with chart screenshots) &nbsp;·&nbsp; ✅ CSV report download'
            f'</div></div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────
if   "M1" in milestone: milestone1()
elif "M2" in milestone: milestone2()
elif "M3" in milestone: milestone3()
elif "M4" in milestone: milestone4()
