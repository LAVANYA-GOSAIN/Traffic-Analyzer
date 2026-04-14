# # /Users/lavanyagosain/Desktop/Skilldevelopment/traffic_data.csv

# ==========================================
# Traffic Congestion Analysis — Streamlit GUI
# Single Model: Linear Regression
# ==========================================

# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score, mean_absolute_error

# # ── Page config ───────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Traffic Congestion Analyzer",
#     page_icon="🚦",
#     layout="wide",
# )

# # ── Custom CSS ────────────────────────────────────────────────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

# html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
# .stApp { background: #0d0f14; color: #e8eaf0; }

# section[data-testid="stSidebar"] {
#     background: #13161d !important;
#     border-right: 1px solid #1f2430;
# }
# section[data-testid="stSidebar"] * { color: #c9cdd8 !important; }

# /* Hero */
# .hero {
#     background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 60%, #0d1b2a 100%);
#     border: 1px solid #2a3a5c;
#     border-radius: 16px;
#     padding: 1.8rem 2.5rem;
#     margin-bottom: 1.5rem;
# }
# .hero-title { font-size: 2rem; font-weight: 800; color: #e8f4fd; margin: 0 0 0.25rem 0; }
# .hero-sub {
#     font-family: 'DM Mono', monospace; font-size: 0.75rem;
#     color: #63b3ed; letter-spacing: 2px; text-transform: uppercase; margin: 0;
# }

# /* KPI cards */
# .kpi {
#     background: #13161d; border: 1px solid #1f2430;
#     border-radius: 12px; padding: 1rem 1.2rem; text-align: center;
# }
# .kpi-label {
#     font-family: 'DM Mono', monospace; font-size: 0.65rem;
#     letter-spacing: 1.5px; color: #6b7280; text-transform: uppercase; margin-bottom: 0.3rem;
# }
# .kpi-value { font-size: 1.6rem; font-weight: 700; color: #63b3ed; }
# .kpi-sub   { font-size: 0.7rem; color: #9ca3af; margin-top: 0.2rem; }

# /* Section headers */
# .sh {
#     font-size: 1rem; font-weight: 700; color: #e8eaf0;
#     border-left: 3px solid #63b3ed; padding-left: 0.7rem;
#     margin: 1.6rem 0 0.9rem 0;
# }

# /* Predict button */
# div.stButton > button {
#     background: linear-gradient(135deg, #1e3a5f, #2a5298);
#     color: #e8f4fd; border: 1px solid #3b6abf;
#     border-radius: 10px; font-family: 'Syne', sans-serif;
#     font-weight: 700; font-size: 1rem;
#     padding: 0.65rem 2rem; width: 100%;
# }
# div.stButton > button:hover {
#     background: linear-gradient(135deg, #2a5298, #3b7de8);
#     border-color: #63b3ed;
# }

# /* Result box */
# .rbox { border-radius: 16px; padding: 1.8rem 1.5rem; text-align: center; margin-top: 1rem; }
# .r-heavy    { background: linear-gradient(135deg,#2d1515,#4a1f1f); border: 2px solid #f87171; }
# .r-moderate { background: linear-gradient(135deg,#2d2015,#4a3a1f); border: 2px solid #fbbf24; }
# .r-free     { background: linear-gradient(135deg,#152d1e,#1f4a2e); border: 2px solid #34d399; }
# .r-speed { font-size: 3rem; font-weight: 800; letter-spacing: -1px; line-height: 1; }
# .r-label {
#     font-family: 'DM Mono', monospace; font-size: 0.88rem;
#     letter-spacing: 2.5px; text-transform: uppercase; margin-top: 0.5rem; font-weight: 600;
# }
# .r-meta { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #9ca3af; margin-top: 0.8rem; }
# </style>
# """, unsafe_allow_html=True)

# # ── Constants ─────────────────────────────────────────────────────────────────
# CSV_PATH = "/Users/lavanyagosain/Desktop/Skilldevelopment/traffic_data.csv"
# FEATURES = ["density_veh_per_km", "occupancy_pct", "avg_wait_time_s"]
# TARGET   = "avg_speed_kmph"

# PAL = {
#     "bg": "#0d0f14", "card": "#13161d", "border": "#1f2430",
#     "accent": "#63b3ed", "text": "#e8eaf0", "muted": "#6b7280",
# }

# def plot_style():
#     plt.rcParams.update({
#         "figure.facecolor": PAL["bg"],    "axes.facecolor":  PAL["card"],
#         "axes.edgecolor":   PAL["border"],"axes.labelcolor": PAL["text"],
#         "xtick.color":      PAL["muted"], "ytick.color":     PAL["muted"],
#         "text.color":       PAL["text"],  "grid.color":      PAL["border"],
#         "grid.linestyle":   "--",         "grid.linewidth":  0.5,
#         "font.family":      "monospace",
#     })

# def congestion(speed):
#     if speed < 20:   return "Heavy Congestion",    "#f87171", "r-heavy",    "🔴"
#     elif speed < 40: return "Moderate Congestion", "#fbbf24", "r-moderate", "🟡"
#     else:            return "Free Flow",           "#34d399", "r-free",     "🟢"

# # ── Train (cached) ────────────────────────────────────────────────────────────
# @st.cache_resource(show_spinner=False)
# def train():
#     df = pd.read_csv(CSV_PATH)[FEATURES + [TARGET]].dropna()
#     X, y = df[FEATURES], df[TARGET]
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42)
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     r2  = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     return df, model, r2, mae, y_test, y_pred

# # ── Hero ──────────────────────────────────────────────────────────────────────
# st.markdown("""
# <div class="hero">
#   <p class="hero-sub">🚦 Linear Regression · Traffic Analysis</p>
#   <h1 class="hero-title">Traffic Congestion Analyzer</h1>
# </div>
# """, unsafe_allow_html=True)

# # ── Sidebar ───────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("### 📂 Dataset")
#     st.code(CSV_PATH, language="bash")
#     st.markdown("**Features used**")
#     for f in FEATURES:
#         st.markdown(f"- `{f}`")
#     st.markdown(f"**Target:** `{TARGET}`")
#     st.markdown("---")
#     st.markdown("**Model:** Linear Regression")
#     st.markdown("**Test split:** 20%  |  Seed: 42")

# # ── Load ──────────────────────────────────────────────────────────────────────
# placeholder = st.empty()
# with placeholder.container():
#     with st.spinner("⏳ Loading dataset & training model…"):
#         try:
#             df, model, r2, mae, y_test, y_pred = train()
#         except FileNotFoundError:
#             st.error(f"❌ CSV not found at:\n`{CSV_PATH}`\n\nUpdate `CSV_PATH` in the script.")
#             st.stop()
# placeholder.empty()

# plot_style()

# # ── KPI Row ───────────────────────────────────────────────────────────────────
# k1, k2, k3, k4 = st.columns(4)
# kpis = [
#     ("DATASET ROWS", f"{len(df):,}",     "after dropna"),
#     ("MODEL",        "Linear Reg.",      "sklearn"),
#     ("R² SCORE",     f"{r2:.4f}",        "on test set"),
#     ("MAE",          f"{mae:.4f} km/h",  "mean abs error"),
# ]
# for col, (lbl, val, sub) in zip([k1, k2, k3, k4], kpis):
#     col.markdown(f"""
#     <div class="kpi">
#       <div class="kpi-label">{lbl}</div>
#       <div class="kpi-value">{val}</div>
#       <div class="kpi-sub">{sub}</div>
#     </div>""", unsafe_allow_html=True)

# # ── Main columns ──────────────────────────────────────────────────────────────
# left, right = st.columns([3, 2], gap="large")

# # ────────────────── LEFT: Charts ──────────────────────────────────────────────
# with left:

#     # EDA row
#     st.markdown('<div class="sh">Exploratory Data Analysis</div>', unsafe_allow_html=True)
#     c1, c2 = st.columns(2)

#     with c1:
#         fig, ax = plt.subplots(figsize=(4.5, 3.5))
#         cmap = sns.diverging_palette(220, 10, as_cmap=True)
#         sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap=cmap,
#                     linewidths=0.5, linecolor=PAL["border"],
#                     ax=ax, cbar_kws={"shrink": 0.8})
#         ax.set_title("Correlation Heatmap", fontsize=10, pad=8)
#         plt.tight_layout()
#         st.pyplot(fig, use_container_width=True)
#         plt.close()

#     with c2:
#         fig, ax = plt.subplots(figsize=(4.5, 3.5))
#         ax.scatter(df["density_veh_per_km"], df["avg_speed_kmph"],
#                    alpha=0.45, color=PAL["accent"], s=12, edgecolors="none")
#         ax.set_xlabel("Vehicle Density (veh/km)")
#         ax.set_ylabel("Avg Speed (km/h)")
#         ax.set_title("Density vs Speed", fontsize=10, pad=8)
#         ax.grid(True, alpha=0.3)
#         plt.tight_layout()
#         st.pyplot(fig, use_container_width=True)
#         plt.close()

#     # Actual vs Predicted
#     st.markdown('<div class="sh">Actual vs Predicted Speed</div>', unsafe_allow_html=True)
#     fig, ax = plt.subplots(figsize=(7, 3.8))
#     ax.scatter(y_test, y_pred, alpha=0.5, color=PAL["accent"], s=14, edgecolors="none")
#     mn = min(y_test.min(), y_pred.min())
#     mx = max(y_test.max(), y_pred.max())
#     ax.plot([mn, mx], [mn, mx], color="#f87171", linewidth=1.2, linestyle="--", label="Perfect fit")
#     ax.set_xlabel("Actual Speed (km/h)")
#     ax.set_ylabel("Predicted Speed (km/h)")
#     ax.set_title(f"Actual vs Predicted  (R² = {r2:.4f})", fontsize=10, pad=8)
#     ax.legend(fontsize=8)
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     st.pyplot(fig, use_container_width=True)
#     plt.close()

#     # Residuals
#     st.markdown('<div class="sh">Residuals Distribution</div>', unsafe_allow_html=True)
#     residuals = y_test.values - y_pred
#     fig, ax = plt.subplots(figsize=(7, 3))
#     ax.hist(residuals, bins=30, color=PAL["accent"], alpha=0.75, edgecolor=PAL["border"])
#     ax.axvline(0, color="#f87171", linewidth=1.2, linestyle="--")
#     ax.set_xlabel("Residual (Actual − Predicted)")
#     ax.set_ylabel("Count")
#     ax.set_title("Residuals Histogram", fontsize=10, pad=8)
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     st.pyplot(fig, use_container_width=True)
#     plt.close()

# # ────────────────── RIGHT: Prediction ────────────────────────────────────────
# with right:
#     st.markdown('<div class="sh">🔮 Predict Traffic Condition</div>', unsafe_allow_html=True)

#     st.markdown("""
#     <div style="font-size:0.82rem; color:#9ca3af; margin-bottom:1.2rem; line-height:1.7;">
#     Enter current road conditions. The Linear Regression model will predict
#     average speed and classify the congestion level.
#     </div>
#     """, unsafe_allow_html=True)

#     density   = st.number_input("🚗  Vehicle Density (veh/km)",
#                                 min_value=0.0, max_value=500.0, value=80.0, step=1.0)
#     occupancy = st.number_input("📊  Road Occupancy (%)",
#                                 min_value=0.0, max_value=100.0, value=45.0, step=0.5)
#     wait_time = st.number_input("⏱️  Avg Waiting Time (sec)",
#                                 min_value=0.0, max_value=600.0, value=30.0, step=1.0)

#     st.markdown("<br>", unsafe_allow_html=True)
#     btn = st.button("▶  Predict Now")

#     if btn:
#         inp   = pd.DataFrame([[density, occupancy, wait_time]], columns=FEATURES)
#         speed = model.predict(inp)[0]
#         label, color, css, emoji = congestion(speed)

#         st.markdown(f"""
#         <div class="rbox {css}">
#           <div style="font-family:'DM Mono',monospace; font-size:0.7rem;
#                       letter-spacing:2px; color:{color}; opacity:0.7;
#                       text-transform:uppercase; margin-bottom:0.5rem;">
#             Predicted Speed
#           </div>
#           <div class="r-speed" style="color:{color}">
#             {speed:.1f}<span style="font-size:1.1rem;font-weight:400"> km/h</span>
#           </div>
#           <div class="r-label" style="color:{color}">{emoji}&nbsp; {label}</div>
#           <hr style="border-color:{color}33; margin:1rem 0 0.8rem 0;">
#           <div style="display:flex; justify-content:space-around;
#                       font-family:'DM Mono',monospace; font-size:0.7rem; color:#9ca3af;">
#             <div><span style="color:{color}">●</span> {density} veh/km</div>
#             <div><span style="color:{color}">●</span> {occupancy}% occ.</div>
#             <div><span style="color:{color}">●</span> {wait_time}s wait</div>
#           </div>
#           <div class="r-meta">Model: Linear Regression &nbsp;|&nbsp; R² {r2:.4f} &nbsp;|&nbsp; MAE {mae:.4f}</div>
#         </div>
#         """, unsafe_allow_html=True)

#         # Scale legend
#         st.markdown("""
#         <div style="margin-top:1rem; background:#13161d; border:1px solid #1f2430;
#                     border-radius:10px; padding:0.9rem 1.1rem;">
#           <div style="font-family:'DM Mono',monospace; font-size:0.68rem; color:#9ca3af;">
#             <div style="color:#e8eaf0; font-weight:700; margin-bottom:0.45rem; font-size:0.75rem;">
#               Congestion Scale
#             </div>
#             <div>🔴 &nbsp;<span style="color:#f87171">Heavy Congestion</span> — speed &lt; 20 km/h</div>
#             <div style="margin-top:0.3rem;">🟡 &nbsp;<span style="color:#fbbf24">Moderate Congestion</span> — 20 – 40 km/h</div>
#             <div style="margin-top:0.3rem;">🟢 &nbsp;<span style="color:#34d399">Free Flow</span> — speed &gt; 40 km/h</div>
#           </div>
#         </div>
#         """, unsafe_allow_html=True)

#     else:
#         st.markdown("""
#         <div style="background:#13161d; border:1px dashed #2a3a5c; border-radius:14px;
#                     padding:3rem 1.5rem; text-align:center; color:#4b5563; margin-top:0.5rem;">
#           <div style="font-size:2.5rem; margin-bottom:0.6rem;">🚦</div>
#           <div style="font-family:'DM Mono',monospace; font-size:0.78rem; letter-spacing:1px;">
#             Enter values above and<br>
#             click <strong style="color:#63b3ed">Predict Now</strong>
#           </div>
#         </div>
#         """, unsafe_allow_html=True)

# # ── Footer ────────────────────────────────────────────────────────────────────
# st.markdown("---")
# st.caption("Traffic Congestion Analysis · Linear Regression · Built with Streamlit")



# ==========================================
# Traffic Congestion Analysis — Streamlit GUI
# Single Model: Linear Regression
# ==========================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Congestion Analyzer",
    page_icon="🚦",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0d0f14; color: #e8eaf0; }

section[data-testid="stSidebar"] {
    background: #13161d !important;
    border-right: 1px solid #1f2430;
}
section[data-testid="stSidebar"] * { color: #c9cdd8 !important; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 60%, #0d1b2a 100%);
    border: 1px solid #2a3a5c;
    border-radius: 16px;
    padding: 1.8rem 2.5rem;
    margin-bottom: 1.5rem;
}
.hero-title { font-size: 2rem; font-weight: 800; color: #e8f4fd; margin: 0 0 0.25rem 0; }
.hero-sub {
    font-family: 'DM Mono', monospace; font-size: 0.75rem;
    color: #63b3ed; letter-spacing: 2px; text-transform: uppercase; margin: 0;
}

/* KPI cards */
.kpi {
    background: #13161d; border: 1px solid #1f2430;
    border-radius: 12px; padding: 1rem 1.2rem; text-align: center;
}
.kpi-label {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    letter-spacing: 1.5px; color: #6b7280; text-transform: uppercase; margin-bottom: 0.3rem;
}
.kpi-value { font-size: 1.6rem; font-weight: 700; color: #63b3ed; }
.kpi-sub   { font-size: 0.7rem; color: #9ca3af; margin-top: 0.2rem; }

/* Section headers */
.sh {
    font-size: 1rem; font-weight: 700; color: #e8eaf0;
    border-left: 3px solid #63b3ed; padding-left: 0.7rem;
    margin: 1.6rem 0 0.9rem 0;
}

/* Slider labels */
div[data-testid="stSlider"] label p {
    font-size: 0.88rem !important;
    color: #c9cdd8 !important;
    font-weight: 600 !important;
}

/* Slider value display */
div[data-testid="stSlider"] div[data-testid="stTickBar"] {
    display: none;
}

/* Predict button */
div.stButton > button {
    background: linear-gradient(135deg, #1e3a5f, #2a5298);
    color: #e8f4fd; border: 1px solid #3b6abf;
    border-radius: 10px; font-family: 'Syne', sans-serif;
    font-weight: 700; font-size: 1rem;
    padding: 0.7rem 2rem; width: 100%;
    margin-top: 0.5rem;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #2a5298, #3b7de8);
    border-color: #63b3ed;
}

/* Result box */
.rbox { border-radius: 16px; padding: 1.8rem 1.5rem; text-align: center; margin-top: 1rem; }
.r-heavy    { background: linear-gradient(135deg,#2d1515,#4a1f1f); border: 2px solid #f87171; }
.r-moderate { background: linear-gradient(135deg,#2d2015,#4a3a1f); border: 2px solid #fbbf24; }
.r-free     { background: linear-gradient(135deg,#152d1e,#1f4a2e); border: 2px solid #34d399; }
.r-speed { font-size: 3rem; font-weight: 800; letter-spacing: -1px; line-height: 1; }
.r-label {
    font-family: 'DM Mono', monospace; font-size: 0.88rem;
    letter-spacing: 2.5px; text-transform: uppercase; margin-top: 0.5rem; font-weight: 600;
}
.r-meta { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #9ca3af; margin-top: 0.8rem; }

/* Input value badges */
.val-badge {
    display: inline-block;
    background: #1a2744;
    border: 1px solid #2a3a5c;
    border-radius: 8px;
    padding: 0.15rem 0.6rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #63b3ed;
    font-weight: 600;
    margin-left: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CSV_PATH = "/Users/lavanyagosain/Desktop/Skilldevelopment/traffic_data.csv"
FEATURES = ["density_veh_per_km", "occupancy_pct", "avg_wait_time_s"]
TARGET   = "avg_speed_kmph"

PAL = {
    "bg": "#0d0f14", "card": "#13161d", "border": "#1f2430",
    "accent": "#63b3ed", "text": "#e8eaf0", "muted": "#6b7280",
}

def plot_style():
    plt.rcParams.update({
        "figure.facecolor": PAL["bg"],    "axes.facecolor":  PAL["card"],
        "axes.edgecolor":   PAL["border"],"axes.labelcolor": PAL["text"],
        "xtick.color":      PAL["muted"], "ytick.color":     PAL["muted"],
        "text.color":       PAL["text"],  "grid.color":      PAL["border"],
        "grid.linestyle":   "--",         "grid.linewidth":  0.5,
        "font.family":      "monospace",
    })

def get_congestion(speed):
    if speed < 20:   return "Heavy Congestion",    "#f87171", "r-heavy",    "🔴"
    elif speed < 40: return "Moderate Congestion", "#fbbf24", "r-moderate", "🟡"
    else:            return "Free Flow",           "#34d399", "r-free",     "🟢"

# ── Train (cached — runs only once) ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train():
    df = pd.read_csv(CSV_PATH)[FEATURES + [TARGET]].dropna()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return df, model, r2, mae, y_test, y_pred

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-sub">🚦 Linear Regression · Traffic Analysis</p>
  <h1 class="hero-title">Traffic Congestion Analyzer</h1>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Dataset")
    st.code(CSV_PATH, language="bash")
    st.markdown("**Features used**")
    for f in FEATURES:
        st.markdown(f"- `{f}`")
    st.markdown(f"**Target:** `{TARGET}`")
    st.markdown("---")
    st.markdown("**Model:** Linear Regression")
    st.markdown("**Test split:** 20%  |  Seed: 42")

# ── Load ──────────────────────────────────────────────────────────────────────
slot = st.empty()
with slot.container():
    with st.spinner("⏳ Loading dataset & training model…"):
        try:
            df, model, r2, mae, y_test, y_pred = train()
        except FileNotFoundError:
            st.error(f"❌ CSV not found at:\n`{CSV_PATH}`\n\nUpdate `CSV_PATH` at the top of the script.")
            st.stop()
slot.empty()

plot_style()

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
kpis = [
    ("DATASET ROWS", f"{len(df):,}",     "after dropna"),
    ("MODEL",        "Linear Reg.",      "sklearn"),
    ("R² SCORE",     f"{r2:.4f}",        "on test set"),
    ("MAE",          f"{mae:.4f} km/h",  "mean abs error"),
]
for col, (lbl, val, sub) in zip([k1, k2, k3, k4], kpis):
    col.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">{lbl}</div>
      <div class="kpi-value">{val}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

# ─────────────── LEFT: Charts ─────────────────────────────────────────────────
with left:

    st.markdown('<div class="sh">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap=cmap,
                    linewidths=0.5, linecolor=PAL["border"],
                    ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title("Correlation Heatmap", fontsize=10, pad=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.scatter(df["density_veh_per_km"], df["avg_speed_kmph"],
                   alpha=0.45, color=PAL["accent"], s=12, edgecolors="none")
        ax.set_xlabel("Vehicle Density (veh/km)")
        ax.set_ylabel("Avg Speed (km/h)")
        ax.set_title("Density vs Speed", fontsize=10, pad=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown('<div class="sh">Actual vs Predicted Speed</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.scatter(y_test, y_pred, alpha=0.5, color=PAL["accent"], s=14, edgecolors="none")
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], color="#f87171", linewidth=1.2,
            linestyle="--", label="Perfect fit")
    ax.set_xlabel("Actual Speed (km/h)")
    ax.set_ylabel("Predicted Speed (km/h)")
    ax.set_title(f"Actual vs Predicted  (R² = {r2:.4f})", fontsize=10, pad=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown('<div class="sh">Residuals Distribution</div>', unsafe_allow_html=True)
    residuals = y_test.values - y_pred
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(residuals, bins=30, color=PAL["accent"], alpha=0.75, edgecolor=PAL["border"])
    ax.axvline(0, color="#f87171", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Residual (Actual − Predicted)")
    ax.set_ylabel("Count")
    ax.set_title("Residuals Histogram", fontsize=10, pad=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ─────────────── RIGHT: Prediction with Sliders ───────────────────────────────
with right:
    st.markdown('<div class="sh">🔮 Predict Traffic Condition</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.82rem; color:#9ca3af; margin-bottom:1.4rem; line-height:1.7;">
    Drag the sliders to set road conditions. The result updates instantly when you click <b style="color:#63b3ed">Predict Now</b>.
    </div>
    """, unsafe_allow_html=True)

    # ── Sliders ───────────────────────────────────────────────────────────────
    density = st.slider(
        "🚗  Vehicle Density (veh/km)",
        min_value=0.0,
        max_value=300.0,
        value=80.0,
        step=1.0,
    )

    occupancy = st.slider(
        "📊  Road Occupancy (%)",
        min_value=0.0,
        max_value=100.0,
        value=45.0,
        step=0.5,
    )

    wait_time = st.slider(
        "⏱️  Avg Waiting Time (sec)",
        min_value=0.0,
        max_value=300.0,
        value=30.0,
        step=1.0,
    )

    # Show current values as badges
    st.markdown(f"""
    <div style="background:#13161d; border:1px solid #1f2430; border-radius:10px;
                padding:0.75rem 1rem; margin: 0.8rem 0 1rem 0;
                font-family:'DM Mono',monospace; font-size:0.72rem; color:#9ca3af;
                display:flex; justify-content:space-between;">
      <div>Density <span class="val-badge">{density:.0f}</span></div>
      <div>Occupancy <span class="val-badge">{occupancy:.1f}%</span></div>
      <div>Wait <span class="val-badge">{wait_time:.0f}s</span></div>
    </div>
    """, unsafe_allow_html=True)

    btn = st.button("▶  Predict Now")

    # ── Result ────────────────────────────────────────────────────────────────
    if btn:
        inp   = pd.DataFrame([[density, occupancy, wait_time]], columns=FEATURES)
        speed = model.predict(inp)[0]
        label, color, css, emoji = get_congestion(speed)

        st.markdown(f"""
        <div class="rbox {css}">
          <div style="font-family:'DM Mono',monospace; font-size:0.68rem;
                      letter-spacing:2px; color:{color}; opacity:0.7;
                      text-transform:uppercase; margin-bottom:0.5rem;">
            Predicted Speed
          </div>
          <div class="r-speed" style="color:{color}">
            {speed:.1f}<span style="font-size:1.1rem;font-weight:400"> km/h</span>
          </div>
          <div class="r-label" style="color:{color}">{emoji}&nbsp; {label}</div>
          <hr style="border-color:{color}33; margin:1rem 0 0.8rem 0;">
          <div style="display:flex; justify-content:space-around;
                      font-family:'DM Mono',monospace; font-size:0.7rem; color:#9ca3af;">
            <div><span style="color:{color}">●</span> {density:.0f} veh/km</div>
            <div><span style="color:{color}">●</span> {occupancy:.1f}% occ.</div>
            <div><span style="color:{color}">●</span> {wait_time:.0f}s wait</div>
          </div>
          <div class="r-meta">
            Linear Regression &nbsp;|&nbsp; R² {r2:.4f} &nbsp;|&nbsp; MAE {mae:.4f}
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:1rem; background:#13161d; border:1px solid #1f2430;
                    border-radius:10px; padding:0.9rem 1.1rem;">
          <div style="font-family:'DM Mono',monospace; font-size:0.68rem; color:#9ca3af;">
            <div style="color:#e8eaf0; font-weight:700; margin-bottom:0.45rem; font-size:0.75rem;">
              Congestion Scale
            </div>
            <div>🔴 &nbsp;<span style="color:#f87171">Heavy Congestion</span> — speed &lt; 20 km/h</div>
            <div style="margin-top:0.3rem;">🟡 &nbsp;<span style="color:#fbbf24">Moderate Congestion</span> — 20 – 40 km/h</div>
            <div style="margin-top:0.3rem;">🟢 &nbsp;<span style="color:#34d399">Free Flow</span> — speed &gt; 40 km/h</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#13161d; border:1px dashed #2a3a5c; border-radius:14px;
                    padding:3rem 1.5rem; text-align:center; color:#4b5563; margin-top:0.5rem;">
          <div style="font-size:2.5rem; margin-bottom:0.6rem;">🚦</div>
          <div style="font-family:'DM Mono',monospace; font-size:0.78rem; letter-spacing:1px;">
            Adjust sliders above and<br>
            click <strong style="color:#63b3ed">Predict Now</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Traffic Congestion Analysis · Linear Regression · Built with Streamlit")