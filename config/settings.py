"""
SmartCart Clustering System — Configuration & Constants
═══════════════════════════════════════════════════════
Central place for paths, colours, chart defaults, and
page configuration. Import from here instead of
hard-coding values across modules.
"""

import os

# ── Resolve project root (one level up from this file's directory) ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── File Paths ──
CSS_PATH = os.path.join(BASE_DIR, "assets", "styles.css")
DATA_PATH = os.path.join(BASE_DIR, "smartcart_customers.csv")

# ── Streamlit Page Config (passed to st.set_page_config) ──
PAGE_CONFIG = {
    "page_title": "CustomerIQ AI",
    "page_icon": "🛒",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# ── Cluster Colour Palette ──
CLUSTER_COLORS = {
    "Cluster 0": "#6C63FF",
    "Cluster 1": "#06B6D4",
    "Cluster 2": "#10B981",
    "Cluster 3": "#F59E0B",
}

GRADIENT_COLORS = ["#6C63FF", "#3B82F6", "#06B6D4", "#10B981", "#A855F7"]

# ── Plotly Chart Defaults ──
CHART_TEMPLATE = "plotly_dark"
CHART_FONT = dict(family="Inter")
CHART_BG = "rgba(0,0,0,0)"

# ── Navigation Page Labels ──
PAGE_LABELS = [
    "🏠 Project Overview",
    "📊 Exploratory Analysis",
    "🔬 Clustering Pipeline",
    "🎯 Cluster Profiles",
    "🔮 Predict Your Cluster",
]
