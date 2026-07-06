"""
SmartCart Customer Clustering System — Streamlit Dashboard
═══════════════════════════════════════════════════════════
Author : Aniket
Purpose: Interactive ML dashboard for customer segmentation
Deploy : Streamlit Community Cloud (free)

HOW STREAMLIT WORKS (for learning):
────────────────────────────────────
1. Streamlit re-runs this ENTIRE script top-to-bottom every time
   the user interacts (clicks a button, moves a slider).
2. @st.cache_data prevents re-loading/re-computing data on every rerun.
3. st.sidebar creates a navigation panel on the left.
4. st.columns() creates horizontal layouts.
5. Plotly charts are interactive (hover, zoom, rotate 3D).

PROJECT STRUCTURE:
──────────────────
  app.py                  ← YOU ARE HERE (entry point)
  assets/styles.css       ← All CSS (dark theme, cards, animations)
  config/settings.py      ← Constants (colors, paths, chart defaults)
  data/loader.py          ← Data pipeline (load, preprocess, cluster)
  components/sidebar.py   ← Sidebar navigation + quick stats
  components/ui_helpers.py← Reusable UI functions (cards, profiles)
  pages_app/overview.py   ← Page 1: Project Overview
  pages_app/eda.py        ← Page 2: Exploratory Analysis
  pages_app/clustering.py ← Page 3: Clustering Pipeline
  pages_app/profiles.py   ← Page 4: Cluster Profiles
  pages_app/predict.py    ← Page 5: Predict Your Cluster
"""

# ══════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════

import streamlit as st

from config.settings import PAGE_CONFIG, CSS_PATH
from components.ui_helpers import inject_css
from components.sidebar import render_sidebar
from data.loader import load_and_preprocess, compute_clustering
from pages_app import overview, eda, clustering, profiles, predict


# ══════════════════════════════════════════════
# PAGE CONFIGURATION  (must be the FIRST Streamlit command)
# ══════════════════════════════════════════════

st.set_page_config(**PAGE_CONFIG)


# ══════════════════════════════════════════════
# INJECT CSS FROM EXTERNAL FILE
# ══════════════════════════════════════════════
# Loads assets/styles.css and injects it into the page.
# This fixes the background visibility bug — the CSS forces
# the dark background even if Streamlit's built-in theme
# doesn't apply in time.

inject_css(CSS_PATH)


# ══════════════════════════════════════════════
# LOAD DATA (runs once, then cached)
# ══════════════════════════════════════════════

(df_raw, df_full, df_cleaned, df_encoded,
 X_scaled, pca_data, pca, scaler, ohe) = load_and_preprocess()

(wcss, sil_scores, optimal_k,
 kmeans_final, labels_kmeans, score_kmeans,
 labels_agg, score_agg) = compute_clustering(pca_data)


# ══════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════

page = render_sidebar(df_cleaned, df_encoded, score_kmeans, score_agg)


# ══════════════════════════════════════════════
# PAGE ROUTING
# ══════════════════════════════════════════════
# Each page module exposes a render() function that draws
# its content. We pass only the data each page needs.

if page == "🏠 Project Overview":
    overview.render(df_raw, df_encoded, pca)

elif page == "📊 Exploratory Analysis":
    eda.render(df_full, df_cleaned)

elif page == "🔬 Clustering Pipeline":
    clustering.render(
        pca, df_encoded, wcss, sil_scores, optimal_k,
        pca_data, labels_kmeans, score_kmeans, labels_agg, score_agg
    )

elif page == "🎯 Cluster Profiles":
    profiles.render(df_encoded, labels_kmeans, pca_data)

elif page == "🔮 Predict Your Cluster":
    predict.render(df_encoded, labels_kmeans, pca_data, kmeans_final, scaler, pca)
