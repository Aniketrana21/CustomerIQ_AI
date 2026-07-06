"""
Page 1 — Project Overview
═════════════════════════
First impression page: hero banner, key metrics, project
description, dataset preview, and feature dictionary.

WHY this page?
First impression matters. A recruiter landing on your app should
immediately understand: What is this? What data? What result?
"""

import streamlit as st
import pandas as pd
from components.ui_helpers import render_metric_card


def render(df_raw, df_encoded, pca):
    """Render the Project Overview page."""

    # Hero banner
    st.markdown("""
    <div class="hero-header">
        <h1>🛒 CustomerIQ AI - Customer Segmentation System</h1>
        <p>Unsupervised ML pipeline that segments 2,240 customers into actionable groups using KMeans & Agglomerative Clustering</p>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("2,240", "Total Customers")
    with col2:
        render_metric_card("22", "Raw Features")
    with col3:
        render_metric_card("4", "Clusters Found")
    with col4:
        variance_pct = f"{sum(pca.explained_variance_ratio_) * 100:.1f}%"
        render_metric_card(variance_pct, "PCA Variance")

    st.markdown("")

    # Project description
    st.markdown('<div class="section-header">📋 About This Project</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("""
        This project applies **unsupervised machine learning** to segment supermarket customers
        into distinct behavioral groups. The pipeline includes:

        1. Data Preprocessing — Missing value imputation, outlier removal, feature engineering
        2. Feature Encoding — One-Hot Encoding for categorical variables
        3. Dimensionality Reduction — PCA (3 components) for visualization & noise reduction
        4. Clustering — KMeans + Agglomerative with Elbow & Silhouette evaluation
        5. Business Interpretation — Cluster profiling with actionable marketing strategies

        Key Insight: Customers naturally fall into 4 distinct segments based on
        spending behavior, income levels, family composition, and purchase channels.
        """)

    with col_right:
        st.markdown("🧰 Tech Stack")
        tech_data = pd.DataFrame({
            "Technology": ["Python", "Pandas", "Scikit-learn", "Plotly", "Streamlit"],
            "Purpose": ["Core Language", "Data Wrangling", "ML Models", "Visualizations", "Web Dashboard"]
        })
        st.dataframe(tech_data, hide_index=True, use_container_width=True)

    # Dataset preview
    st.markdown('<div class="section-header">📦 Dataset Preview</div>', unsafe_allow_html=True)

    with st.expander("🔽 Click to view raw data (first 100 rows)", expanded=False):
        st.dataframe(df_raw.head(100), use_container_width=True, height=400)

    # Feature dictionary
    st.markdown('<div class="section-header">📖 Feature Dictionary</div>', unsafe_allow_html=True)

    features = pd.DataFrame({
        "Feature": ["Income", "Recency", "MntWines/Fruits/Meat/Fish/Sweets/Gold",
                     "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
                     "NumStorePurchases", "NumWebVisitsMonth", "Complain", "Response"],
        "Description": [
            "Customer's yearly household income",
            "Days since last purchase",
            "Amount spent on each product category (last 2 years)",
            "Number of purchases made with a discount",
            "Number of purchases through website",
            "Number of purchases using catalog",
            "Number of purchases in physical stores",
            "Number of visits to company's website per month",
            "1 if customer complained in last 2 years",
            "1 if customer accepted the last campaign offer"
        ],
        "Type": ["Numeric", "Numeric", "Numeric", "Numeric", "Numeric",
                 "Numeric", "Numeric", "Numeric", "Binary", "Binary"]
    })
    st.dataframe(features, hide_index=True, use_container_width=True)
