"""
Page 2 — Exploratory Data Analysis
════════════════════════════════════
Income & age distributions, spending by category,
correlation heatmap, and demographic breakdowns.

WHY this page?
EDA shows you understand the data BEFORE applying ML.
Recruiters look for this — it proves analytical thinking.
"""

import streamlit as st
import plotly.express as px
from config.settings import CHART_TEMPLATE, CHART_FONT, CHART_BG


def render(df_full, df_cleaned):
    """Render the Exploratory Data Analysis page."""

    st.markdown("""
    <div class="hero-header">
        <h1>📊 Exploratory Data Analysis</h1>
        <p>Understanding the data before applying machine learning</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Income Distribution ──
    st.markdown('<div class="section-header">💰 Income Distribution</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # WHY Plotly instead of matplotlib?
        # Plotly charts are INTERACTIVE — hover to see exact values,
        # zoom in, pan around. Way better for a web dashboard.
        fig = px.histogram(
            df_full, x="Income", nbins=50,
            color_discrete_sequence=["#6C63FF"],
            title="Income Distribution (with outlier threshold)"
        )
        fig.add_vline(x=600_000, line_dash="dash", line_color="#EF4444",
                      annotation_text="Outlier Threshold (600K)")
        fig.update_layout(
            template=CHART_TEMPLATE,
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
            font=CHART_FONT,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Age distribution
        fig = px.histogram(
            df_full, x="Age", nbins=40,
            color_discrete_sequence=["#06B6D4"],
            title="Age Distribution (with outlier threshold)"
        )
        fig.add_vline(x=90, line_dash="dash", line_color="#EF4444",
                      annotation_text="Outlier Threshold (90)")
        fig.update_layout(
            template=CHART_TEMPLATE,
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
            font=CHART_FONT,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Spending by Category ──
    st.markdown('<div class="section-header">🛍️ Spending by Category</div>', unsafe_allow_html=True)

    spend_cols = ["MntWines", "MntFruits", "MntMeatProducts",
                  "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    spend_means = df_full[spend_cols].mean().reset_index()
    spend_means.columns = ["Category", "Average Spend"]
    spend_means["Category"] = spend_means["Category"].str.replace("Mnt", "")

    fig = px.bar(
        spend_means, x="Category", y="Average Spend",
        color="Average Spend",
        color_continuous_scale=["#06B6D4", "#6C63FF", "#A855F7"],
        title="Average Spending by Product Category"
    )
    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=CHART_FONT,
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Correlation Heatmap ──
    st.markdown('<div class="section-header">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)

    # WHY a heatmap? It reveals which features are correlated.
    # E.g., Income ↔ MntWines might be strongly positive — rich customers
    # buy more wine. This informs which features matter for clustering.
    corr_cols = ["Income", "Recency", "Total_spending", "Total_children",
                 "Age", "Customer_tenure_date", "NumWebPurchases",
                 "NumStorePurchases", "NumCatalogPurchases", "NumWebVisitsMonth"]
    # Only use columns that exist in df_cleaned
    available_corr_cols = [c for c in corr_cols if c in df_cleaned.columns]
    corr_matrix = df_cleaned[available_corr_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Matrix",
        aspect="auto"
    )
    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=CHART_FONT,
        height=550
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Education & Marital breakdown ──
    st.markdown('<div class="section-header">👥 Customer Demographics</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        edu_counts = df_full["Education"].value_counts().reset_index()
        edu_counts.columns = ["Education", "Count"]
        fig = px.pie(
            edu_counts, names="Education", values="Count",
            title="Education Levels",
            color_discrete_sequence=["#6C63FF", "#3B82F6", "#06B6D4", "#10B981", "#A855F7"],
            hole=0.4
        )
        fig.update_layout(
            template=CHART_TEMPLATE,
            paper_bgcolor=CHART_BG,
            font=CHART_FONT,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        mar_counts = df_full["Living_with"].value_counts().reset_index()
        mar_counts.columns = ["Status", "Count"]
        fig = px.pie(
            mar_counts, names="Status", values="Count",
            title="Living Situation",
            color_discrete_sequence=["#A855F7", "#F59E0B"],
            hole=0.4
        )
        fig.update_layout(
            template=CHART_TEMPLATE,
            paper_bgcolor=CHART_BG,
            font=CHART_FONT,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
