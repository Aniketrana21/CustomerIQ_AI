"""
Page 4 — Cluster Profiles & Business Insights
═══════════════════════════════════════════════
Cluster distribution, profile cards, spending-vs-income
scatter, radar comparison, and detailed summary table.

WHY this page?
Numbers without context are meaningless. This page translates
cluster statistics into BUSINESS STORIES. That's what impresses
recruiters — it shows you can bridge ML and business.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from components.ui_helpers import get_cluster_profiles
from config.settings import CHART_TEMPLATE, CHART_FONT, CHART_BG


def render(df_encoded, labels_kmeans, pca_data):
    """Render the Cluster Profiles page."""

    st.markdown("""
    <div class="hero-header">
        <h1>🎯 Cluster Profiles & Business Insights</h1>
        <p>Translating data clusters into actionable marketing strategies</p>
    </div>
    """, unsafe_allow_html=True)

    profiles, cluster_summary = get_cluster_profiles(df_encoded, labels_kmeans)

    # ── Cluster Distribution ──
    st.markdown('<div class="section-header">📊 Cluster Distribution</div>', unsafe_allow_html=True)

    cluster_counts = pd.Series(labels_kmeans).value_counts().sort_index().reset_index()
    cluster_counts.columns = ["Cluster", "Count"]
    cluster_counts["Label"] = cluster_counts["Cluster"].map(
        lambda c: profiles[c]["label"]
    )
    cluster_counts["Percentage"] = (cluster_counts["Count"] / cluster_counts["Count"].sum() * 100).round(1)

    fig = px.bar(
        cluster_counts, x="Label", y="Count",
        color="Label",
        color_discrete_sequence=[profiles[i]["color"] for i in sorted(profiles.keys())],
        title="Customer Distribution Across Clusters",
        text="Percentage"
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=CHART_FONT,
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Cluster Profile Cards ──
    st.markdown('<div class="section-header">🧬 Cluster Profiles</div>', unsafe_allow_html=True)

    cols = st.columns(2)
    for idx, (cluster_id, profile) in enumerate(sorted(profiles.items())):
        with cols[idx % 2]:
            count = int(cluster_counts[cluster_counts["Cluster"] == cluster_id]["Count"].values[0])
            pct = float(cluster_counts[cluster_counts["Cluster"] == cluster_id]["Percentage"].values[0])
            st.markdown(f"""
            <div class="cluster-card" style="background: linear-gradient(135deg, {profile['color']}15, {profile['color']}08);">
                <h3>{profile['label']}</h3>
                <p><strong>{count} customers ({pct}%)</strong></p>
                <p>{profile['description']}</p>
                <p>📈 Avg Income: <strong>Rs.{profile['income']:,.0f}</strong></p>
                <p>🛒 Avg Spending: <strong>Rs.{profile['spending']:,.0f}</strong></p>
                <p>⏰ Avg Recency: <strong>{profile['recency']:.0f} days</strong></p>
                <p>👶 Avg Children: <strong>{profile['children']:.1f}</strong></p>
            </div>
            """, unsafe_allow_html=True)

    # ── Spending vs Income Scatter ──
    st.markdown('<div class="section-header">💰 Spending vs Income by Cluster</div>', unsafe_allow_html=True)

    scatter_df = df_encoded.copy()
    scatter_df["Cluster"] = labels_kmeans
    scatter_df["Cluster_Label"] = scatter_df["Cluster"].map(
        lambda c: profiles[c]["label"]
    )

    fig = px.scatter(
        scatter_df, x="Income", y="Total_spending",
        color="Cluster_Label",
        color_discrete_map={profiles[i]["label"]: profiles[i]["color"] for i in profiles},
        title="Income vs Total Spending (colored by cluster)",
        opacity=0.7,
        hover_data=["Recency", "Total_children"]
    )
    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=CHART_FONT,
        height=500,
        xaxis_title="Annual Income (Rs)",
        yaxis_title="Total Spending (Rs)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Radar Chart — Cluster Comparison ──
    st.markdown('<div class="section-header">🕸️ Cluster Radar Comparison</div>', unsafe_allow_html=True)

    # WHY radar chart? It shows MULTIPLE dimensions at once.
    # Each "spoke" is a feature, each cluster is a polygon.
    # You can instantly see which cluster is "big" on what dimension.
    radar_features = ["Income", "Total_spending", "Recency",
                      "Total_children", "NumWebPurchases", "NumStorePurchases"]
    available_radar = [f for f in radar_features if f in cluster_summary.columns]

    if available_radar:
        # Normalize to 0-1 for fair comparison on radar
        radar_data = cluster_summary[available_radar].copy()
        radar_normalized = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min() + 1e-10)

        fig = go.Figure()
        for cluster_id in sorted(radar_normalized.index):
            values = radar_normalized.loc[cluster_id].tolist()
            values.append(values[0])  # Close the polygon
            cats = available_radar + [available_radar[0]]

            fig.add_trace(go.Scatterpolar(
                r=values, theta=cats,
                fill="toself",
                name=profiles[cluster_id]["label"],
                line=dict(color=profiles[cluster_id]["color"]),
                opacity=0.6
            ))

        fig.update_layout(
            title="Cluster Comparison — Normalized Feature Radar",
            polar=dict(
                bgcolor=CHART_BG,
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            template=CHART_TEMPLATE,
            paper_bgcolor=CHART_BG,
            font=CHART_FONT,
            height=550,
            legend=dict(orientation="h", y=-0.1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Cluster Summary Table ──
    st.markdown('<div class="section-header">📋 Detailed Cluster Summary</div>', unsafe_allow_html=True)

    with st.expander("🔽 View full cluster statistics", expanded=False):
        display_summary = cluster_summary.round(2)
        st.dataframe(display_summary, use_container_width=True)
