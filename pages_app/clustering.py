"""
Page 3 — Clustering Pipeline
═════════════════════════════
PCA variance, Elbow method, Silhouette scores, algorithm
comparison, and the 3D interactive scatter plot.

WHY this page?
This is the ML CORE of your project. It shows:
- PCA variance (proves 3 components are enough)
- Elbow method (proves K=4 is optimal)
- Silhouette score (validates cluster quality)
- 3D interactive scatter (the WOW factor)
"""

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from components.ui_helpers import render_metric_card
from config.settings import CHART_TEMPLATE, CHART_FONT, CHART_BG, CLUSTER_COLORS


def render(pca, df_encoded, wcss, sil_scores, optimal_k,
           pca_data, labels_kmeans, score_kmeans, labels_agg, score_agg):
    """Render the Clustering Pipeline page."""

    st.markdown("""
    <div class="hero-header">
        <h1>🔬 Clustering Pipeline</h1>
        <p>From raw features to customer segments — step by step</p>
    </div>
    """, unsafe_allow_html=True)

    # ── PCA Explained Variance ──
    st.markdown('<div class="section-header">📐 PCA — Dimensionality Reduction</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        var_ratios = pca.explained_variance_ratio_
        cumulative = np.cumsum(var_ratios)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["PC1", "PC2", "PC3"],
            y=var_ratios * 100,
            name="Individual",
            marker_color=["#6C63FF", "#3B82F6", "#06B6D4"],
            text=[f"{v:.1f}%" for v in var_ratios * 100],
            textposition="auto"
        ))
        fig.add_trace(go.Scatter(
            x=["PC1", "PC2", "PC3"],
            y=cumulative * 100,
            name="Cumulative",
            mode="lines+markers+text",
            line=dict(color="#F59E0B", width=3),
            marker=dict(size=10),
            text=[f"{v:.1f}%" for v in cumulative * 100],
            textposition="top center"
        ))
        fig.update_layout(
            title="PCA Explained Variance Ratio",
            yaxis_title="Variance Explained (%)",
            template=CHART_TEMPLATE,
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
            font=CHART_FONT,
            height=400,
            legend=dict(orientation="h", y=1.12)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("")
        st.markdown("")
        total_var = sum(var_ratios) * 100
        st.info(f"""
        PCA Summary

        - PC1: {var_ratios[0]*100:.1f}% variance
        - PC2: {var_ratios[1]*100:.1f}% variance
        - PC3: {var_ratios[2]*100:.1f}% variance
        - Total: {total_var:.1f}% captured

        3 components capture over {total_var:.0f}% of the
        original {df_encoded.shape[1]}-feature information.
        """)

    # ── Elbow + Silhouette (Combined Plot) ──
    st.markdown('<div class="section-header">🔍 Finding Optimal K</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, 11)), y=wcss,
            mode="lines+markers",
            name="WCSS (Inertia)",
            line=dict(color="#6C63FF", width=3),
            marker=dict(size=8)
        ))
        # Mark the elbow point
        if optimal_k:
            fig.add_vline(x=optimal_k, line_dash="dash", line_color="#F59E0B",
                          annotation_text=f"Elbow at K={optimal_k}")
        fig.update_layout(
            title="Elbow Method (WCSS vs K)",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="WCSS (Within-Cluster Sum of Squares)",
            template=CHART_TEMPLATE,
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
            font=CHART_FONT,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(2, 11)), y=sil_scores,
            mode="lines+markers",
            name="Silhouette Score",
            line=dict(color="#10B981", width=3),
            marker=dict(size=8)
        ))
        # Mark K=4
        k4_idx = 2  # index for K=4 in range(2,11)
        fig.add_annotation(
            x=4, y=sil_scores[k4_idx],
            text=f"K=4: {sil_scores[k4_idx]:.3f}",
            showarrow=True, arrowhead=2,
            font=dict(color="#F59E0B", size=13)
        )
        fig.update_layout(
            title="Silhouette Score vs K",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Silhouette Score",
            template=CHART_TEMPLATE,
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
            font=CHART_FONT,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Algorithm Comparison ──
    st.markdown('<div class="section-header">⚔️ Algorithm Comparison (K=4)</div>', unsafe_allow_html=True)

    comp_col1, comp_col2, comp_col3 = st.columns([1, 1, 1])
    with comp_col1:
        render_metric_card(f"{score_kmeans:.4f}", "KMeans Silhouette")
    with comp_col2:
        render_metric_card(f"{score_agg:.4f}", "Agglomerative Silhouette")
    with comp_col3:
        winner = "KMeans" if score_kmeans >= score_agg else "Agglomerative"
        render_metric_card(f"🏆 {winner}", "Better Algorithm")

    st.markdown("")

    # ── 3D Clustering Scatter — THE WOW FACTOR ──
    st.markdown('<div class="section-header">🌐 3D Cluster Visualization</div>', unsafe_allow_html=True)

    # WHY Plotly 3D? This is the single most impressive visual.
    # Users can ROTATE, ZOOM, HOVER to explore clusters.
    # matplotlib 3D plots are static images — useless on the web.

    algo_choice = st.radio(
        "Select Algorithm",
        ["KMeans", "Agglomerative"],
        horizontal=True
    )

    labels = labels_kmeans if algo_choice == "KMeans" else labels_agg
    cluster_names = [f"Cluster {l}" for l in labels]

    fig = px.scatter_3d(
        x=pca_data[:, 0], y=pca_data[:, 1], z=pca_data[:, 2],
        color=cluster_names,
        color_discrete_map=CLUSTER_COLORS,
        title=f"{algo_choice} Clustering — 3D PCA Space (drag to rotate!)",
        labels={"x": "PC1", "y": "PC2", "z": "PC3", "color": "Cluster"}
    )
    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=CHART_BG,
        font=CHART_FONT,
        height=650,
        scene=dict(
            xaxis=dict(backgroundcolor=CHART_BG),
            yaxis=dict(backgroundcolor=CHART_BG),
            zaxis=dict(backgroundcolor=CHART_BG)
        )
    )
    fig.update_traces(marker=dict(size=3, opacity=0.8))
    st.plotly_chart(fig, use_container_width=True)
