"""
Page 5 — Predict Your Cluster
══════════════════════════════
Interactive prediction: user enters demographics via
sliders/dropdowns, and the fitted KMeans model assigns
them to a cluster with visual feedback.

WHY this page?
THIS IS THE LINKEDIN WOW-FACTOR. Recruiters can input their own
data and see which cluster they'd fall into. Interactive demos
get 10x more engagement than static screenshots.

HOW IT WORKS:
1. User fills in their demographics via sliders/dropdowns
2. We apply the SAME preprocessing pipeline as training
3. We use the fitted scaler → PCA → KMeans.predict()
4. We show which cluster they belong to + interpretation
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from components.ui_helpers import get_cluster_profiles
from config.settings import CHART_TEMPLATE, CHART_FONT, CHART_BG, CLUSTER_COLORS


def render(df_encoded, labels_kmeans, pca_data, kmeans_final, scaler, pca):
    """Render the Predict Your Cluster page."""

    st.markdown("""
    <div class="hero-header">
        <h1>🔮 Predict Your Cluster</h1>
        <p>Enter your profile and see which customer segment you belong to!</p>
    </div>
    """, unsafe_allow_html=True)

    profiles, _ = get_cluster_profiles(df_encoded, labels_kmeans)

    st.markdown('<div class="section-header">📝 Enter Your Profile</div>', unsafe_allow_html=True)

    # ── Input form with sliders and dropdowns ──
    # WHY st.columns? Arranges inputs side-by-side for a compact layout.
    col1, col2, col3 = st.columns(3)

    with col1:
        income = st.slider("💰 Annual Income (Rs)", 1000, 200000, 50000, step=1000)
        age = st.slider("🎂 Age", 18, 85, 35)
        education = st.selectbox("🎓 Education", ["UnderGraduate", "Graduate", "PostGraduate"])

    with col2:
        total_spending = st.slider("🛒 Total Spending (Rs)", 0, 25000, 500, step=500)
        recency = st.slider("⏰ Days Since Last Purchase", 0, 100, 30)
        living = st.selectbox("🏠 Living With", ["Alone", "Partner"])

    with col3:
        total_children = st.slider("👶 Total Children", 0, 5, 1)
        num_deals = st.slider("🏷️ Deal Purchases", 0, 15, 3)
        num_web = st.slider("🌐 Web Purchases", 0, 20, 5)
        num_catalog = st.slider("📖 Catalog Purchases", 0, 15, 2)
        num_store = st.slider("🏪 Store Purchases", 0, 15, 5)
        web_visits = st.slider("👀 Web Visits/Month", 0, 20, 5)

    st.markdown("")

    if st.button("🚀 Predict My Cluster", type="primary", use_container_width=True):

        # ── Build the input row ──
        # WHY this exact column order? It MUST match df_encoded.columns.
        # If the order is wrong, the scaler/PCA will misinterpret the values.

        # Start with numeric features (same order as df_encoded)
        input_data = {}

        # Map the column names from df_encoded to user inputs
        for col in df_encoded.columns:
            if col == "Income":
                input_data[col] = income
            elif col == "Recency":
                input_data[col] = recency
            elif col == "NumDealsPurchases":
                input_data[col] = num_deals
            elif col == "NumWebPurchases":
                input_data[col] = num_web
            elif col == "NumCatalogPurchases":
                input_data[col] = num_catalog
            elif col == "NumStorePurchases":
                input_data[col] = num_store
            elif col == "NumWebVisitsMonth":
                input_data[col] = web_visits
            elif col == "Complain":
                input_data[col] = 0
            elif col == "Response":
                input_data[col] = 0
            elif col == "Age":
                input_data[col] = age
            elif col == "Customer_tenure_date":
                input_data[col] = 365  # Default: 1 year customer
            elif col == "Total_spending":
                input_data[col] = total_spending
            elif col == "Total_children":
                input_data[col] = total_children
            elif col == "Education_Graduate":
                input_data[col] = 1.0 if education == "Graduate" else 0.0
            elif col == "Education_PostGraduate":
                input_data[col] = 1.0 if education == "PostGraduate" else 0.0
            elif col == "Education_UnderGraduate":
                input_data[col] = 1.0 if education == "UnderGraduate" else 0.0
            elif col == "Living_with_Alone":
                input_data[col] = 1.0 if living == "Alone" else 0.0
            elif col == "Living_with_Partner":
                input_data[col] = 1.0 if living == "Partner" else 0.0
            else:
                input_data[col] = 0  # Default for any unexpected column

        input_df = pd.DataFrame([input_data])

        # ── Apply the SAME pipeline as training ──
        # CRITICAL: Must use the SAME scaler and PCA that were fitted on training data.
        # If you fit a NEW scaler, the numbers would be on a different scale → wrong prediction.
        input_scaled = scaler.transform(input_df)
        input_pca = pca.transform(input_scaled)

        # ── Predict cluster ──
        # WHY KMeans for prediction?
        # KMeans has a .predict() method (assigns to nearest centroid).
        # AgglomerativeClustering does NOT — it can only fit, not predict new data.
        predicted_cluster = kmeans_final.predict(input_pca)[0]
        profile = profiles[predicted_cluster]

        # ── Display result with animation ──
        st.markdown("")
        st.markdown("---")
        st.markdown("")

        result_col1, result_col2 = st.columns([1, 2])

        with result_col1:
            st.markdown(f"""
            <div class="metric-card" style="border-color: {profile['color']}; border-width: 2px;">
                <div class="metric-value" style="font-size: 3rem;">
                    {profile['label'].split(' ')[0]}
                </div>
                <div class="metric-label" style="font-size: 1.1rem; color: {profile['color']};">
                    {profile['label']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with result_col2:
            st.success(f"### You belong to: {profile['label']}")
            st.markdown(f"**{profile['description']}**")

            st.markdown(f"""
            **Your profile vs. cluster averages:**
            - 💰 Your Income: **${income:,} (Cluster avg: ${profile['income']:,.0f}**)
            - 🛒 Your Spending: **${total_spending:,} (Cluster avg: ${profile['spending']:,.0f}**)
            - ⏰ Your Recency: **{recency} days** (Cluster avg: **{profile['recency']:.0f} days**)
            - 👶 Your Children: **{total_children}** (Cluster avg: **{profile['children']:.1f}**)
            """)

        # ── Show where the user falls in 3D space ──
        st.markdown('<div class="section-header">🌐 Your Position in Cluster Space</div>', unsafe_allow_html=True)

        cluster_names_list = [f"Cluster {l}" for l in labels_kmeans]
        color_map_pred = {
            **CLUSTER_COLORS,
            "⭐ YOU": "#EF4444"
        }

        fig = px.scatter_3d(
            x=pca_data[:, 0], y=pca_data[:, 1], z=pca_data[:, 2],
            color=cluster_names_list,
            color_discrete_map=color_map_pred,
            opacity=0.4,
            labels={"x": "PC1", "y": "PC2", "z": "PC3", "color": "Cluster"}
        )

        # Add the user's point as a large red star
        fig.add_trace(go.Scatter3d(
            x=[input_pca[0, 0]], y=[input_pca[0, 1]], z=[input_pca[0, 2]],
            mode="markers+text",
            marker=dict(size=12, color="#EF4444", symbol="diamond"),
            text=["⭐ YOU"],
            textposition="top center",
            name="⭐ YOU",
            showlegend=True
        ))

        fig.update_layout(
            title="Your Position Among All Customers (red diamond = you)",
            template=CHART_TEMPLATE,
            paper_bgcolor=CHART_BG,
            font=CHART_FONT,
            height=600,
            scene=dict(
                xaxis=dict(backgroundcolor=CHART_BG),
                yaxis=dict(backgroundcolor=CHART_BG),
                zaxis=dict(backgroundcolor=CHART_BG)
            )
        )
        fig.update_traces(marker=dict(size=3), selector=dict(mode="markers"))
        st.plotly_chart(fig, use_container_width=True)
