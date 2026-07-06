"""
SmartCart — Sidebar Navigation
══════════════════════════════
Renders the left sidebar with page links and quick stats.
Returns the selected page label string.
"""

import streamlit as st
from config.settings import PAGE_LABELS


def render_sidebar(df_cleaned, df_encoded, score_kmeans, score_agg):
    """
    Draw sidebar navigation and quick stats.

    Returns
    -------
    page : str — the label of the currently selected page
    """
    with st.sidebar:
        st.markdown("## 🛒 CustomerIQ AI")
        st.markdown("---")

        page = st.radio(
            "Navigate",
            PAGE_LABELS,
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Quick stats in sidebar
        st.markdown("### 📈 Quick Stats")
        st.metric("Customers", f"{len(df_cleaned):,}")
        st.metric("Features", f"{df_encoded.shape[1]}")
        st.metric("Clusters", "4")
        st.metric("Best Silhouette", f"{max(score_kmeans, score_agg):.3f}")

        st.markdown("---")
        st.markdown(
            "<p style='text-align:center; color:#6B7280; font-size:0.8rem;'>"
            "Built with ❤️ by Aniket<br>Powered by Streamlit</p>",
            unsafe_allow_html=True
        )

    return page
