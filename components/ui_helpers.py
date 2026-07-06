"""
SmartCart — Reusable UI Helper Functions
════════════════════════════════════════
Contains CSS injection, metric cards, and cluster profiling
logic used across multiple pages.
"""

import streamlit as st


def inject_css(css_path: str) -> None:
    """
    Load an external CSS file and inject it into the Streamlit app.

    WHY a separate function?
    Keeping CSS in a .css file lets us use syntax highlighting,
    linting, and keeps Python files focused on logic.
    """
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_metric_card(value, label: str) -> None:
    """
    Renders a single styled metric card using HTML.
    Matches the .metric-card class defined in assets/styles.css.
    """
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def get_cluster_profiles(df_encoded, labels):
    """
    Generates business-friendly cluster labels based on cluster means.

    WHY? Raw cluster numbers (0, 1, 2, 3) mean nothing to a recruiter.
    Naming them "Premium Loyalists" tells a STORY — that's what gets
    LinkedIn engagement.

    Returns
    -------
    profiles : dict   — {cluster_id: {label, description, color, ...}}
    summary  : DataFrame — cluster-level means
    """
    df_temp = df_encoded.copy()
    df_temp["Cluster"] = labels
    summary = df_temp.groupby("Cluster").mean()

    profiles = {}
    for cluster_id in sorted(summary.index):
        row = summary.loc[cluster_id]
        income = row.get("Income", 0)
        spending = row.get("Total_spending", 0)
        recency = row.get("Recency", 0)
        children = row.get("Total_children", 0)

        # Rule-based labeling using cluster characteristics
        if spending > summary["Total_spending"].median() and income > summary["Income"].median():
            label = "💎 Premium Loyalists"
            desc = "High income, high spending customers. Your VIPs — invest in retention & exclusive offers."
            color = "#6C63FF"
        elif children > summary["Total_children"].median() and spending <= summary["Total_spending"].median():
            label = "👨‍👩‍👧‍👦 Budget Families"
            desc = "Family-oriented, price-sensitive shoppers. Target with bundles, deals & family packs."
            color = "#06B6D4"
        elif recency > summary["Recency"].median() and spending <= summary["Total_spending"].median():
            label = "😴 Dormant Customers"
            desc = "Haven't purchased recently. Need re-engagement campaigns, win-back offers."
            color = "#EF4444"
        else:
            label = "🌟 Rising Spenders"
            desc = "Mid-range customers with growth potential. Upsell opportunities & loyalty programs."
            color = "#10B981"

        profiles[cluster_id] = {
            "label": label, "description": desc, "color": color,
            "income": income, "spending": spending,
            "recency": recency, "children": children
        }

    return profiles, summary
