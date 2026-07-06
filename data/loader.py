"""
SmartCart — Data Loading & Preprocessing Pipeline
══════════════════════════════════════════════════
Cached functions for loading the CSV, engineering features,
encoding, scaling, PCA, and running clustering algorithms.

Both functions use @st.cache_data so they execute only once
and are reused across every Streamlit rerun.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

from config.settings import DATA_PATH


# ══════════════════════════════════════════════
# DATA LOADING & PREPROCESSING (CACHED)
# ══════════════════════════════════════════════
# WHY @st.cache_data?
# Without caching, Streamlit would reload the CSV and re-run ALL
# preprocessing on EVERY user interaction (slider move, button click).
# @st.cache_data stores the result in memory after the first run.
# Subsequent reruns skip straight to the cached result → FAST.
#
# WHY return multiple things?
# We need both the raw df (for EDA) and the processed data (for ML).

@st.cache_data
def load_and_preprocess():
    """
    Full preprocessing pipeline:
    1. Load CSV
    2. Impute missing values (median)
    3. Engineer features (Age, Tenure, Total_spending, Total_children)
    4. Simplify categoricals
    5. Drop unused columns
    6. Remove outliers
    7. One-Hot Encode
    8. Standard Scale
    9. PCA (3 components)

    Returns
    -------
    df_raw, df_full, df_cleaned, df_encoded, X_scaled, pca_data, pca, scaler, ohe
    """

    # ── Step 1: Load raw data ──
    df = pd.read_csv(DATA_PATH)
    df_raw = df.copy()  # Keep a copy for EDA display

    # ── Step 2: Handle missing values ──
    # WHY median? Income is right-skewed (few people earn a lot).
    # Mean would be pulled up by those outliers. Median is robust.
    df["Income"] = df["Income"].fillna(df["Income"].median())

    # ── Step 3: Feature engineering ──
    # WHY create new features? Raw features like Year_Birth aren't
    # directly useful. Age is more interpretable and ML-friendly.
    df["Age"] = 2026 - df["Year_Birth"]

    # Customer tenure: how long they've been a customer (in days)
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    reference_date = df["Dt_Customer"].max()
    df["Customer_tenure_date"] = (reference_date - df["Dt_Customer"]).dt.days

    # Total spending across all categories
    df["Total_spending"] = (
        df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"]
        + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    )

    # Total children (kids + teens)
    df["Total_children"] = df["Kidhome"] + df["Teenhome"]

    # ── Step 4: Simplify categorical features ──
    # WHY? Too many categories = too many one-hot columns = curse of dimensionality.
    # Grouping "PhD" and "Master" as "PostGraduate" reduces noise.
    df["Education"] = df["Education"].replace({
        "2n Cycle": "UnderGraduate", "Basic": "UnderGraduate",
        "Graduation": "Graduate",
        "PhD": "PostGraduate", "Master": "PostGraduate"
    })

    df["Living_with"] = df["Marital_Status"].replace({
        "Married": "Partner", "Together": "Partner",
        "Single": "Alone", "Divorced": "Alone",
        "Widow": "Alone", "Absurd": "Alone", "YOLO": "Alone"
    })

    # ── Step 5: Drop columns not useful for clustering ──
    # WHY drop these? ID is just an identifier, Year_Birth is replaced by Age,
    # individual spending columns are replaced by Total_spending, etc.
    cols_to_drop = [
        "ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome",
        "Dt_Customer", "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]
    df_cleaned = df.drop(columns=cols_to_drop)

    # ── Step 6: Remove outliers ──
    # WHY? Extreme values distort clustering. A 130-year-old customer
    # is clearly a data error. Income > 600K is also suspicious.
    df_cleaned = df_cleaned[df_cleaned["Age"] < 90]
    df_cleaned = df_cleaned[df_cleaned["Income"] < 600_000]

    # ── Step 7: One-Hot Encoding for categorical features ──
    # WHY? ML algorithms need numbers, not strings.
    # OneHotEncoder creates binary columns: Education_Graduate = 1/0
    cat_cols = ["Education", "Living_with"]
    ohe = OneHotEncoder(sparse_output=False)  # sparse_output=False gives us a regular array
    encoded_array = ohe.fit_transform(df_cleaned[cat_cols])
    enc_df = pd.DataFrame(
        encoded_array,
        columns=ohe.get_feature_names_out(cat_cols),
        index=df_cleaned.index
    )
    df_encoded = pd.concat([df_cleaned.drop(columns=cat_cols), enc_df], axis=1)

    # ── Step 8: Standard Scaling ──
    # WHY? Features have different scales (Income: ~50K, Recency: ~50).
    # KMeans uses Euclidean distance — large-scale features would dominate.
    # StandardScaler: mean=0, std=1 for all features → fair distance calculation.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)

    # ── Step 9: PCA (dimensionality reduction) ──
    # WHY 3 components? We have ~16 features after encoding. Plotting 16D
    # is impossible. PCA finds the 3 directions of maximum variance.
    # These 3 PCs capture ~55-60% of total information (check variance ratio).
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(X_scaled)

    return df_raw, df, df_cleaned, df_encoded, X_scaled, pca_data, pca, scaler, ohe


# ══════════════════════════════════════════════
# CLUSTERING (CACHED)
# ══════════════════════════════════════════════

@st.cache_data
def compute_clustering(_pca_data):
    """
    Runs KMeans and Agglomerative clustering + evaluation metrics.

    WHY separate function?
    Clustering is computationally expensive. Caching it means we
    only compute once, even if the user switches between pages.

    NOTE: The underscore prefix in _pca_data tells Streamlit
    "don't try to hash this numpy array" (avoids hashing errors).
    """
    # ── Elbow Method: find optimal K ──
    # WHY? We try K=1 to K=10 and measure WCSS (Within-Cluster Sum of Squares).
    # The "elbow" point is where adding more clusters stops helping much.
    wcss = []
    silhouette_scores = []
    K_range = range(2, 11)

    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(_pca_data)
        wcss.append(km.inertia_)

    # ── Silhouette Scores ──
    # WHY? Silhouette measures how well-separated clusters are (-1 to 1).
    # Higher = better. Unlike WCSS, it considers both cohesion AND separation.
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(_pca_data)
        silhouette_scores.append(silhouette_score(_pca_data, labels))

    # ── Find optimal K automatically ──
    knee = KneeLocator(range(1, 11), wcss, curve="convex", direction="decreasing")
    optimal_k = knee.elbow if knee.elbow else 4

    # ── Final clustering with K=4 ──
    # KMeans (used for prediction — it has .predict() method)
    kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels_kmeans = kmeans_final.fit_predict(_pca_data)
    score_kmeans = silhouette_score(_pca_data, labels_kmeans)

    # Agglomerative (used for comparison — often better quality clusters)
    agg_final = AgglomerativeClustering(n_clusters=4, linkage="ward")
    labels_agg = agg_final.fit_predict(_pca_data)
    score_agg = silhouette_score(_pca_data, labels_agg)

    return (wcss, silhouette_scores, optimal_k,
            kmeans_final, labels_kmeans, score_kmeans,
            labels_agg, score_agg)
