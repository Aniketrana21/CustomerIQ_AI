# 🛒 SmartCart Customer Clustering System

An **interactive ML dashboard** that segments supermarket customers into actionable marketing groups using unsupervised learning.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## 🎯 Project Overview

This project applies **KMeans** and **Agglomerative Clustering** to 2,240 customer records to identify 4 distinct behavioral segments:

| Cluster | Label | Description |
|---------|-------|-------------|
| 0 | 💎 Premium Loyalists | High income, high spending — invest in retention |
| 1 | 👨‍👩‍👧‍👦 Budget Families | Family-oriented, price-sensitive — target with bundles |
| 2 | 🌟 Rising Spenders | Mid-range with growth potential — upsell opportunities |
| 3 | 😴 Dormant Customers | Low engagement — need re-engagement campaigns |

---

## 🧰 Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python | Core language |
| Pandas & NumPy | Data wrangling |
| Scikit-learn | KMeans, PCA, StandardScaler, OneHotEncoder |
| Plotly | Interactive visualizations (3D scatter, radar charts) |
| Streamlit | Web dashboard framework |

---

## 📊 ML Pipeline

```
Raw Data (2240 × 22)
    ↓ Missing value imputation (median for Income)
    ↓ Feature engineering (Age, Tenure, Total_spending, Total_children)
    ↓ Categorical encoding (OneHotEncoder)
    ↓ Outlier removal (Age < 90, Income < 600K)
    ↓ Standard Scaling
    ↓ PCA (3 components → ~55% variance)
    ↓ KMeans + Agglomerative Clustering (K=4)
    ↓ Silhouette evaluation → Business interpretation
```

---

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SmartCart_Clustering_System.git
cd SmartCart_Clustering_System

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

---

## 📂 Project Structure

```
SmartCart_Clustering_System/
├── app.py                           # Streamlit dashboard (5 interactive pages)
├── smartcart_customers.csv          # Customer dataset
├── SmartCart_Clustering_System.ipynb # Original Jupyter notebook
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── .streamlit/
    └── config.toml                  # Dark theme configuration
```

---

## 🌐 Deployment on Streamlit Cloud

1. Push this repo to **GitHub** (public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → Select your repo → Set `app.py` as the main file
4. Click **Deploy** — it auto-installs from `requirements.txt`
5. Share the generated URL on LinkedIn!

---

## 📬 Connect

**Built by Aniket** — feel free to connect on [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
