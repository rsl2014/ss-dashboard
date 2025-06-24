import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ─── Custom CSS Styles ─────────────────────────────────────────────────────────
css = '''
/* Global Styles */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f8fafc;
    color: #333;
}

/* Sidebar Enhancements */
[data-testid="stSidebar"] {
    background-color: #f1f5f9;
    border-right: 1px solid #e2e8f0;
}

/* Improved Tables */
.stDataFrame {
    border-radius: 8px;
    overflow: hidden;
}
.stDataFrame th {
    background-color: #e2e8f0 !important;
    font-weight: 600 !important;
    color: #334155 !important;
}
.stDataFrame td {
    padding: 12px !important;
}
.stDataFrame tr:nth-child(odd) td {
    background-color: #fafafa !important;
}
.stDataFrame tr:hover td {
    background-color: #e2e8f0 !important;
}

/* Metrics Cards */
div[data-testid="metric-container"] {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}

/* Expander Enhancement */
.stExpander {
    border-radius: 8px;
    border: 1px solid #cbd5e1;
}
'''
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
# ─── Metrics Cards ───────────────────────────────────────────────────
st.subheader("📈 Dropout Risk & Metrics")
metric_cols = st.columns(4)

metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metric_values = [accuracy_score(y_test, y_pred),
                 precision_score(y_test, y_pred),
                 recall_score(y_test, y_pred),
                 f1_score(y_test, y_pred)]

for col, label, val in zip(metric_cols, metric_labels, metric_values):
    with col.container(border=True):
        st.metric(label, f"{val:.2%}")


# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Success AI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🎓 Student Success AI Dashboard")
st.markdown("---")

# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

data_path = "data/ss_data.csv"
df = load_data(data_path)
if 'StudentID' not in df.columns:
    df.insert(0, 'StudentID', df.index + 1)

# ─── Sidebar Settings ─────────────────────────────────────────────────────────
st.sidebar.header("🔧 Configuration")
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4)
st.sidebar.markdown("---")

# ─── Preprocessing & Clustering ───────────────────────────────────────────────
df_enc = pd.get_dummies(df.drop(columns=['StudentID']), columns=['Major'], drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_enc)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
df[['PC1', 'PC2']] = pca.fit_transform(X_scaled)

# ─── Descriptive Cluster Labels ───────────────────────────────────────────────
descriptions = [
    "High Achiever: strong GPA & engagement",
    "Moderate Performer: on track academically",
    "At-Risk Academics: low GPA & credits",
    "Financially Strained: holds & low engagement",
    "First-Gen Support Needed: guidance required",
    "Pell Eligible: moderate risk profile",
    "High Withdrawals: tutoring recommended",
    "Low Belonging: social integration concern",
    "Campus Champions: highly engaged",
    "STEM Focused High Performers"
]
cluster_map = {i: descriptions[i] if i < len(descriptions) else f"Cluster {i+1}" for i in range(n_clusters)}
df['ClusterLabel'] = df['Cluster'].map(cluster_map)

# ─── Sidebar Cluster Counts ────────────────────────────────────────────────────
st.sidebar.subheader("Cluster Counts 📊")
st.sidebar.bar_chart(df['ClusterLabel'].value_counts())

# ─── Main Dashboard ────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Cluster Profile Summary 🧩")
    summary = df.groupby('ClusterLabel').mean(numeric_only=True).round(2)
    st.dataframe(summary.style.set_properties(**{'background-color':'#f0f0f0','border':'1px solid #ddd'}))
with col2:
    st.subheader("Clusters PCA Scatter 🎯")
   fig, ax = plt.subplots(figsize=(9, 7))
sns.scatterplot(
    data=df, x='PC1', y='PC2',
    hue='ClusterLabel', palette='Set2', s=100, alpha=0.9, ax=ax
)
ax.set_xlabel("Principal Component 1 (PC1)", fontsize=12)
ax.set_ylabel("Principal Component 2 (PC2)", fontsize=12)
ax.set_title("Student Clusters Visualization", fontsize=14, fontweight='bold')
ax.legend(title='Cluster', loc='best', fontsize='small', title_fontsize='medium', bbox_to_anchor=(1,1))
st.pyplot(fig)

st.markdown("---")

# ─── Interactive Student Explorer ────────────────────────────────────
st.subheader("📌 Student Explorer")
with st.expander("🔍 Filter & Search"):
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        selected_clusters = st.multiselect(
            "Select Clusters:", df['ClusterLabel'].unique(), default=df['ClusterLabel'].unique())
    with col2:
        gpa_range = st.slider("GPA Range:", 0.0, 4.0, (2.0, 4.0), step=0.1)
    with col3:
        search_id = st.text_input("Search Student ID:")

filtered_df = df[df['ClusterLabel'].isin(selected_clusters)]
filtered_df = filtered_df[(filtered_df['GPA'] >= gpa_range[0]) & (filtered_df['GPA'] <= gpa_range[1])]
if search_id:
    filtered_df = filtered_df[filtered_df['StudentID'].astype(str).str.contains(search_id)]

st.dataframe(
    filtered_df[['StudentID', 'ClusterLabel', 'GPA', 'CreditsCompleted', 'CampusEngagementScore']]
    .rename(columns={'ClusterLabel': 'Cluster'})
    .reset_index(drop=True)
)


# ─── Dropout Risk Predictor ───────────────────────────────────────────────────
st.subheader("📈 Dropout Risk & Metrics")
df['DropoutFlag'] = (df['GPA'] < 2.2).astype(int)
model_df = df_enc.copy()
model_df['DropoutFlag'] = df['DropoutFlag']
y = model_df['DropoutFlag']
X = model_df.drop(columns=['DropoutFlag'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]
m1, m2, m3, m4 = st.columns(4)
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred)
}
for col, (name, val) in zip([m1, m2, m3, m4], metrics.items()):
    col.metric(name, f"{val:.2f}")

st.markdown("---")

# ─── Advisor Chatbot Placeholder ───────────────────────────────────────────────
st.subheader("💬 Advisor Chatbot (Coming Soon)")
st.info("This feature is under development and will be available in a future release.")
st.markdown("---")

# ─── Data Download ───────────────────────────────────────────────────────────
st.markdown("### 📥 Download Clustered Data")
st.download_button(
    "⬇️ Download CSV",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name="clustered_students.csv",
    mime="text/csv",
    use_container_width=True,
)


