import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Page configuration
st.set_page_config(page_title="Student Success AI", layout="wide")
st.title("Student Success AI Clustering Dashboard")

# Load data
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

data_path = "data/ss_data.csv"
df = load_data(data_path)

# Sidebar settings
st.sidebar.header("Settings")
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4)

# Preprocessing: encode and scale
df_encoded = pd.get_dummies(df, columns=["Major"], drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Dimensionality reduction for visualization
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
df['PC1'], df['PC2'] = components[:, 0], components[:, 1]

# Map descriptive labels for 4 clusters
cluster_labels = {
    0: "Financially Strained but Engaged",
    1: "Moderate Risk (Low Belonging)",
    2: "High Performing, Financially At-Risk",
    3: "Low Advising Use, Stable"
}
if n_clusters == 4:
    df['Cluster_Label'] = df['Cluster'].map(cluster_labels)
else:
    df['Cluster_Label'] = df['Cluster'].astype(str)

# Display cluster counts
st.sidebar.subheader("Cluster Counts")
st.sidebar.bar_chart(df['Cluster_Label'].value_counts())

# PCA scatter plot
st.header("PCA Scatter Plot of Student Clusters")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster_Label', palette='Set2', s=60, ax=ax)
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_title("Student Clusters by PCA")
st.pyplot(fig)

# Cluster profile summary
st.subheader("Cluster Profile Summary")
cluster_summary = df.groupby('Cluster_Label').mean(numeric_only=True).round(2)
st.dataframe(cluster_summary)

# Dropout simulation
np.random.seed(42)
df['DroppedOut'] = np.where(
    df['Cluster'].isin([0, 1]),
    np.random.binomial(1, 0.4, len(df)),
    np.random.binomial(1, 0.1, len(df))
)

st.subheader("Dropout Simulation")
dropout_counts = df['DroppedOut'].map({0: 'Stayed', 1: 'DroppedOut'}).value_counts()
st.bar_chart(dropout_counts)

# Download clustered data
st.subheader("Download Clustered Data")
csv_data = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label='Download CSV',
    data=csv_data,
    file_name='Clustered_Student_Success_Data.csv',
    mime='text/csv'
)
