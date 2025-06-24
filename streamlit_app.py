import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='ClusterLabel', palette='Set2', s=80, alpha=0.8, ax=ax)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
    st.pyplot(fig)

st.markdown("---")

# ─── Interactive Table ────────────────────────────────────────────────────────
st.subheader("📌 Student Explorer")
with st.expander("Filter Options"):
    sel = st.multiselect("Clusters", options=df['ClusterLabel'].unique())
    gmin, gmax = st.slider("GPA Range", 0.0, 4.0, (0.0, 4.0), 0.1)
    table = df.copy()
    if sel:
        table = table[table['ClusterLabel'].isin(sel)]
    table = table[(table['GPA']>=gmin)&(table['GPA']<=gmax)]
    st.dataframe(table[['StudentID','ClusterLabel','GPA','CreditsCompleted','CampusEngagementScore']].rename(columns={'ClusterLabel':'Cluster'}))

st.markdown("---")

# ─── Dropout Risk Predictor ───────────────────────────────────────────────────
st.subheader("📈 Dropout Risk & Metrics")
df['DropoutFlag'] = (df['GPA'] < 2.2).astype(int)
model_df = df_enc.copy()
model_df['DropoutFlag'] = df['DropoutFlag']
y = model_df['DropoutFlag']
X = model_df.drop(columns=['DropoutFlag'])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]
m1,m2,m3,m4 = st.columns(4)
metrics = {
    'Accuracy': accuracy_score(y_test,y_pred),
    'Precision': precision_score(y_test,y_pred),
    'Recall': recall_score(y_test,y_pred),
    'F1 Score': f1_score(y_test,y_pred)
}
for col, (name, val) in zip([m1,m2,m3,m4], metrics.items()):
    col.metric(name, f"{val:.2f}")

st.markdown("---")

# ─── Advisor NLP Chatbot ──────────────────────────────────────────────────────
st.subheader("💬 Advisor NLP Chatbot")
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.warning("🔑 Please set your OPENAI_API_KEY in Streamlit secrets or environment to enable the chatbot.")
else:
    openai.api_key = api_key
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role":"system","content":"You are an AI assistant for academic advisors."}]
    # Receive user input
    user_prompt = st.chat_input("Ask about student clusters or retention strategies:")
    if user_prompt:
        st.session_state.chat_history.append({"role":"user","content":user_prompt})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.chat_history
        )
        assistant_msg = response.choices[0].message.content
        st.session_state.chat_history.append({"role":"assistant","content":assistant_msg})
    # Display chat messages
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])

st.markdown("---")

# ─── Data Download ───────────────────────────────────────────────────────────
st.subheader("📥 Download Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, file_name='clustered_students.csv', mime='text/csv')

