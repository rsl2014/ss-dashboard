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

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Success AI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS Styles ─────────────────────────────────────────────────────────
css = '''
body {
    font-family: 'Inter', sans-serif;
    background-color: #f8fafc;
}
[data-testid="stSidebar"] {
    background-color: #f1f5f9;
    border-right: 1px solid #e2e8f0;
}
.stDataFrame th, .stDataFrame td {
    padding: 0.75rem 1rem !important;
    border-bottom: 1px solid #e2e8f0;
}
.stDataFrame th {
    background-color: #f8fafc !important;
    font-weight: 600 !important;
}
.stDataFrame tr:hover td {
    background-color: #f1f5f9 !important;
}
'''
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

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

# ─── Interactive Student Explorer ────────────────────────────────────
st.subheader("📌 Student Explorer")
with st.expander("🔍 Filter & Search Options"):
    sel = st.multiselect("Clusters", options=df['ClusterLabel'].unique())
    gmin, gmax = st.slider("GPA Range", 0.0, 4.0, (0.0, 4.0), 0.1)
    search_id = st.text_input("Search by Student ID (exact match):")
    table = df.copy()
    if sel:
        table = table[table['ClusterLabel'].isin(sel)]
    table = table[(table['GPA']>=gmin)&(table['GPA']<=gmax)]
    if search_id:
        try:
            sid = int(search_id)
            table = table[table['StudentID'] == sid]
        except ValueError:
            st.error("Please enter a valid integer Student ID.")
    display_cols = ['StudentID','ClusterLabel','GPA','CreditsCompleted','CampusEngagementScore']
    st.dataframe(table[display_cols].rename(columns={'ClusterLabel':'Cluster'}).reset_index(drop=True))
st.markdown("---")

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

metric_cols = st.columns(4)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)]
for col, metric, val in zip(metric_cols, metrics, values):
    col.metric(metric, f"{val:.2%}")

st.markdown("---")

# ─── Advisor FAQ (Dropdown-style) ────────────────────────────────────────────
st.subheader("💬 Advisor FAQ")

faq_options = [
    "What is a high achiever?",
    "How can I help at-risk students?",
    "What resources help first-generation students?",
    "Recommendations for financially strained students",
    "How to increase campus engagement?",
    "What indicates a student may drop out?",
    "How do I support Pell-eligible students?",
    "Benefits of peer mentoring programs",
    "Strategies to retain moderate performers",
    "Importance of sense of belonging",
    "How to handle students with high withdrawals",
    "How can advisors identify first-gen students?",
    "Effective communication strategies with students",
    "Best practices for advising sessions",
    "What interventions help STEM-focused students?",
    "Why track campus engagement scores?",
    "Role of financial counseling in student retention",
    "Tips for increasing academic motivation",
    "How to address low social integration",
    "Indicators of financial stress in students"
]

selected_faq = st.selectbox("Choose a question to see the answer:", faq_options)

faq_answers = {
    faq_options[0]:  "A high achiever typically has a strong GPA, high engagement scores, and is on track academically.",
    faq_options[1]:  "At-risk students benefit from academic tutoring, regular advising appointments, and financial counseling if needed.",
    faq_options[2]:  "First-gen students thrive with dedicated mentoring, specialized workshops, and academic advising tailored to their unique needs.",
    faq_options[3]:  "Connect these students with financial aid counseling, budgeting workshops, and emergency grant options.",
    faq_options[4]:  "Encourage participation in clubs, campus events, peer mentorship programs, and community service projects.",
    faq_options[5]:  "Key indicators include consistently low GPA, high number of course withdrawals, financial holds, and low campus engagement.",
    faq_options[6]:  "Provide targeted financial literacy workshops, ensure timely financial aid support, and encourage participation in engagement activities.",
    faq_options[7]:  "Peer mentoring can improve students' sense of belonging, provide academic support, and increase overall campus engagement.",
    faq_options[8]:  "Moderate performers benefit from proactive academic advising, goal-setting sessions, and encouragement to participate in high-impact practices like internships or research.",
    faq_options[9]:  "A strong sense of belonging is crucial for student retention, academic performance, and overall student success.",
    faq_options[10]: "Students with multiple withdrawals may need academic counseling, tutoring support, and clear discussions about managing course loads.",
    faq_options[11]: "First-gen students can typically be identified through admissions data, student intake forms, or self-reported surveys.",
    faq_options[12]: "Personalized messaging, regular check-ins, and clear communication about resources and opportunities enhance student engagement.",
    faq_options[13]: "Effective advising includes active listening, setting clear goals, creating follow-up action plans, and documenting each interaction.",
    faq_options[14]: "STEM-focused students thrive with specialized tutoring, research opportunities, internship placements, and participation in academic clubs.",
    faq_options[15]: "Campus engagement scores help advisors identify students who might be socially isolated or at risk of dropping out, allowing targeted interventions.",
    faq_options[16]: "Financial counseling provides students with essential budgeting skills, awareness of financial aid resources, and support managing financial stress.",
    faq_options[17]: "Academic motivation can be increased by helping students set achievable goals, celebrating small wins, and connecting coursework to career aspirations.",
    faq_options[18]: "Encourage students to join clubs, attend campus events, utilize peer mentoring programs, and participate in orientation activities to build stronger social connections.",
    faq_options[19]: "Indicators include repeated financial holds, late tuition payments, frequent visits to financial aid offices, and expressing anxiety about finances."
}

st.info(faq_answers[selected_faq])

# ─── Data Download ───────────────────────────────────────────────────────────
st.subheader("📥 Download Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, file_name='clustered_students.csv', mime='text/csv')
