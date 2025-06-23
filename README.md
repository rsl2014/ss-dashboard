# 🎓 Student Success AI Dashboard

An interactive Streamlit application that enables higher-education advisors to:

* **Cluster student risk profiles** using K-Means and descriptive labels
* **Visualize clusters** in 2D via PCA
* **Filter and explore** individual student records by cluster and GPA
* **Predict dropout risk** with a logistic regression model and view key metrics
* **Identify top at-risk students** for targeted interventions
* **Download enriched data** for further analysis

---

## 🚀 Features

* **Dynamic Clustering:** Adjust number of clusters (2–10) and view descriptive names like “High Achiever” or “Financially Strained.”
* **PCA Visualization:** 2D scatter plot of student clusters for quick insight.
* **Filterable Student Table:** Select clusters & GPA ranges to drill down on individual students.
* **Dropout Risk Predictor:** Logistic regression model with metrics (Accuracy, Precision, Recall, F1) and probability distributions.
* **At-Risk List:** Top 10 students ranked by predicted dropout probability.
* **Data Export:** Download the full clustered dataset as CSV.

---

## 📂 Repository Structure

```
ss-dashboard/
├── .devcontainer/          # Dev Container configuration for VS Code
│   └── devcontainer.json
├── data/                   # Datasets
│   └── ss_data.csv         # Synthetic or real student data
├── streamlit_app.py        # Main Streamlit dashboard script
├── requirements.txt        # Python dependencies
└── README.md               # Project overview and instructions
```

---

## 🛠 Prerequisites

* **Git** and **Python 3.11+** (or use the Dev Container)
* **Streamlit** & required libraries (see `requirements.txt`)

---

## 📥 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ss-dashboard.git
cd ss-dashboard
```

### 2. (Optional) Dev Container

* Open in VS Code with the [Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers) extension
* Click **Reopen in Container** to build the environment automatically

### 3. Local Setup

```bash
python -m venv .venv         # Create virtual environment
env\Scripts\activate      # Windows PowerShell
source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt
```

---

## ▶️ Running the Dashboard

```bash
streamlit run streamlit_app.py
```

* Navigate to **[http://localhost:8501](http://localhost:8501)** or the forwarded Codespaces URL

---

## 🔄 How to Use

1. **Select clusters** and **explore PCA plot**
2. **Filter students** by cluster & GPA in the table
3. **View dropout metrics** and probability distribution
4. **Identify top 10 at-risk** students for targeted outreach
5. **Download** the full CSV for reporting

---

## 🤝 Contributing

1. Fork this repo
2. Create a branch (`git checkout -b feature/awesome`)
3. Commit changes (`git commit -m "Add awesome feature"`)
4. Push to branch (`git push origin feature/awesome`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**. See `LICENSE` for details.

```
```
