import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

from data_loader import generate_dataset, load_dataset
from preprocessor import preprocess
from models import train_and_evaluate
from exporter import export_for_powerbi

st.set_page_config(page_title="Smart Campus Analyzer", page_icon="🎓", layout="wide")

st.sidebar.title("🎓 Smart Campus Analyzer")
st.sidebar.markdown("---")
n_students = st.sidebar.slider("Number of Students", 100, 1000, 500, step=50)
run_btn = st.sidebar.button("▶ Run Analysis", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.caption("Powered by Scikit-learn + Streamlit")

st.title("🎓 Smart Campus Analyzer")
st.markdown("Identify at-risk students and support data-driven decisions.")

if not run_btn:
    st.info("👈 Set the number of students and click **Run Analysis** to start.")
    st.stop()

with st.spinner("Running ML pipeline..."):
    generate_dataset(n=n_students)
    df = load_dataset("dataset.csv")
    df_clean, X_scaled, y, feature_cols = preprocess(df)
    results, winner = train_and_evaluate(X_scaled, y)
    export_df = export_for_powerbi(df_clean, X_scaled, results["lr"], results["rf"], feature_cols)

st.success(f"Analysis complete. Best Model: **{winner}**")

# KPI Cards
st.markdown("### 📊 Key Metrics")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Students", len(df_clean))
c2.metric("At-Risk", int(df_clean["at_risk"].sum()))
c3.metric("Safe", int((df_clean["at_risk"] == 0).sum()))
c4.metric("RF Accuracy", f"{results['rf']['accuracy']:.2%}")
c5.metric("LR Accuracy", f"{results['lr']['accuracy']:.2%}")

st.markdown("---")

# Model Comparison
st.markdown("### 🤖 Model Comparison")
col_a, col_b = st.columns(2)
with col_a:
    fig, ax = plt.subplots(figsize=(5, 3))
    vals = [results["lr"]["accuracy"], results["rf"]["accuracy"]]
    bars = ax.bar(["Logistic Regression", "Random Forest"], vals, color=["#9b59b6", "#1abc9c"])
    ax.set_ylim(0, 1.1); ax.set_title("Accuracy")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.4f}", ha="center", fontsize=9)
    st.pyplot(fig); plt.close()

with col_b:
    fig, ax = plt.subplots(figsize=(5, 3))
    vals = [results["lr"]["auc"], results["rf"]["auc"]]
    bars = ax.bar(["Logistic Regression", "Random Forest"], vals, color=["#e74c3c", "#3498db"])
    ax.set_ylim(0, 1.1); ax.set_title("AUC-ROC")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.4f}", ha="center", fontsize=9)
    st.pyplot(fig); plt.close()

# Confusion Matrices
st.markdown("### 🔢 Confusion Matrices")
col_c, col_d = st.columns(2)
for col, cm, title in zip([col_c, col_d],
                           [results["lr"]["confusion"], results["rf"]["confusion"]],
                           ["Logistic Regression", "Random Forest"]):
    with col:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Safe", "At-Risk"], yticklabels=["Safe", "At-Risk"])
        ax.set_title(title); ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
        st.pyplot(fig); plt.close()

st.markdown("---")

# Charts
st.markdown("### 📈 Visualizations")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    counts = df_clean["at_risk"].value_counts()
    ax.bar(["Safe", "At-Risk"], [counts.get(0,0), counts.get(1,0)], color=["#2ecc71","#e74c3c"])
    ax.set_title("Risk Distribution")
    for i, v in enumerate([counts.get(0,0), counts.get(1,0)]):
        ax.text(i, v+1, str(v), ha="center", fontweight="bold")
    st.pyplot(fig); plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.regplot(data=df_clean, x="study_hours_per_day", y="marks", ax=ax,
                scatter_kws={"alpha":0.4,"color":"#3498db"}, line_kws={"color":"#e67e22"})
    ax.set_title("Study Hours vs Marks")
    st.pyplot(fig); plt.close()

col3, col4 = st.columns(2)
with col3:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = df_clean["at_risk"].map({0:"#2ecc71", 1:"#e74c3c"})
    ax.scatter(df_clean["attendance_pct"], df_clean["marks"], c=colors, alpha=0.5)
    ax.set_xlabel("Attendance (%)"); ax.set_ylabel("Marks"); ax.set_title("Attendance vs Marks")
    ax.legend(handles=[Patch(color="#2ecc71",label="Safe"), Patch(color="#e74c3c",label="At-Risk")])
    st.pyplot(fig); plt.close()

with col4:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    imp = results["rf"]["model"].feature_importances_
    idx = np.argsort(imp)
    ax.barh([feature_cols[i] for i in idx], imp[idx], color="#2980b9")
    ax.set_title("RF Feature Importances")
    st.pyplot(fig); plt.close()

col5, col6 = st.columns(2)
with col5:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    risk_income = df_clean.groupby("family_income")["at_risk"].mean().sort_values()
    risk_income.plot(kind="bar", ax=ax, color=["#27ae60","#f39c12","#c0392b"])
    ax.set_title("Risk Rate by Family Income"); ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for i, v in enumerate(risk_income):
        ax.text(i, v+0.005, f"{v:.1%}", ha="center", fontsize=9)
    st.pyplot(fig); plt.close()

with col6:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for label, grp in df_clean.groupby("at_risk"):
        ax.hist(grp["attendance_pct"], bins=20, alpha=0.6,
                label="At-Risk" if label==1 else "Safe",
                color="#e74c3c" if label==1 else "#2ecc71")
    ax.set_title("Attendance Distribution by Risk"); ax.set_xlabel("Attendance (%)"); ax.legend()
    st.pyplot(fig); plt.close()

st.markdown("---")

# Insights
st.markdown("### 💡 Insights")
low_att = df_clean[df_clean["attendance_pct"] < 60]
high_att = df_clean[df_clean["attendance_pct"] >= 60]
corr = df_clean["study_hours_per_day"].corr(df_clean["marks"])
i1, i2, i3 = st.columns(3)
i1.info(f"Low Attendance (<60%) Risk Rate: **{low_att['at_risk'].mean():.1%}**")
i2.info(f"Study Hours vs Marks Correlation: **{corr:.4f}** ({'Strong' if corr>0.5 else 'Moderate'})")
i3.info(f"High Attendance (≥60%) Risk Rate: **{high_att['at_risk'].mean():.1%}**")

# Recommendations
st.markdown("### ✅ Recommendations")
recs = []
if (df_clean["attendance_pct"] < 60).mean() > 0.2:
    recs.append(f"📌 {(df_clean['attendance_pct']<60).mean():.1%} students below 60% attendance — launch attendance drives.")
if (df_clean["study_hours_per_day"] < 2).mean() > 0.15:
    recs.append(f"📌 {(df_clean['study_hours_per_day']<2).mean():.1%} students study <2 hrs/day — introduce study support programs.")
if (df_clean["assignments_completed"] < 5).mean() > 0.2:
    recs.append(f"📌 {(df_clean['assignments_completed']<5).mean():.1%} students completed <5 assignments — enforce submission policies.")
low_inc = df_clean[df_clean["family_income"]=="Low"]["at_risk"].mean()
if low_inc > 0.3:
    recs.append(f"📌 Low-income students: {low_inc:.1%} risk rate — consider financial aid.")
if not recs:
    recs.append("✅ Campus performance looks healthy.")
for r in recs:
    st.warning(r)

st.markdown("---")

# Data Table + Download
st.markdown("### 📋 Power BI Export Preview")
st.dataframe(export_df.head(50), use_container_width=True)
csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇ Download Power BI CSV", data=csv,
                   file_name="powerbi_campus_data.csv", mime="text/csv", use_container_width=True)
