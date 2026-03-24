import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

from data_loader import generate_dataset, load_dataset
from preprocessor import preprocess
from models_v2 import train_all_models
from exporter import export_for_powerbi
from auth import verify_admin, verify_student, get_student_data
from time_series import generate_semester_data, get_student_trend
from pdf_report import generate_pdf

st.set_page_config(page_title="Smart Campus Analyzer", page_icon="🎓", layout="wide")

# ── Auto-generate dataset on first run so student login works ─────────────────
import os
if not os.path.exists("dataset.csv"):
    from data_loader import generate_dataset
    generate_dataset(n=1000)

# ── Session State ─────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.student_id = None
    st.session_state.data_ready = False
    st.session_state.export_df = None
    st.session_state.sem_df = None
    st.session_state.results = None
    st.session_state.df_clean = None
    st.session_state.feature_cols = None
    st.session_state.best = None

# ── Login Page ────────────────────────────────────────────────────────────────
def login_page():
    st.title("🎓 Smart Campus Analyzer")
    st.markdown("### Login")
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        role = st.radio("Login as", ["Admin", "Student"], horizontal=True)
        username = st.text_input("Username / Student ID")
        password = st.text_input("Password", type="password")
        if role == "Student":
            st.caption("💡 Password = last 4 characters of your Student ID (e.g. STU0042 → 0042)")
        else:
            st.caption("💡 Admin credentials: admin / admin123")

        if st.button("Login", use_container_width=True):
            if role == "Admin":
                if verify_admin(username, password):
                    st.session_state.logged_in = True
                    st.session_state.role = "admin"
                    st.rerun()
                else:
                    st.error("Invalid admin credentials.")
            else:
                df = load_dataset("dataset.csv") if st.session_state.data_ready else None
                if df is None:
                    st.warning("Dataset not ready. Ask admin to run analysis first.")
                elif verify_student(username, password, df):
                    st.session_state.logged_in = True
                    st.session_state.role = "student"
                    st.session_state.student_id = username
                    st.rerun()
                else:
                    st.error("Invalid Student ID or password.")

# ── Admin Dashboard ───────────────────────────────────────────────────────────
def admin_dashboard():
    st.sidebar.title("🎓 Admin Panel")
    n_students = st.sidebar.slider("Number of Students", 100, 1000, 500, step=50)
    run_btn = st.sidebar.button("▶ Run Analysis", use_container_width=True)
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

    st.title("🎓 Smart Campus Analyzer — Admin Dashboard")

    if run_btn:
        with st.spinner("Running full ML pipeline..."):
            generate_dataset(n=n_students)
            df = load_dataset("dataset.csv")
            df_clean, X_scaled, y, feature_cols = preprocess(df)
            results, best = train_all_models(X_scaled, y)

            # Use best model for export
            best_key = best
            lr_res = results.get("Logistic Regression", results[best_key])
            rf_res = results.get("Random Forest", results[best_key])
            export_df = export_for_powerbi(df_clean, X_scaled, lr_res, rf_res, feature_cols)
            sem_df = generate_semester_data(df_clean)

            st.session_state.data_ready   = True
            st.session_state.export_df    = export_df
            st.session_state.sem_df       = sem_df
            st.session_state.results      = results
            st.session_state.df_clean     = df_clean
            st.session_state.feature_cols = feature_cols
            st.session_state.best         = best

        st.success(f"Analysis complete. Best Model: **{best}**")

    if not st.session_state.data_ready:
        st.info("👈 Click **Run Analysis** to start.")
        return

    df_clean     = st.session_state.df_clean
    results      = st.session_state.results
    export_df    = st.session_state.export_df
    feature_cols = st.session_state.feature_cols
    sem_df       = st.session_state.sem_df
    best         = st.session_state.best

    # KPI
    st.markdown("### 📊 Key Metrics")
    cols = st.columns(5)
    cols[0].metric("Total Students", len(df_clean))
    cols[1].metric("At-Risk", int(df_clean["at_risk"].sum()))
    cols[2].metric("Safe", int((df_clean["at_risk"]==0).sum()))
    cols[3].metric("Best Model", best)
    cols[4].metric("Best Accuracy", f"{results[best]['accuracy']:.2%}")

    st.markdown("---")

    # Model Comparison
    st.markdown("### 🤖 All Models Comparison")
    model_names = [k for k in results if k not in ("X_test","y_test","X_train","y_train")]
    accs  = [results[m]["accuracy"] for m in model_names]
    aucs  = [results[m]["auc"]      for m in model_names]

    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(model_names, accs, color=["#9b59b6","#1abc9c","#e67e22","#e74c3c","#3498db"])
        ax.set_ylim(0, 1.15); ax.set_title("Accuracy"); ax.set_xticklabels(model_names, rotation=15, ha="right")
        for bar, v in zip(bars, accs):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontsize=8)
        st.pyplot(fig); plt.close()

    with col_b:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(model_names, aucs, color=["#9b59b6","#1abc9c","#e67e22","#e74c3c","#3498db"])
        ax.set_ylim(0, 1.15); ax.set_title("AUC-ROC"); ax.set_xticklabels(model_names, rotation=15, ha="right")
        for bar, v in zip(bars, aucs):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontsize=8)
        st.pyplot(fig); plt.close()

    # Confusion matrices
    st.markdown("### 🔢 Confusion Matrices")
    cm_cols = st.columns(len(model_names))
    for col, mname in zip(cm_cols, model_names):
        with col:
            fig, ax = plt.subplots(figsize=(3, 2.5))
            sns.heatmap(results[mname]["confusion"], annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Safe","At-Risk"], yticklabels=["Safe","At-Risk"])
            ax.set_title(mname, fontsize=8); ax.set_ylabel("Actual", fontsize=7); ax.set_xlabel("Predicted", fontsize=7)
            st.pyplot(fig); plt.close()

    st.markdown("---")

    # Charts
    st.markdown("### 📈 Visualizations")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts = df_clean["at_risk"].value_counts()
        ax.bar(["Safe","At-Risk"], [counts.get(0,0), counts.get(1,0)], color=["#2ecc71","#e74c3c"])
        ax.set_title("Risk Distribution")
        for i, v in enumerate([counts.get(0,0), counts.get(1,0)]):
            ax.text(i, v+1, str(v), ha="center", fontweight="bold")
        st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.regplot(data=df_clean, x="study_hours_per_day", y="marks", ax=ax,
                    scatter_kws={"alpha":0.4,"color":"#3498db"}, line_kws={"color":"#e67e22"})
        ax.set_title("Study Hours vs Marks")
        st.pyplot(fig); plt.close()

    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        clrs = df_clean["at_risk"].map({0:"#2ecc71", 1:"#e74c3c"})
        ax.scatter(df_clean["attendance_pct"], df_clean["marks"], c=clrs, alpha=0.5)
        ax.set_xlabel("Attendance (%)"); ax.set_ylabel("Marks"); ax.set_title("Attendance vs Marks")
        ax.legend(handles=[Patch(color="#2ecc71",label="Safe"), Patch(color="#e74c3c",label="At-Risk")])
        st.pyplot(fig); plt.close()

    with c4:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        rf_res = results.get("Random Forest", results[best])
        imp = rf_res["model"].feature_importances_
        idx = np.argsort(imp)
        ax.barh([feature_cols[i] for i in idx], imp[idx], color="#2980b9")
        ax.set_title("Feature Importances (RF)")
        st.pyplot(fig); plt.close()

    c5, c6 = st.columns(2)
    with c5:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ri = df_clean.groupby("family_income")["at_risk"].mean().sort_values()
        ri.plot(kind="bar", ax=ax, color=["#27ae60","#f39c12","#c0392b"])
        ax.set_title("Risk Rate by Family Income"); ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        for i, v in enumerate(ri): ax.text(i, v+0.005, f"{v:.1%}", ha="center", fontsize=9)
        st.pyplot(fig); plt.close()

    with c6:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for label, grp in df_clean.groupby("at_risk"):
            ax.hist(grp["attendance_pct"], bins=20, alpha=0.6,
                    label="At-Risk" if label==1 else "Safe",
                    color="#e74c3c" if label==1 else "#2ecc71")
        ax.set_title("Attendance Distribution"); ax.set_xlabel("Attendance (%)"); ax.legend()
        st.pyplot(fig); plt.close()

    st.markdown("---")

    # Time Series
    st.markdown("### 📅 Semester Trend (Sample Students)")
    sample_ids = df_clean["student_id"].head(5).tolist()
    selected_id = st.selectbox("Select Student for Trend", sample_ids)
    trend = get_student_trend(selected_id, sem_df)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, col, color, title in zip(axes,
                                      ["marks","attendance","study_hours"],
                                      ["#3498db","#2ecc71","#e67e22"],
                                      ["Marks","Attendance (%)","Study Hours"]):
        ax.plot(trend["semester"], trend[col], marker="o", color=color, linewidth=2)
        ax.set_title(title); ax.set_ylim(0, None)
        for x, y in zip(trend["semester"], trend[col]):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0,6), fontsize=8, ha="center")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")

    # Insights
    st.markdown("### 💡 Insights")
    low_att  = df_clean[df_clean["attendance_pct"] < 60]
    high_att = df_clean[df_clean["attendance_pct"] >= 60]
    corr = df_clean["study_hours_per_day"].corr(df_clean["marks"])
    i1, i2, i3 = st.columns(3)
    i1.info(f"Low Attendance Risk Rate: **{low_att['at_risk'].mean():.1%}**")
    i2.info(f"Study Hours vs Marks Correlation: **{corr:.4f}**")
    i3.info(f"High Attendance Risk Rate: **{high_att['at_risk'].mean():.1%}**")

    # Recommendations
    st.markdown("### ✅ Recommendations")
    if (df_clean["attendance_pct"] < 60).mean() > 0.2:
        st.warning(f"📌 {(df_clean['attendance_pct']<60).mean():.1%} students below 60% attendance.")
    if (df_clean["study_hours_per_day"] < 2).mean() > 0.15:
        st.warning(f"📌 {(df_clean['study_hours_per_day']<2).mean():.1%} students study <2 hrs/day.")
    if (df_clean["assignments_completed"] < 5).mean() > 0.2:
        st.warning(f"📌 {(df_clean['assignments_completed']<5).mean():.1%} students completed <5 assignments.")

    st.markdown("---")

    # Data Table + Download
    st.markdown("### 📋 Power BI Export")
    st.dataframe(export_df.head(50), use_container_width=True)
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Power BI CSV", data=csv,
                       file_name="powerbi_campus_data.csv", mime="text/csv",
                       use_container_width=True)

# ── Student Dashboard ─────────────────────────────────────────────────────────
def student_dashboard():
    student_id = st.session_state.student_id
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

    st.title(f"🎓 Student Portal — {student_id}")

    if not st.session_state.data_ready or st.session_state.export_df is None:
        st.warning("Analysis not run yet. Please ask your administrator to run the analysis first.")
        return

    export_df = st.session_state.export_df
    sem_df    = st.session_state.sem_df
    student   = export_df[export_df["student_id"] == student_id]

    if student.empty:
        st.error("Student data not found.")
        return

    row = student.iloc[0]

    # Risk badge
    risk_color = "#e74c3c" if row["at_risk"] == 1 else "#2ecc71"
    st.markdown(
        f'<div style="background:{risk_color};padding:12px;border-radius:8px;color:white;'
        f'font-size:18px;font-weight:bold;text-align:center;">'
        f'Risk Status: {row["risk_label"]} — {row["risk_tier"]}</div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Attendance",    f"{row['attendance_pct']}%")
    c2.metric("Marks",         row["marks"])
    c3.metric("Study Hrs/Day", row["study_hours_per_day"])
    c4.metric("Risk Score",    row["rf_risk_score"])

    st.markdown("---")

    # Semester Trend
    st.markdown("### 📅 Your Semester Trend")
    trend = get_student_trend(student_id, sem_df)
    if not trend.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, col, color, title in zip(axes,
                                          ["marks","attendance","study_hours"],
                                          ["#3498db","#2ecc71","#e67e22"],
                                          ["Marks","Attendance (%)","Study Hours"]):
            ax.plot(trend["semester"], trend[col], marker="o", color=color, linewidth=2)
            ax.set_title(title)
            for x, y in zip(trend["semester"], trend[col]):
                ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0,6), fontsize=8, ha="center")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")

    # Recommendation
    st.markdown("### ✅ Your Recommendation")
    st.info(f"📌 {row['recommendation']}")

    st.markdown("---")

    # PDF Download
    st.markdown("### 📄 Download Your Report")
    trend_data = get_student_trend(student_id, sem_df) if sem_df is not None else None
    pdf_bytes = generate_pdf(row.to_dict(), trend_data)
    st.download_button(
        label="⬇ Download PDF Report",
        data=pdf_bytes,
        file_name=f"report_{student_id}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

# ── Router ────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    login_page()
elif st.session_state.role == "admin":
    admin_dashboard()
elif st.session_state.role == "student":
    student_dashboard()
