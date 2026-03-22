import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs("charts", exist_ok=True)
sns.set_theme(style="whitegrid")


def plot_risk_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["at_risk"].value_counts()
    ax.bar(["Safe", "At-Risk"], [counts.get(0, 0), counts.get(1, 0)],
           color=["#2ecc71", "#e74c3c"])
    ax.set_title("Student Risk Distribution")
    ax.set_ylabel("Count")
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        ax.text(i, v + 2, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("charts/risk_distribution.png", dpi=150)
    plt.close()
    print("[Visualizer] Saved: charts/risk_distribution.png")


def plot_attendance_vs_marks(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = df["at_risk"].map({0: "#2ecc71", 1: "#e74c3c"})
    ax.scatter(df["attendance_pct"], df["marks"], c=colors, alpha=0.5, edgecolors="none")
    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("Marks")
    ax.set_title("Attendance vs Marks (Red = At-Risk)")
    from matplotlib.patches import Patch
    legend = [Patch(color="#2ecc71", label="Safe"), Patch(color="#e74c3c", label="At-Risk")]
    ax.legend(handles=legend)
    plt.tight_layout()
    plt.savefig("charts/attendance_vs_marks.png", dpi=150)
    plt.close()
    print("[Visualizer] Saved: charts/attendance_vs_marks.png")


def plot_study_hours_vs_marks(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.regplot(data=df, x="study_hours_per_day", y="marks", ax=ax,
                scatter_kws={"alpha": 0.4, "color": "#3498db"},
                line_kws={"color": "#e67e22"})
    ax.set_title("Study Hours vs Marks (with Trend Line)")
    ax.set_xlabel("Study Hours per Day")
    ax.set_ylabel("Marks")
    plt.tight_layout()
    plt.savefig("charts/study_hours_vs_marks.png", dpi=150)
    plt.close()
    print("[Visualizer] Saved: charts/study_hours_vs_marks.png")


def plot_model_comparison(lr_acc, rf_acc, lr_auc, rf_auc):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    models = ["Logistic Regression", "Random Forest"]

    axes[0].bar(models, [lr_acc, rf_acc], color=["#9b59b6", "#1abc9c"])
    axes[0].set_title("Model Accuracy Comparison")
    axes[0].set_ylim(0, 1)
    for i, v in enumerate([lr_acc, rf_acc]):
        axes[0].text(i, v + 0.01, f"{v:.4f}", ha="center", fontweight="bold")

    axes[1].bar(models, [lr_auc, rf_auc], color=["#e74c3c", "#3498db"])
    axes[1].set_title("Model AUC-ROC Comparison")
    axes[1].set_ylim(0, 1)
    for i, v in enumerate([lr_auc, rf_auc]):
        axes[1].text(i, v + 0.01, f"{v:.4f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("charts/model_comparison.png", dpi=150)
    plt.close()
    print("[Visualizer] Saved: charts/model_comparison.png")


def plot_feature_importance(rf_model, feature_cols):
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_cols[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sorted_features[::-1], sorted_importances[::-1], color="#2980b9")
    ax.set_title("Random Forest - Feature Importances")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("charts/feature_importance.png", dpi=150)
    plt.close()
    print("[Visualizer] Saved: charts/feature_importance.png")


def plot_confusion_matrices(lr_cm, rf_cm):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, cm, title in zip(axes, [lr_cm, rf_cm],
                              ["Logistic Regression", "Random Forest"]):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Safe", "At-Risk"],
                    yticklabels=["Safe", "At-Risk"])
        ax.set_title(f"Confusion Matrix - {title}")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("charts/confusion_matrices.png", dpi=150)
    plt.close()
    print("[Visualizer] Saved: charts/confusion_matrices.png")


def plot_risk_by_income(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    risk_by_income = df.groupby("family_income")["at_risk"].mean().sort_values()
    risk_by_income.plot(kind="bar", ax=ax, color=["#27ae60", "#f39c12", "#c0392b"])
    ax.set_title("At-Risk Rate by Family Income")
    ax.set_ylabel("Risk Rate")
    ax.set_xlabel("Family Income")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for i, v in enumerate(risk_by_income):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("charts/risk_by_income.png", dpi=150)
    plt.close()
    print("[Visualizer] Saved: charts/risk_by_income.png")
