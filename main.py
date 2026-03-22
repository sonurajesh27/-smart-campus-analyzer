"""
Smart Campus Analyzer
=====================
An intelligent campus analytics system using ML to identify at-risk students
and generate actionable insights for administrators.
"""

from data_loader import generate_dataset, load_dataset
from preprocessor import preprocess
from models import train_and_evaluate
from insights import generate_insights, generate_recommendations
from visualizer import (
    plot_risk_distribution,
    plot_attendance_vs_marks,
    plot_study_hours_vs_marks,
    plot_model_comparison,
    plot_feature_importance,
    plot_confusion_matrices,
    plot_risk_by_income,
)
from exporter import export_for_powerbi


def main():
    print("\n" + "="*55)
    print("       SMART CAMPUS ANALYZER - STARTING")
    print("="*55 + "\n")

    # 1. Load / Generate Data
    generate_dataset(n=500)
    df = load_dataset("dataset.csv")

    # 2. Preprocess
    df_clean, X_scaled, y, feature_cols = preprocess(df)

    # 3. Train Models
    results, winner = train_and_evaluate(X_scaled, y)

    print("\n--- Logistic Regression Report ---")
    print(results["lr"]["report"])
    print("--- Random Forest Report ---")
    print(results["rf"]["report"])

    # 4. Insights
    generate_insights(df_clean)
    generate_recommendations(df_clean)

    # 5. Visualizations
    print("[Visualizer] Generating charts...")
    plot_risk_distribution(df_clean)
    plot_attendance_vs_marks(df_clean)
    plot_study_hours_vs_marks(df_clean)
    plot_model_comparison(
        results["lr"]["accuracy"], results["rf"]["accuracy"],
        results["lr"]["auc"],      results["rf"]["auc"]
    )
    plot_feature_importance(results["rf"]["model"], feature_cols)
    plot_confusion_matrices(results["lr"]["confusion"], results["rf"]["confusion"])
    plot_risk_by_income(df_clean)

    # 6. Export for Power BI
    export_for_powerbi(
        df_clean, X_scaled,
        results["lr"], results["rf"],
        feature_cols
    )

    print("\n" + "="*55)
    print(f"  DONE. Best Model: {winner}")
    print(f"  Charts saved in: charts/")
    print(f"  Power BI file  : powerbi_campus_data.csv")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
