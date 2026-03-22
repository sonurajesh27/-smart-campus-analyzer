import pandas as pd
import numpy as np

def export_for_powerbi(df, X_scaled, lr_results, rf_results, feature_cols):
    """
    Merge predictions and risk scores back into the original dataframe
    and export a clean CSV ready for Power BI.
    """
    print("[Exporter] Preparing Power BI export...")

    export_df = df.copy().reset_index(drop=True)

    # Full-dataset predictions using best model (Random Forest)
    rf_model = rf_results["model"]
    lr_model = lr_results["model"]

    export_df["lr_prediction"] = lr_model.predict(X_scaled)
    export_df["rf_prediction"] = rf_model.predict(X_scaled)
    export_df["lr_risk_score"] = lr_model.predict_proba(X_scaled)[:, 1].round(4)
    export_df["rf_risk_score"] = rf_model.predict_proba(X_scaled)[:, 1].round(4)

    # Risk label
    export_df["risk_label"] = export_df["rf_prediction"].map({0: "Safe", 1: "At-Risk"})

    # Risk tier based on rf_risk_score
    def risk_tier(score):
        if score >= 0.7:
            return "High Risk"
        elif score >= 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"

    export_df["risk_tier"] = export_df["rf_risk_score"].apply(risk_tier)

    # Recommendation column
    def recommend(row):
        tips = []
        if row["attendance_pct"] < 60:
            tips.append("Improve attendance")
        if row["study_hours_per_day"] < 2:
            tips.append("Increase study hours")
        if row["assignments_completed"] < 5:
            tips.append("Complete assignments")
        if row["previous_gpa"] < 2.0:
            tips.append("Seek academic counseling")
        return "; ".join(tips) if tips else "On track"

    export_df["recommendation"] = export_df.apply(recommend, axis=1)

    # Drop encoded columns (not needed in Power BI)
    export_df.drop(columns=["gender_enc", "income_enc"], errors="ignore", inplace=True)

    output_path = "powerbi_campus_data.csv"
    export_df.to_csv(output_path, index=False)
    print(f"[Exporter] Power BI dataset saved: {output_path} ({len(export_df)} rows, {len(export_df.columns)} columns)")
    return export_df
