import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess(df):
    """Clean and encode the dataset for ML models."""
    print("[Preprocessor] Starting preprocessing...")

    df = df.copy()

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Fill missing values (pandas 3.x CoW-compatible)
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categoricals
    le = LabelEncoder()
    df["gender_enc"] = le.fit_transform(df["gender"])
    df["income_enc"] = le.fit_transform(df["family_income"])

    # Feature columns for model
    feature_cols = [
        "attendance_pct", "study_hours_per_day", "assignments_completed",
        "previous_gpa", "sleep_hours", "extracurricular",
        "internet_access", "gender_enc", "income_enc"
    ]

    X = df[feature_cols]
    y = df["at_risk"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    print(f"[Preprocessor] Features ready: {X_scaled.shape}, At-risk ratio: {y.mean():.2%}")
    return df, X_scaled, y, feature_cols
