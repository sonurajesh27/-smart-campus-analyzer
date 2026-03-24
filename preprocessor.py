import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess(df):
    print("[Preprocessor] Starting preprocessing...")
    df = df.copy()
    df.drop_duplicates(inplace=True)

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    le = LabelEncoder()
    df["gender_enc"]       = le.fit_transform(df["gender"])
    df["income_enc"]       = le.fit_transform(df["family_income"])
    df["distance_enc"]     = le.fit_transform(df["distance_from_home"]) \
                             if "distance_from_home" in df.columns else 0
    df["parent_edu_enc"]   = le.fit_transform(df["parental_education"]) \
                             if "parental_education" in df.columns else 0

    feature_cols = [
        "attendance_pct", "study_hours_per_day", "assignments_completed",
        "previous_gpa", "sleep_hours", "extracurricular", "internet_access",
        "tutoring", "part_time_job",
        "gender_enc", "income_enc", "distance_enc", "parent_edu_enc"
    ]
    # Keep only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df["at_risk"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    print(f"[Preprocessor] Features: {X_scaled.shape} | At-risk: {y.mean():.2%}")
    return df, X_scaled, y, feature_cols
