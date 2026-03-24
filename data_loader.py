import pandas as pd
import numpy as np

# Kaggle-style realistic distributions based on
# "Student Performance Factors" dataset (public domain)
def generate_dataset(n=1000, seed=42):
    """Generate a Kaggle-style realistic student dataset."""
    np.random.seed(seed)

    student_ids      = [f"STU{str(i).zfill(4)}" for i in range(1, n + 1)]
    gender           = np.random.choice(["Male", "Female"], n, p=[0.51, 0.49])
    family_income    = np.random.choice(["Low", "Medium", "High"], n, p=[0.28, 0.52, 0.20])
    parental_edu     = np.random.choice(["None", "High School", "Bachelor", "Master"], n,
                                         p=[0.10, 0.35, 0.40, 0.15])
    internet_access  = np.random.choice([0, 1], n, p=[0.20, 0.80])
    extracurricular  = np.random.choice([0, 1], n, p=[0.45, 0.55])
    tutoring         = np.random.choice([0, 1], n, p=[0.60, 0.40])

    attendance_pct       = np.clip(np.random.normal(78, 14, n), 10, 100)
    study_hours_per_day  = np.clip(np.random.normal(4.8, 2.1, n), 0, 12)
    assignments_completed= np.random.randint(0, 11, n)
    previous_gpa         = np.clip(np.random.normal(2.9, 0.65, n), 0.0, 4.0)
    sleep_hours          = np.clip(np.random.normal(6.8, 1.1, n), 3, 10)
    distance_from_home   = np.random.choice(["Near", "Moderate", "Far"], n, p=[0.40, 0.35, 0.25])
    part_time_job        = np.random.choice([0, 1], n, p=[0.70, 0.30])

    # Marks formula (Kaggle-inspired)
    marks = (
        0.25 * attendance_pct
        + 3.8  * study_hours_per_day
        + 1.2  * assignments_completed
        + 7.5  * previous_gpa
        + 2.0  * tutoring
        - 1.5  * part_time_job
        + np.random.normal(0, 4, n)
    )
    marks = np.clip(marks, 0, 100).round(2)

    at_risk = ((attendance_pct < 60) | (marks < 40) | (study_hours_per_day < 2)).astype(int)

    df = pd.DataFrame({
        "student_id":           student_ids,
        "gender":               gender,
        "family_income":        family_income,
        "parental_education":   parental_edu,
        "distance_from_home":   distance_from_home,
        "internet_access":      internet_access,
        "extracurricular":      extracurricular,
        "tutoring":             tutoring,
        "part_time_job":        part_time_job,
        "attendance_pct":       attendance_pct.round(2),
        "study_hours_per_day":  study_hours_per_day.round(2),
        "assignments_completed":assignments_completed,
        "previous_gpa":         previous_gpa.round(2),
        "sleep_hours":          sleep_hours.round(2),
        "marks":                marks,
        "at_risk":              at_risk,
    })

    df.to_csv("dataset.csv", index=False)
    print(f"[DataLoader] Kaggle-style dataset generated: {n} students → dataset.csv")
    return df


def load_dataset(path="dataset.csv"):
    df = pd.read_csv(path)
    print(f"[DataLoader] Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
