import pandas as pd
import numpy as np

def generate_dataset(n=500, seed=42):
    """Generate a realistic student dataset simulating Kaggle-style data."""
    np.random.seed(seed)

    student_ids = [f"STU{str(i).zfill(4)}" for i in range(1, n + 1)]
    attendance = np.clip(np.random.normal(75, 15, n), 10, 100)
    study_hours = np.clip(np.random.normal(5, 2, n), 0, 12)
    assignments_done = np.random.randint(0, 11, n)
    prev_gpa = np.clip(np.random.normal(2.8, 0.7, n), 0, 4.0)
    sleep_hours = np.clip(np.random.normal(6.5, 1.2, n), 3, 10)
    extracurricular = np.random.randint(0, 2, n)
    internet_access = np.random.randint(0, 2, n)
    family_income = np.random.choice(["Low", "Medium", "High"], n, p=[0.3, 0.5, 0.2])
    gender = np.random.choice(["Male", "Female"], n)

    # Marks influenced by attendance, study hours, assignments, gpa
    marks = (
        0.3 * attendance
        + 4.0 * study_hours
        + 1.5 * assignments_done
        + 8.0 * prev_gpa
        + np.random.normal(0, 5, n)
    )
    marks = np.clip(marks, 0, 100)

    # At-risk: low attendance OR low marks OR low study hours
    at_risk = ((attendance < 60) | (marks < 40) | (study_hours < 2)).astype(int)

    df = pd.DataFrame({
        "student_id": student_ids,
        "gender": gender,
        "family_income": family_income,
        "attendance_pct": attendance.round(2),
        "study_hours_per_day": study_hours.round(2),
        "assignments_completed": assignments_done,
        "previous_gpa": prev_gpa.round(2),
        "sleep_hours": sleep_hours.round(2),
        "extracurricular": extracurricular,
        "internet_access": internet_access,
        "marks": marks.round(2),
        "at_risk": at_risk
    })

    df.to_csv("dataset.csv", index=False)
    print(f"[DataLoader] Dataset generated: {n} students saved to dataset.csv")
    return df


def load_dataset(path="dataset.csv"):
    """Load dataset from CSV."""
    df = pd.read_csv(path)
    print(f"[DataLoader] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
