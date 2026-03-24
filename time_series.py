import pandas as pd
import numpy as np

SEMESTERS = ["Sem 1", "Sem 2", "Sem 3", "Sem 4"]

def generate_semester_data(df, seed=42):
    """
    Simulate semester-wise performance for each student.
    Returns a long-format DataFrame with one row per student per semester.
    """
    np.random.seed(seed)
    records = []

    for _, row in df.iterrows():
        base_att   = row["attendance_pct"]
        base_study = row["study_hours_per_day"]
        base_marks = row["marks"]

        for i, sem in enumerate(SEMESTERS):
            # Add slight variation each semester
            att   = np.clip(base_att   + np.random.normal(0, 5),  10, 100)
            study = np.clip(base_study + np.random.normal(0, 0.5), 0,  12)
            marks = np.clip(base_marks + np.random.normal(i * 1.5, 4), 0, 100)
            risk  = int((att < 60) or (marks < 40) or (study < 2))

            records.append({
                "student_id": row["student_id"],
                "semester":   sem,
                "attendance":  round(att, 2),
                "study_hours": round(study, 2),
                "marks":       round(marks, 2),
                "risk_score":  round(np.clip(
                    (1 - att/100) * 0.4 + (1 - study/12) * 0.3 + (1 - marks/100) * 0.3, 0, 1
                ), 4),
                "at_risk": risk,
            })

    return pd.DataFrame(records)


def get_student_trend(student_id, sem_df):
    """Return semester trend for a single student."""
    return sem_df[sem_df["student_id"] == student_id].reset_index(drop=True)
