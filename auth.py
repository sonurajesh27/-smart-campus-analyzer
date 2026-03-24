import pandas as pd
import hashlib

# Simple in-memory user store (admin + students loaded from dataset)
ADMIN_CREDENTIALS = {
    "admin": hashlib.sha256("admin123".encode()).hexdigest()
}

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_admin(username, password):
    return (username in ADMIN_CREDENTIALS and
            ADMIN_CREDENTIALS[username] == hash_password(password))

def verify_student(student_id, password, df):
    """Students log in with their student_id and password = last 4 chars of ID."""
    if df is None or "student_id" not in df.columns:
        return False
    expected_password = student_id[-4:]
    return (student_id in df["student_id"].values and
            password == expected_password)

def get_student_data(student_id, df):
    row = df[df["student_id"] == student_id]
    if row.empty:
        return None
    return row.iloc[0]
