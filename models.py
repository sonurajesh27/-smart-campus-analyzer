import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)

def train_and_evaluate(X, y):
    """Train Logistic Regression and Random Forest, return results and predictions."""
    print("[Models] Training models...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    results = {
        "lr": {
            "model": lr,
            "accuracy": accuracy_score(y_test, lr_pred),
            "auc": roc_auc_score(y_test, lr_proba),
            "report": classification_report(y_test, lr_pred),
            "confusion": confusion_matrix(y_test, lr_pred),
            "pred": lr_pred,
            "proba": lr_proba,
        },
        "rf": {
            "model": rf,
            "accuracy": accuracy_score(y_test, rf_pred),
            "auc": roc_auc_score(y_test, rf_proba),
            "report": classification_report(y_test, rf_pred),
            "confusion": confusion_matrix(y_test, rf_pred),
            "pred": rf_pred,
            "proba": rf_proba,
        },
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train,
    }

    print(f"[Models] Logistic Regression  -> Accuracy: {results['lr']['accuracy']:.4f} | AUC: {results['lr']['auc']:.4f}")
    print(f"[Models] Random Forest        -> Accuracy: {results['rf']['accuracy']:.4f} | AUC: {results['rf']['auc']:.4f}")

    winner = "Random Forest" if results["rf"]["accuracy"] >= results["lr"]["accuracy"] else "Logistic Regression"
    print(f"[Models] Best Model: {winner}")

    return results, winner
