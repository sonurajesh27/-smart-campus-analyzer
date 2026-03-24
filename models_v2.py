import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

def train_all_models(X, y):
    print("[Models v2] Training 5 models...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_defs = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=100, random_state=42,
                                             eval_metric="logloss", verbosity=0),
        "SVM":                 SVC(probability=True, random_state=42),
        "Neural Network":      MLPClassifier(hidden_layer_sizes=(64, 32),
                                             max_iter=500, random_state=42),
    }

    results = {}
    for name, model in model_defs.items():
        model.fit(X_train, y_train)
        pred  = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "model":     model,
            "accuracy":  accuracy_score(y_test, pred),
            "auc":       roc_auc_score(y_test, proba),
            "report":    classification_report(y_test, pred),
            "confusion": confusion_matrix(y_test, pred),
            "pred":      pred,
            "proba":     proba,
        }
        print(f"  {name:22s} -> Acc: {results[name]['accuracy']:.4f} | AUC: {results[name]['auc']:.4f}")

    best = max(results, key=lambda k: results[k]["accuracy"])
    print(f"[Models v2] Best Model: {best}")

    results["X_test"]  = X_test
    results["y_test"]  = y_test
    results["X_train"] = X_train
    results["y_train"] = y_train

    return results, best
