# 🎓 Smart Campus Analyzer

An intelligent campus analytics system built with Python, Machine Learning, and Streamlit. Helps administrators identify at-risk students and make data-driven decisions.

---

## 🚀 Features

- Generates a realistic student dataset (attendance, GPA, study hours, assignments, etc.)
- Data cleaning and preprocessing with feature engineering
- Trains and compares **Logistic Regression** and **Random Forest** models
- Predicts at-risk students with risk scores and tiers
- Generates key insights:
  - Low attendance impact on risk
  - Study hours vs marks correlation
  - Risk rate by family income
- Rule-based recommendations for administrators
- Interactive web dashboard built with **Streamlit**
- Exports a clean dataset for **Power BI** dashboards

---

## 📁 Project Structure

```
smart_campus_analyzer/
├── app.py              # Streamlit web application
├── main.py             # CLI entry point
├── data_loader.py      # Dataset generation and loading
├── preprocessor.py     # Data cleaning and feature encoding
├── models.py           # Logistic Regression + Random Forest
├── insights.py         # Analytics and recommendations
├── visualizer.py       # Matplotlib/Seaborn charts
├── exporter.py         # Power BI CSV export
└── requirements.txt    # Dependencies
```

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/smart-campus-analyzer.git
cd smart-campus-analyzer
python -m pip install -r requirements.txt
```

---

## ▶️ Run the Web App

```bash
python -m streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 🖥️ Run as CLI

```bash
python main.py
```

Outputs charts to `charts/` and exports `powerbi_campus_data.csv`.

---

## 📊 Model Results

| Model               | Accuracy | AUC-ROC |
|---------------------|----------|---------|
| Logistic Regression | ~87%     | ~0.91   |
| Random Forest       | ~99%     | ~0.99   |

---

## 📦 Power BI Integration

After running the app, import `powerbi_campus_data.csv` into Power BI. It includes:

- `risk_label` — Safe / At-Risk
- `risk_tier` — Low / Medium / High Risk
- `rf_risk_score` — probability score (0–1)
- `recommendation` — actionable suggestion per student

---

## 🛠️ Tech Stack

- Python 3.x
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn
- Streamlit

---

## 📄 License

MIT License
