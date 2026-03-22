import pandas as pd

def generate_insights(df):
    """Print key insights from the dataset."""
    print("\n" + "="*55)
    print("         SMART CAMPUS ANALYZER - INSIGHTS")
    print("="*55)

    total = len(df)
    at_risk_count = df["at_risk"].sum()
    print(f"\n Total Students      : {total}")
    print(f" At-Risk Students    : {at_risk_count} ({at_risk_count/total:.1%})")
    print(f" Safe Students       : {total - at_risk_count} ({(total-at_risk_count)/total:.1%})")

    # Attendance impact
    low_att = df[df["attendance_pct"] < 60]
    high_att = df[df["attendance_pct"] >= 60]
    print(f"\n[Insight 1] Low Attendance (<60%) Risk Rate  : {low_att['at_risk'].mean():.1%}")
    print(f"[Insight 1] High Attendance (>=60%) Risk Rate: {high_att['at_risk'].mean():.1%}")

    # Study hours vs marks
    corr = df["study_hours_per_day"].corr(df["marks"])
    print(f"\n[Insight 2] Study Hours vs Marks Correlation : {corr:.4f}")
    if corr > 0.5:
        print("            Strong positive correlation — more study = higher marks.")
    elif corr > 0.2:
        print("            Moderate positive correlation.")
    else:
        print("            Weak correlation — other factors dominate.")

    # GPA impact
    low_gpa = df[df["previous_gpa"] < 2.0]["at_risk"].mean()
    high_gpa = df[df["previous_gpa"] >= 3.0]["at_risk"].mean()
    print(f"\n[Insight 3] Low GPA (<2.0) Risk Rate         : {low_gpa:.1%}")
    print(f"[Insight 3] High GPA (>=3.0) Risk Rate       : {high_gpa:.1%}")

    # Income impact
    print("\n[Insight 4] Risk Rate by Family Income:")
    for income, grp in df.groupby("family_income"):
        print(f"            {income:8s}: {grp['at_risk'].mean():.1%}")

    print("\n" + "="*55)


def generate_recommendations(df):
    """Rule-based recommendations for administrators."""
    print("\n[Recommendations]")

    recs = []

    low_att_pct = (df["attendance_pct"] < 60).mean()
    if low_att_pct > 0.2:
        recs.append(f"  * {low_att_pct:.1%} students have attendance below 60%. Launch attendance drives.")

    low_study = (df["study_hours_per_day"] < 2).mean()
    if low_study > 0.15:
        recs.append(f"  * {low_study:.1%} students study less than 2 hrs/day. Introduce study support programs.")

    low_assign = (df["assignments_completed"] < 5).mean()
    if low_assign > 0.2:
        recs.append(f"  * {low_assign:.1%} students completed fewer than 5 assignments. Enforce submission policies.")

    low_income_risk = df[df["family_income"] == "Low"]["at_risk"].mean()
    if low_income_risk > 0.3:
        recs.append(f"  * Low-income students have {low_income_risk:.1%} risk rate. Consider financial aid programs.")

    if not recs:
        recs.append("  * Campus performance looks healthy. Continue monitoring trends.")

    for r in recs:
        print(r)
    print()
