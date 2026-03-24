from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import cm
import io

def generate_pdf(student_row, sem_trend=None):
    """Generate a PDF report for a student and return bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", fontSize=18, fontName="Helvetica-Bold",
                                  textColor=colors.HexColor("#2c3e50"), spaceAfter=6)
    heading_style = ParagraphStyle("heading", fontSize=13, fontName="Helvetica-Bold",
                                    textColor=colors.HexColor("#2980b9"), spaceAfter=4)
    normal = styles["Normal"]

    risk_color = colors.HexColor("#e74c3c") if student_row.get("at_risk", 0) == 1 \
                 else colors.HexColor("#27ae60")
    risk_label = "AT-RISK" if student_row.get("at_risk", 0) == 1 else "SAFE"

    story = []

    # Header
    story.append(Paragraph("🎓 Smart Campus Analyzer", title_style))
    story.append(Paragraph("Student Performance Report", styles["Heading2"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#bdc3c7")))
    story.append(Spacer(1, 0.4*cm))

    # Student Info
    story.append(Paragraph("Student Information", heading_style))
    info_data = [
        ["Student ID",   str(student_row.get("student_id", "N/A"))],
        ["Gender",       str(student_row.get("gender", "N/A"))],
        ["Family Income",str(student_row.get("family_income", "N/A"))],
        ["Risk Status",  risk_label],
        ["Risk Tier",    str(student_row.get("risk_tier", "N/A"))],
        ["RF Risk Score",str(student_row.get("rf_risk_score", "N/A"))],
    ]
    info_table = Table(info_data, colWidths=[5*cm, 10*cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ecf0f1")),
        ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
        ("TEXTCOLOR",  (1, 3), (1, 3), risk_color),
        ("FONTNAME",   (1, 3), (1, 3), "Helvetica-Bold"),
        ("PADDING",    (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.5*cm))

    # Academic Performance
    story.append(Paragraph("Academic Performance", heading_style))
    perf_data = [
        ["Metric", "Value"],
        ["Attendance (%)",         str(student_row.get("attendance_pct", "N/A"))],
        ["Study Hours / Day",      str(student_row.get("study_hours_per_day", "N/A"))],
        ["Assignments Completed",  str(student_row.get("assignments_completed", "N/A"))],
        ["Previous GPA",           str(student_row.get("previous_gpa", "N/A"))],
        ["Sleep Hours",            str(student_row.get("sleep_hours", "N/A"))],
        ["Marks",                  str(student_row.get("marks", "N/A"))],
    ]
    perf_table = Table(perf_data, colWidths=[7*cm, 8*cm])
    perf_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#2980b9")),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
        ("PADDING",      (0, 0), (-1, -1), 6),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 0.5*cm))

    # Semester Trend
    if sem_trend is not None and not sem_trend.empty:
        story.append(Paragraph("Semester-wise Trend", heading_style))
        sem_data = [["Semester", "Attendance", "Study Hrs", "Marks", "Risk Score", "Status"]]
        for _, r in sem_trend.iterrows():
            sem_data.append([
                r["semester"],
                str(r["attendance"]),
                str(r["study_hours"]),
                str(r["marks"]),
                str(r["risk_score"]),
                "At-Risk" if r["at_risk"] == 1 else "Safe",
            ])
        sem_table = Table(sem_data, colWidths=[2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
        sem_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#1abc9c")),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
            ("PADDING",       (0, 0), (-1, -1), 5),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ]))
        story.append(sem_table)
        story.append(Spacer(1, 0.5*cm))

    # Recommendation
    story.append(Paragraph("Recommendation", heading_style))
    rec = str(student_row.get("recommendation", "On track"))
    story.append(Paragraph(rec, normal))
    story.append(Spacer(1, 0.5*cm))

    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#bdc3c7")))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph("Generated by Smart Campus Analyzer", styles["Italic"]))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
