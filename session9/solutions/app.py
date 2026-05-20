"""
Creatinine Clearance Calculator
Cockcroft-Gault equation with nephrotoxic-drug alerts.
Educational demo — not for clinical use.
"""

import os
import sqlite3
import gradio as gr

# Initialize DB on startup if missing (matters for HF Spaces)
if not os.path.exists("drugs.db"):
    import init_db  # noqa: F401


def calculate_crcl(age, weight_kg, serum_cr, sex):
    """Cockcroft-Gault equation with nephrotoxic-drug alerts."""
    if not all([age, weight_kg, serum_cr]):
        return "Please fill in all fields.", ""

    crcl = ((140 - age) * weight_kg) / (72 * serum_cr)
    if sex == "Female":
        crcl *= 0.85
    crcl = round(crcl, 1)

    conn = sqlite3.connect("drugs.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT name, class, note FROM nephrotoxic_drugs WHERE crcl_threshold >= ?",
        (crcl,),
    )
    flagged = cur.fetchall()
    conn.close()

    if flagged:
        alerts = "\n".join(f"⚠️ {d[0]} ({d[1]}): {d[2]}" for d in flagged)
    else:
        alerts = "✅ No nephrotoxic-drug alerts at this CrCl."

    return f"Estimated CrCl: {crcl} mL/min", alerts


demo = gr.Interface(
    fn=calculate_crcl,
    inputs=[
        gr.Number(label="Age (years)"),
        gr.Number(label="Weight (kg)"),
        gr.Number(label="Serum creatinine (mg/dL)"),
        gr.Radio(["Male", "Female"], label="Sex"),
    ],
    outputs=[
        gr.Textbox(label="CrCl result"),
        gr.Textbox(label="Drug alerts", lines=5),
    ],
    title="Creatinine Clearance Calculator",
    description="Cockcroft-Gault with nephrotoxic-drug alerts. Educational demo — not for clinical use.",
)

if __name__ == "__main__":
    demo.launch()
