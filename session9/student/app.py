"""
Creatinine Clearance Calculator
Cockcroft-Gault equation with nephrotoxic-drug alerts.
Educational demo — not for clinical use.

YOUR TASK: Complete the calculate_crcl function and the Gradio interface.
"""

import os
import sqlite3
import gradio as gr

# Initialize DB on startup if missing (matters for HF Spaces)
if not os.path.exists("drugs.db"):
    import init_db  # noqa: F401


def calculate_crcl(age, weight_kg, serum_cr, sex):
    """Cockcroft-Gault equation with nephrotoxic-drug alerts.

    Formula:  CrCl = ((140 - age) * weight_kg) / (72 * serum_cr)
    If female, multiply by 0.85
    """
    # TODO: Validate that all inputs are provided
    # Hint: return "Please fill in all fields.", "" if any is missing

    # TODO: Apply the Cockcroft-Gault formula
    # crcl = ...

    # TODO: Apply the female correction (multiply by 0.85)

    # TODO: Round CrCl to 1 decimal place

    # TODO: Query the database for nephrotoxic drugs with threshold >= crcl
    # conn = sqlite3.connect("drugs.db")
    # cur = conn.cursor()
    # cur.execute("SELECT name, class, note FROM nephrotoxic_drugs WHERE crcl_threshold >= ?", (crcl,))
    # flagged = cur.fetchall()
    # conn.close()

    # TODO: Build the alerts string
    # If flagged: format as "⚠️ {name} ({class}): {note}" per line
    # Else: "✅ No nephrotoxic-drug alerts at this CrCl."

    # TODO: Return tuple (crcl_message, alerts_message)
    pass


# TODO: Build the Gradio Interface
# Inputs: age (Number), weight_kg (Number), serum_cr (Number), sex (Radio: Male/Female)
# Outputs: CrCl result (Textbox), Drug alerts (Textbox with lines=5)
# Title: "Creatinine Clearance Calculator"
# Description: "Cockcroft-Gault with nephrotoxic-drug alerts. Educational demo — not for clinical use."

# demo = gr.Interface(
#     fn=...,
#     inputs=[...],
#     outputs=[...],
#     title=...,
#     description=...,
# )


if __name__ == "__main__":
    # TODO: Launch the demo
    # demo.launch()
    pass
