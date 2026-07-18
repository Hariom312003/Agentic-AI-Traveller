"""
PDF export.

Uses fpdf2 directly (no HTML-to-PDF conversion step, no wkhtmltopdf/Chromium
dependency) so this works anywhere Python does, on whatever machine this
gets deployed to.

Unicode note: fpdf2's built-in core fonts (Helvetica/Times/Courier) only
support Latin-1. LLM-generated text routinely contains "smart" punctuation
(em-dashes, curly quotes, ellipses) that falls outside it and would
otherwise crash export with a `FPDFUnicodeEncodingException` on whatever
itinerary happens to trigger it first. `_safe_text` normalizes the common
cases to plain ASCII equivalents and falls back to dropping anything still
unmappable (rather than crashing) for the rare truly out-of-range
character — a dropped character is a much better failure mode than a
failed export.
"""
from __future__ import annotations

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from src.utils.timeutil import utc_now

_SLOT_LABELS = {"morning": "Morning", "afternoon": "Afternoon", "evening": "Evening", "night": "Night"}

_UNICODE_TO_ASCII = {
    "\u2014": "-", "\u2013": "-",           # em dash, en dash
    "\u2018": "'", "\u2019": "'",           # curly single quotes
    "\u201c": '"', "\u201d": '"',           # curly double quotes
    "\u2026": "...",                          # ellipsis
    "\u2022": "-",                             # bullet
    "\u20b9": "Rs.",                          # Indian Rupee sign
}


def _safe_text(value: object) -> str:
    text = "" if value is None else str(value)
    for unicode_char, ascii_equivalent in _UNICODE_TO_ASCII.items():
        text = text.replace(unicode_char, ascii_equivalent)
    # Final safety net: anything still outside Latin-1 (core PDF fonts'
    # supported range) is dropped rather than crashing the export.
    return text.encode("latin-1", errors="ignore").decode("latin-1")


class _ItineraryPDF(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, self.title_text, align="L")
        self.ln(10)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def build_itinerary_pdf(itinerary: dict, budget: dict | None, rewards: dict | None) -> bytes:
    pdf = _ItineraryPDF(format="A4", unit="mm")
    pdf.title_text = _safe_text(f"{itinerary['destination']} - {itinerary['duration_days']}-day itinerary")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # --- Title ---
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(22, 33, 58)
    pdf.cell(0, 14, _safe_text(f"{itinerary['destination']} Trip Itinerary"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, _safe_text(
        f"{itinerary['duration_days']} days  |  {itinerary['travelers_count']} traveler(s)  |  "
        f"Generated {utc_now().strftime('%d %b %Y')}"
    ), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # --- Days ---
    for day in itinerary["days"]:
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(47, 110, 94)
        day_title = f"Day {day['day_number']}" + (f" - {day['theme']}" if day.get("theme") else "")
        pdf.cell(0, 10, _safe_text(day_title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(220, 220, 210)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 180, pdf.get_y())
        pdf.ln(2)

        for slot in ("morning", "afternoon", "evening", "night"):
            activities = day.get(slot, [])
            if not activities:
                continue
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(192, 138, 40)
            pdf.cell(0, 7, _SLOT_LABELS[slot].upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            for activity in activities:
                pdf.set_font("Helvetica", "B", 10.5)
                pdf.set_text_color(22, 33, 58)
                time_str = f"{activity['start_time']}  " if activity.get("start_time") else ""
                pdf.multi_cell(0, 6, _safe_text(f"{time_str}{activity['title']}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                if activity.get("description"):
                    pdf.set_font("Helvetica", "", 9.5)
                    pdf.set_text_color(70, 70, 70)
                    pdf.multi_cell(0, 5.2, _safe_text(activity["description"]), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                cost_bits = []
                if activity.get("estimated_cost"):
                    cost_bits.append(f"~{activity['estimated_cost']:.0f} {activity.get('currency', 'INR')}")
                if activity.get("duration_minutes"):
                    cost_bits.append(f"{activity['duration_minutes']} min")
                if cost_bits:
                    pdf.set_font("Helvetica", "I", 8.5)
                    pdf.set_text_color(130, 130, 130)
                    pdf.cell(0, 5, _safe_text(" - ".join(cost_bits)), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(1.5)
            pdf.ln(1)
        pdf.ln(3)

    # --- Budget ---
    if budget:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(22, 33, 58)
        pdf.cell(0, 12, "Budget Breakdown", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10.5)
        rows = [
            ("Flights", budget.get("flights", 0)), ("Hotels", budget.get("hotels", 0)),
            ("Food", budget.get("food", 0)), ("Activities", budget.get("activities", 0)),
            ("Shopping", budget.get("shopping", 0)), ("Local Transport", budget.get("local_transport", 0)),
            ("Emergency Buffer", budget.get("emergency_buffer", 0)), ("Taxes & Fees", budget.get("taxes_and_fees", 0)),
        ]
        for label, value in rows:
            pdf.set_text_color(60, 60, 60)
            pdf.cell(90, 8, label)
            pdf.cell(0, 8, f"{value:,.0f} {budget.get('currency', 'INR')}",
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
        pdf.set_draw_color(200, 200, 200)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 180, pdf.get_y())
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(22, 33, 58)
        pdf.cell(90, 10, "Total")
        pdf.cell(0, 10, f"{budget.get('total', 0):,.0f} {budget.get('currency', 'INR')}",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")

    # --- Rewards ---
    if rewards and rewards.get("recommendations"):
        pdf.ln(6)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(22, 33, 58)
        pdf.cell(0, 10, "Reward & Savings Ideas (illustrative)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(140, 140, 140)
        pdf.multi_cell(0, 5, _safe_text(rewards.get("disclaimer", "")), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)
        for rec in rewards["recommendations"]:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(40, 40, 40)
            pdf.cell(0, 6, _safe_text(f"{rec['category']}: {rec['instrument']}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(90, 90, 90)
            pdf.multi_cell(0, 5, _safe_text(rec["reason"]), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)

    return bytes(pdf.output())
