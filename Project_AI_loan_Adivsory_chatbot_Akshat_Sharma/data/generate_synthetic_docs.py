"""
Generates synthetic bank loan-policy PDFs for the Loan Advisory Agent project.

Why synthetic instead of scraped: gives us full control over ground truth,
so we can build a matched eval set (question -> exact source fact) to measure
retrieval and generation accuracy later. In a real deployment these would be
swapped for actual bank policy PDFs / RBI circulars.
"""

import os
from fpdf import FPDF
from fpdf.enums import XPos, YPos

OUT_DIR = os.path.join(os.path.dirname(__file__), "raw_pdfs")
os.makedirs(OUT_DIR, exist_ok=True)


class PolicyPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, self.title_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.ln(2)

    def section(self, heading, body_lines):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, heading, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 10)
        for line in body_lines:
            self.set_x(self.l_margin)
            self.multi_cell(0, 6, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)

    def table(self, headers, rows):
        self.set_font("Helvetica", "B", 10)
        col_w = 190 / len(headers)
        for h in headers:
            self.cell(col_w, 7, h, border=1)
        self.ln()
        self.set_font("Helvetica", "", 10)
        for row in rows:
            for cell in row:
                self.cell(col_w, 7, str(cell), border=1)
            self.ln()
        self.ln(4)


DOCS = [
    {
        "filename": "aarna_bank_personal_loan_policy.pdf",
        "title": "Aarna Bank - Personal Loan Policy 2026",
        "sections": [
            ("1. Eligibility Criteria", [
                "Applicant must be a resident Indian citizen aged between 21 and 60 years at loan maturity.",
                "Minimum net monthly income of INR 25,000 for salaried applicants in metro cities, and INR 20,000 in non-metro cities.",
                "Self-employed applicants must show a minimum annual income of INR 3,00,000 with at least 2 years of business continuity.",
                "Minimum CIBIL score required is 700. Applicants with a score below 700 may be considered with additional collateral at the branch manager's discretion.",
                "Applicant must have been employed with the current employer for a minimum of 1 year, and total work experience of at least 2 years.",
            ]),
            ("2. Loan Amount and Tenure", [
                "Loan amount ranges from INR 50,000 to INR 25,00,000 for salaried applicants.",
                "Loan amount ranges from INR 50,000 to INR 15,00,000 for self-employed applicants.",
                "Maximum tenure is 60 months (5 years) for all personal loan applicants.",
                "Minimum tenure is 12 months.",
            ]),
            ("3. Interest Rate Slabs", []),
            ("4. Processing Fees and Charges", [
                "Processing fee is 2% of the loan amount or INR 2,500, whichever is higher, plus applicable GST.",
                "Prepayment/foreclosure charges are 4% of the outstanding principal if foreclosed within 12 months of disbursement, and 2% thereafter.",
                "No prepayment charges apply after 36 months from disbursement.",
                "Late payment penalty is 2% per month on the overdue EMI amount.",
            ]),
            ("5. Documents Required", [
                "Identity proof: PAN card and Aadhaar card.",
                "Address proof: Aadhaar, passport, or utility bill not older than 3 months.",
                "Income proof (salaried): last 3 months' salary slips and last 6 months' bank statement.",
                "Income proof (self-employed): last 2 years' ITR and audited financials, plus last 12 months' bank statement.",
                "Passport-size photographs (2).",
            ]),
        ],
        "table": {
            "after_section": "3. Interest Rate Slabs",
            "headers": ["CIBIL Score", "Interest Rate (p.a.)"],
            "rows": [
                ["750 and above", "10.50%"],
                ["700 - 749", "11.75%"],
                ["650 - 699", "13.50% (collateral required)"],
                ["Below 650", "Not eligible"],
            ],
        },
    },
    {
        "filename": "sundhara_finance_personal_loan_policy.pdf",
        "title": "Sundhara Finance Ltd - Personal Loan Guidelines 2026",
        "sections": [
            ("1. Eligibility Criteria", [
                "Age of applicant must be between 23 and 58 years at the time of application.",
                "Minimum monthly take-home income of INR 30,000 for salaried employees.",
                "Self-employed professionals (doctors, CAs, architects) require minimum annual turnover of INR 5,00,000.",
                "Minimum credit score requirement is 720. This is stricter than most peer NBFCs.",
                "Applicant must not have any loan default in the last 24 months as per credit bureau records.",
            ]),
            ("2. Loan Amount and Tenure", [
                "Loan amount available from INR 1,00,000 up to INR 20,00,000.",
                "Maximum repayment tenure is 48 months (4 years).",
                "Minimum tenure is 6 months.",
            ]),
            ("3. Interest Rate Slabs", []),
            ("4. Processing Fees and Charges", [
                "One-time processing fee of 2.5% of sanctioned amount, non-refundable.",
                "Foreclosure allowed after 6 EMIs with a charge of 5% on outstanding principal.",
                "Bounce charges of INR 750 per instance for failed EMI auto-debit.",
                "Duplicate statement/NOC charge: INR 500 per document.",
            ]),
            ("5. Documents Required", [
                "PAN card and government-issued photo ID.",
                "Latest 3 salary slips and Form 16 for salaried applicants.",
                "Bank statements for last 6 months from salary/primary account.",
                "For self-employed: GST returns and P&L statement for last 2 financial years.",
            ]),
        ],
        "table": {
            "after_section": "3. Interest Rate Slabs",
            "headers": ["CIBIL Score", "Interest Rate (p.a.)"],
            "rows": [
                ["760 and above", "10.99%"],
                ["720 - 759", "12.25%"],
                ["Below 720", "Not eligible"],
            ],
        },
    },
    {
        "filename": "vistara_home_loan_policy.pdf",
        "title": "Vistara Housing Finance - Home Loan Policy 2026",
        "sections": [
            ("1. Eligibility Criteria", [
                "Applicant age must be between 21 and 65 years at loan maturity (retirement age considered for salaried).",
                "Minimum monthly income of INR 35,000 required for salaried applicants; INR 40,000 for self-employed.",
                "Minimum CIBIL score of 650 required; scores between 650-699 attract a 0.25% interest rate markup.",
                "Co-applicant income can be clubbed for eligibility if co-applicant is an immediate family member.",
            ]),
            ("2. Loan Amount and Loan-to-Value (LTV) Ratio", [
                "Loans up to INR 30,00,000: LTV up to 90% of property value.",
                "Loans between INR 30,00,001 and INR 75,00,000: LTV up to 80%.",
                "Loans above INR 75,00,000: LTV up to 75%.",
                "Maximum tenure is 30 years, subject to applicant's age at maturity.",
            ]),
            ("3. Interest Rate Slabs", []),
            ("4. Processing Fees and Charges", [
                "Processing fee: 0.50% of loan amount plus applicable taxes, minimum INR 5,000.",
                "No prepayment penalty on floating-rate home loans as per RBI guidelines for individual borrowers.",
                "Fixed-rate home loans attract a 2% foreclosure charge on outstanding principal.",
                "Legal and technical valuation charges are borne by the applicant, typically INR 8,000-15,000.",
            ]),
            ("5. Documents Required", [
                "Identity and address proof (Aadhaar, PAN, passport).",
                "Property documents: sale agreement, title deed, encumbrance certificate.",
                "Income proof as applicable to salaried/self-employed category.",
                "Last 12 months' bank statement showing EMI/rent payment history if any.",
            ]),
        ],
        "table": {
            "after_section": "3. Interest Rate Slabs",
            "headers": ["CIBIL Score", "Interest Rate (p.a.)"],
            "rows": [
                ["750 and above", "8.75%"],
                ["700 - 749", "9.10%"],
                ["650 - 699", "9.35%"],
                ["Below 650", "Not eligible"],
            ],
        },
    },
]


def build_pdf(doc):
    pdf = PolicyPDF()
    pdf.title_text = doc["title"]
    pdf.add_page()
    for heading, body in doc["sections"]:
        pdf.section(heading, body)
        if doc.get("table") and doc["table"]["after_section"] == heading:
            pdf.table(doc["table"]["headers"], doc["table"]["rows"])
    out_path = os.path.join(OUT_DIR, doc["filename"])
    pdf.output(out_path)
    print(f"Created {out_path}")


if __name__ == "__main__":
    for d in DOCS:
        build_pdf(d)
    print(f"\n{len(DOCS)} synthetic policy PDFs generated in {OUT_DIR}")
