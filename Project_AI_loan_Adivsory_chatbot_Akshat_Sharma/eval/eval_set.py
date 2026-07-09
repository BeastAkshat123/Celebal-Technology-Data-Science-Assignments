"""
Ground-truth evaluation set for the Loan Advisory Agent.

Each item maps a natural-language question to the document + section it
should be retrieved from. Because we built the source PDFs ourselves
(synthetic dataset), we know the exact ground truth -- this lets us measure
retrieval hit-rate objectively instead of just eyeballing answers.
"""

EVAL_SET = [
    {
        "query": "What is the minimum CIBIL score required for a personal loan at Aarna Bank?",
        "expected_doc": "aarna_bank_personal_loan_policy.pdf",
        "expected_section_keywords": ["eligibility"],
    },
    {
        "query": "What is the age limit for Sundhara Finance personal loan applicants?",
        "expected_doc": "sundhara_finance_personal_loan_policy.pdf",
        "expected_section_keywords": ["eligibility"],
    },
    {
        "query": "What is the maximum tenure for a Vistara home loan?",
        "expected_doc": "vistara_home_loan_policy.pdf",
        "expected_section_keywords": ["loan amount", "ltv"],
    },
    {
        "query": "What is the processing fee for an Aarna Bank personal loan?",
        "expected_doc": "aarna_bank_personal_loan_policy.pdf",
        "expected_section_keywords": ["fees", "charges"],
    },
    {
        "query": "What happens if I bounce an EMI payment at Sundhara Finance?",
        "expected_doc": "sundhara_finance_personal_loan_policy.pdf",
        "expected_section_keywords": ["fees", "charges"],
    },
    {
        "query": "What documents are needed for a self-employed applicant at Aarna Bank?",
        "expected_doc": "aarna_bank_personal_loan_policy.pdf",
        "expected_section_keywords": ["documents"],
    },
    {
        "query": "What is the loan-to-value ratio for a Vistara home loan between 30 and 75 lakhs?",
        "expected_doc": "vistara_home_loan_policy.pdf",
        "expected_section_keywords": ["loan amount", "ltv"],
    },
    {
        "query": "Is there a foreclosure penalty on floating-rate Vistara home loans?",
        "expected_doc": "vistara_home_loan_policy.pdf",
        "expected_section_keywords": ["fees", "charges"],
    },
    {
        "query": "What interest rate applies for a CIBIL score of 730 at Sundhara Finance?",
        "expected_doc": "sundhara_finance_personal_loan_policy.pdf",
        "expected_section_keywords": ["interest"],
    },
    {
        "query": "Can co-applicant income be combined for a Vistara home loan?",
        "expected_doc": "vistara_home_loan_policy.pdf",
        "expected_section_keywords": ["eligibility"],
    },
]
