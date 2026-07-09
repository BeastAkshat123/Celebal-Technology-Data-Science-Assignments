"""
Deterministic EMI (Equated Monthly Installment) calculator.

LLMs are unreliable at multi-step arithmetic. Any numeric loan calculation
must be computed in code and handed to the LLM as a fact to explain, never
generated freely. This module is the "tool" the agent calls for EMI, total
interest, and amortization questions.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class EMIResult:
    principal: float
    annual_rate_pct: float
    tenure_months: int
    emi: float
    total_payment: float
    total_interest: float


def calculate_emi(principal: float, annual_rate_pct: float, tenure_months: int) -> EMIResult:
    if principal <= 0 or tenure_months <= 0:
        raise ValueError("principal and tenure_months must be positive")
    if annual_rate_pct < 0:
        raise ValueError("annual_rate_pct cannot be negative")

    monthly_rate = annual_rate_pct / 12 / 100

    if monthly_rate == 0:
        emi = principal / tenure_months
    else:
        factor = (1 + monthly_rate) ** tenure_months
        emi = principal * monthly_rate * factor / (factor - 1)

    total_payment = emi * tenure_months
    total_interest = total_payment - principal

    return EMIResult(
        principal=round(principal, 2),
        annual_rate_pct=annual_rate_pct,
        tenure_months=tenure_months,
        emi=round(emi, 2),
        total_payment=round(total_payment, 2),
        total_interest=round(total_interest, 2),
    )


@dataclass
class AmortizationRow:
    month: int
    emi: float
    principal_component: float
    interest_component: float
    remaining_balance: float


def amortization_schedule(principal: float, annual_rate_pct: float, tenure_months: int) -> List[AmortizationRow]:
    result = calculate_emi(principal, annual_rate_pct, tenure_months)
    monthly_rate = annual_rate_pct / 12 / 100
    balance = principal
    schedule = []
    for month in range(1, tenure_months + 1):
        interest_component = balance * monthly_rate
        principal_component = result.emi - interest_component
        balance = max(0.0, balance - principal_component)
        schedule.append(AmortizationRow(
            month=month,
            emi=result.emi,
            principal_component=round(principal_component, 2),
            interest_component=round(interest_component, 2),
            remaining_balance=round(balance, 2),
        ))
    return schedule


if __name__ == "__main__":
    r = calculate_emi(principal=500000, annual_rate_pct=11.75, tenure_months=48)
    print(r)
    print(f"\nMonthly EMI: INR {r.emi:,.2f}")
    print(f"Total interest payable: INR {r.total_interest:,.2f}")
    print(f"Total payment: INR {r.total_payment:,.2f}")

    print("\nFirst 3 months of amortization:")
    for row in amortization_schedule(500000, 11.75, 48)[:3]:
        print(row)
