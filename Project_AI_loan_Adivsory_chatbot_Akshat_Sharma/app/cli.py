"""
Simple interactive CLI for the Loan Advisory Agent.

Usage:
    python3 app/cli.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import LoanAdvisoryAgent


def main():
    print("=" * 60)
    print("  Loan Advisory Agent (CLI demo)")
    print("=" * 60)
    if not any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY")):
        print("\n[Note] No LLM API key set -- running in TEMPLATE MODE.")
        print("Answers will show raw retrieved facts instead of an LLM-composed")
        print("response. Set one of these env vars to enable full natural-language answers:")
        print("    export GROQ_API_KEY=your_key_here        (free, recommended)")
        print("    export OPENROUTER_API_KEY=your_key_here   (free)")
        print("    export ANTHROPIC_API_KEY=your_key_here    (paid)\n")

    agent = LoanAdvisoryAgent()
    print("\nAsk a loan-related question (or type 'quit' to exit).")
    print("Example: 'What is the EMI for a 5 lakh loan at 11.75% for 48 months?'\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        resp = agent.ask(query)
        print(f"\nAgent: {resp.answer}\n")
        print(f"[confidence: {resp.confidence} | mode: {resp.mode}]")
        if resp.warnings:
            print(f"[warnings: {resp.warnings}]")
        print()


if __name__ == "__main__":
    main()
