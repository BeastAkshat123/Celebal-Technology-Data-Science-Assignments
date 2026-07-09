"""
Streamlit UI for the Loan Advisory Agent.

Design: bank-ledger / passbook aesthetic. Confidence is shown as a
verification stamp (VERIFIED / PARTIAL MATCH / UNVERIFIED) since that's
literally what the validation layer checks.

Run locally:
    streamlit run app/streamlit_app.py

Deploy: works as-is on Streamlit Community Cloud. Set your API key under
the app's Settings -> Secrets (see README) rather than a .env file, since
.env files aren't uploaded when you deploy from GitHub.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

# --- API key resolution: local .env file OR Streamlit Cloud secrets -------
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

try:
    for _key in ("ANTHROPIC_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        if _key in st.secrets and not os.environ.get(_key):
            os.environ[_key] = st.secrets[_key]
except Exception:
    pass  # no secrets.toml present -- normal for local runs using .env instead

from src.agent import LoanAdvisoryAgent

st.set_page_config(page_title="Loan Advisory Agent", page_icon="\U0001F4D2", layout="centered")

STAMP_STYLE = {
    "high":   {"label": "VERIFIED",      "color": "#4C8C5F"},
    "medium": {"label": "PARTIAL MATCH", "color": "#B98A2E"},
    "low":    {"label": "UNVERIFIED",    "color": "#8B3A3A"},
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --ink: #101826;
    --ink-light: #17233A;
    --parchment: #E9E4D6;
    --brass: #C9A227;
    --line: rgba(233, 228, 214, 0.16);
    --high: #4C8C5F;
    --medium: #B98A2E;
    --low: #8B3A3A;
}

.stApp { background-color: var(--ink); color: var(--parchment); }

h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Lora', serif !important;
    letter-spacing: 0.01em;
}

.ledger-title {
    font-family: 'Lora', serif;
    font-weight: 700;
    font-size: 2.1rem;
    color: var(--parchment);
    border-bottom: 2px solid var(--brass);
    padding-bottom: 0.5rem;
    margin-bottom: 0.2rem;
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
}
.ledger-title .mark { color: var(--brass); font-size: 1.6rem; }
.ledger-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: rgba(233, 228, 214, 0.65);
    margin-bottom: 1.6rem;
}

[data-testid="stChatMessage"] {
    background-color: var(--ink-light);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 0.4rem 0.2rem;
}

.entry-card {
    border: 1px solid var(--line);
    border-left: 3px solid var(--brass);
    background: var(--ink-light);
    padding: 1rem 1.2rem;
    border-radius: 4px;
    margin-bottom: 0.6rem;
    white-space: pre-wrap;
}
.entry-card p { font-size: 0.98rem; line-height: 1.55; margin: 0; }

.stamp-wrap { display: flex; align-items: center; gap: 0.9rem; margin: 0.5rem 0 0.8rem 0; }
.stamp {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    padding: 0.35rem 0.7rem;
    border: 2px dashed var(--stamp-color, var(--brass));
    color: var(--stamp-color, var(--brass));
    border-radius: 3px;
    transform: rotate(-3deg);
    display: inline-block;
}
.stamp-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: rgba(233, 228, 214, 0.55);
}

.source-line {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: rgba(233, 228, 214, 0.75);
    border-top: 1px dotted var(--line);
    padding-top: 0.4rem;
    margin-top: 0.4rem;
}
.source-line::before { content: "\2014 "; color: var(--brass); }

.warn-strip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #D9B04A;
    background: rgba(185, 138, 46, 0.1);
    border-left: 3px solid var(--medium);
    padding: 0.5rem 0.7rem;
    margin-top: 0.5rem;
    border-radius: 2px;
}

section[data-testid="stSidebar"] {
    background-color: var(--ink-light);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .stButton button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    text-align: left;
    background: transparent;
    border: 1px solid var(--line);
    color: var(--parchment);
    width: 100%;
}
section[data-testid="stSidebar"] .stButton button:hover {
    border-color: var(--brass);
    color: var(--brass);
}

[data-testid="stChatInput"] textarea { font-family: 'IBM Plex Mono', monospace; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_agent():
    return LoanAdvisoryAgent()


def render_stamp(confidence: str) -> str:
    style = STAMP_STYLE.get(confidence, STAMP_STYLE["low"])
    return f'<span class="stamp" style="--stamp-color:{style["color"]}">{style["label"]}</span>'


def render_response(resp):
    st.markdown(f'<div class="entry-card">{resp.answer}</div>', unsafe_allow_html=True)

    stamp_html = render_stamp(resp.confidence)
    st.markdown(
        f'<div class="stamp-wrap">{stamp_html}<span class="stamp-meta">mode: {resp.mode}</span></div>',
        unsafe_allow_html=True,
    )

    for w in resp.warnings:
        st.markdown(f'<div class="warn-strip">{w}</div>', unsafe_allow_html=True)

    if resp.sources:
        with st.expander(f"Sources ({len(resp.sources)})"):
            for s in resp.sources:
                st.markdown(f'<div class="source-line">{s}</div>', unsafe_allow_html=True)

    if resp.retrieved_chunks:
        with st.expander("Retrieved context"):
            for rc in resp.retrieved_chunks:
                st.markdown(f"**{rc.chunk.doc_name} — {rc.chunk.section_title}** `score: {rc.score:.2f}`")
                st.code(rc.chunk.text, language=None)


st.markdown(
    '<div class="ledger-title"><span class="mark">\U0001F4D2</span> Loan Advisory Ledger</div>'
    '<div class="ledger-sub">grounded answers &middot; every entry stamped against source documents</div>',
    unsafe_allow_html=True,
)

_has_llm_key = any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"))
if not _has_llm_key:
    st.markdown(
        '<div class="warn-strip">NO LLM KEY DETECTED &mdash; running in template mode. '
        'Set GROQ_API_KEY, OPENROUTER_API_KEY, or ANTHROPIC_API_KEY (via .env locally, '
        'or Secrets if deployed) to enable full generation.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

agent = load_agent()

with st.sidebar:
    st.markdown("### \U0001F4D6 How this works")
    st.markdown(
        "- **Retrieval:** hybrid BM25 + TF-IDF over section-aware chunks\n"
        "- **EMI math:** computed deterministically, never by the LLM\n"
        "- **Verification:** every answer is stamped after checking for "
        "unsupported numbers and out-of-scope questions"
    )
    st.markdown("#### Stamp legend")
    st.markdown(render_stamp("high") + " &nbsp; fully grounded in retrieved text", unsafe_allow_html=True)
    st.markdown(render_stamp("medium") + " &nbsp; partial / moderate match", unsafe_allow_html=True)
    st.markdown(render_stamp("low") + " &nbsp; low confidence / possibly out of scope", unsafe_allow_html=True)

    st.divider()
    st.markdown("### Quick queries")
    examples = [
        "Minimum CIBIL score at Aarna Bank?",
        "EMI for 5 lakh at 11.75% over 48 months?",
        "LTV ratio for Vistara home loan above 75L?",
        "Docs needed as a self-employed applicant?",
    ]
    queued = None
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}"):
            queued = ex

    st.divider()
    if st.button("\U0001F5D1\uFE0F Clear conversation"):
        st.session_state.history = []
        st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["query"])
    with st.chat_message("assistant", avatar="\U0001F4D2"):
        render_response(turn["response"])

user_query = st.chat_input("Ask about eligibility, EMIs, interest rates, or fees...")
final_query = queued or user_query

if final_query:
    with st.chat_message("user"):
        st.markdown(final_query)
    with st.chat_message("assistant", avatar="\U0001F4D2"):
        with st.spinner("Checking the ledger..."):
            resp = agent.ask(final_query)
        render_response(resp)

    st.session_state.history.append({"query": final_query, "response": resp})
