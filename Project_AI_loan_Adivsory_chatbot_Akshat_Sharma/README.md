# AI Loan Advisory Agent

A retrieval-augmented (RAG) system that answers natural-language questions
about loan eligibility, EMIs, interest rates, and fees by grounding every
answer in actual policy PDFs — with a validation layer that flags
hallucinated or unsupported claims before they reach the user.

## Architecture

```
                 ┌─────────────────┐
  User query ──▶ │  Intent Router   │──▶ EMI calculation? ──▶ deterministic EMI tool
                 └────────┬────────┘                              │
                          │ policy question                       │
                          ▼                                       ▼
                 ┌─────────────────┐                    ┌──────────────────┐
                 │ Hybrid Retriever │                    │  LLM Generation   │
                 │ (BM25 + TF-IDF)  │───────────────────▶│  (strict grounding │
                 └─────────────────┘   retrieved chunks  │   prompt + cites)  │
                                                          └─────────┬────────┘
                                                                    ▼
                                                          ┌──────────────────┐
                                                          │ Faithfulness      │
                                                          │ Validator (stamps │
                                                          │ every answer)      │
                                                          └─────────┬────────┘
                                                                    ▼
                                                        Answer + sources + stamp
```

## Project structure

```
data/
  generate_synthetic_docs.py   # generates the synthetic loan policy PDFs
  raw_pdfs/                    # the dataset (3 fictional lenders)
src/
  ingestion/parser.py           # PDF text + table extraction
  ingestion/chunker.py          # section-aware chunking
  retrieval/retriever.py        # hybrid BM25 + TF-IDF retriever
  generation/router.py          # detects EMI-calculation intent
  generation/emi_calculator.py  # deterministic EMI math
  generation/generator.py       # LLM call (Anthropic/Groq/OpenRouter), .env-aware
  validation/faithfulness.py    # hallucination + confidence ("stamp") checks
  agent.py                      # orchestrates the full pipeline
app/
  cli.py                        # terminal chat interface
  streamlit_app.py              # web UI (ledger/stamp aesthetic)
eval/
  eval_set.py + run_eval.py     # ground-truth retrieval accuracy check
```

---

## Part 1: Run it locally

### 1. Install dependencies
```bash
cd loan_agent_v2
python3 -m venv venv
source venv/Scripts/activate       # Windows Git Bash
# or: venv\Scripts\activate.bat    # Windows CMD
# or: source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### 2. Generate the dataset (one-time)
```bash
python3 data/generate_synthetic_docs.py
```

### 3. Set up a free LLM key
Get one free key (no card needed) from **[console.groq.com/keys](https://console.groq.com/keys)**.

Copy the env template and fill it in:
```bash
cp .env.example .env
```
Open `.env` in a text editor and set:
```
GROQ_API_KEY=gsk_your_actual_key_here
```
Save. (`.env` is git-ignored, so this key never gets committed.)

### 4. Run the app
```bash
cd app
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501`. The "Mode" label under each answer should read `llm:groq` once the key is picked up correctly.

Alternative: terminal-only version — `python3 app/cli.py` from the project root.

---

## Part 2: Deploy it (Streamlit Community Cloud — free)

This gets you a public URL like `yourproject.streamlit.app` that anyone can open, no local setup needed on their end.

### 1. Push the project to GitHub
```bash
git init
git add .
git commit -m "Loan advisory agent"
```
Create a new repo on [github.com](https://github.com/new), then:
```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```
Your `.env` and `venv/` won't be pushed — `.gitignore` already excludes them.

### 2. Deploy on Streamlit Community Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
2. Click **New app**.
3. Pick your repo, branch `main`, and set the main file path to:
   ```
   app/streamlit_app.py
   ```
4. Click **Deploy**. It'll install `requirements.txt` automatically and build the app.

### 3. Add your API key as a Secret (do this before or after first deploy)
Since `.env` files aren't uploaded to GitHub (and shouldn't be), Streamlit Cloud uses its own secrets system instead:

1. On your deployed app's page, click **Settings → Secrets**.
2. Paste in:
   ```
   GROQ_API_KEY = "gsk_your_actual_key_here"
   ```
3. Save. The app restarts automatically and picks it up — `streamlit_app.py` already checks `st.secrets` for this, no code changes needed.

### 4. Done
Your app is now live at a public URL you can share (e.g. for your internship submission). Every time you `git push` an update, Streamlit Cloud redeploys automatically.

---

## Free LLM provider options

| Provider | Cost | Card required? | Notes |
|---|---|---|---|
| **Groq** | Free | No | Recommended. Fast, Llama 3.3 70B. console.groq.com |
| **OpenRouter** | Free | No | 28+ free models. openrouter.ai |
| Anthropic | Paid | Yes | Highest quality |

Priority order if multiple keys are set: `ANTHROPIC_API_KEY` → `GROQ_API_KEY` → `OPENROUTER_API_KEY` → template fallback (raw retrieved facts, no LLM).

## Troubleshooting

**"Invalid API Key" / 401 error**: usually means a stale key was left `export`ed in your terminal from earlier testing. `generator.py` loads `.env` with `override=True` specifically to prevent this — make sure you're on this version of the file, close your terminal fully, reopen, and retry. If it still fails, generate a brand new key (old ones can be silently revoked).

**Streamlit shows "No LLM key detected" but you set one**: confirm `.env` is in the project root (not inside `app/`), has no quotes around the value, and no space around `=`.

## Current status

| Component | Status |
|---|---|
| PDF parsing (incl. tables) | Working |
| Section-aware chunking | Working |
| Hybrid retrieval (BM25 + TF-IDF) | Working — 100% top-3 doc accuracy on eval set |
| EMI calculator | Working, verified against standard formula |
| Intent routing | Working |
| LLM generation | Working with Groq/OpenRouter/Anthropic |
| Hallucination/confidence validation | Working |
| Dense (embedding-based) retrieval | Not implemented — see `embed_dense()` stub in `retriever.py` for where to add it |

## Next steps to extend this
1. Swap TF-IDF/BM25 for dense embeddings (sentence-transformers or a hosted embeddings API) + a vector DB like Chroma.
2. Expand the synthetic dataset and eval set with more lenders and edge cases.
3. Add conversation memory so follow-ups resolve against prior context.
4. Add re-ranking (cross-encoder) on top of initial retrieval.
