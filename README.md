# SHL Assessment Recommender

A production-ready **conversational AI agent** that helps hiring managers find the right SHL assessments from the official SHL product catalog. Built with a stateless FastAPI service, BM25 keyword retrieval, and a live-benchmarked free LLM fallback chain via OpenRouter.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Running the Server](#running-the-server)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Deployment](#deployment)
- [Design Decisions](#design-decisions)

---

## Overview

This agent accepts multi-turn conversations and recommends SHL assessments grounded in the real SHL product catalog (377 assessments). It:

- Asks targeted clarifying questions before recommending (avoids shotgun recommendations)
- Returns structured JSON with `reply`, `recommendations`, and `end_of_conversation`
- Validates every recommendation URL against the live catalog — **zero hallucination on URLs**
- Enforces a strict 8-turn conversation cap with dynamic prompting
- Deflects off-topic queries (salary, legal, prompt injection) via regex pre-filtering

---

## Architecture

```
User Request (POST /chat)
        │
        ▼
┌─────────────────────┐
│   FastAPI (main.py) │  ← Stateless, validates schema, enforces turn limit
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   agent.py          │  ← Off-topic filter → BM25 retrieval → LLM call → JSON parse → validate
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
retriever.py   OpenRouter API
(BM25/TF-IDF)  (deepseek/deepseek-v4-flash:free)
    │
catalog.json (377 SHL assessments)
```

| Component | Technology |
|---|---|
| **Web Framework** | FastAPI + Uvicorn |
| **LLM** | `openai/gpt-oss-20b:free` via [OpenRouter](https://openrouter.ai) (free tier, ~3s/turn) |
| **LLM Fallbacks** | `minimax/minimax-m2.5:free` → `openai/gpt-oss-120b:free` (auto-cycle on 429) |
| **Retrieval** | Custom BM25 / TF-IDF in pure Python + NumPy |
| **Catalog** | 377 SHL assessments scraped and structured into `catalog.json` |
| **Validation** | Double-pass: JSON extraction + URL/name cross-check against catalog |

---

## Project Structure

```
shl/
├── main.py               # FastAPI app — /health and /chat endpoints
├── agent.py              # Core agent logic (retrieval + LLM + validation)
├── retriever.py          # BM25 index — tokenization, scoring, search
├── catalog.json          # Structured SHL product catalog (377 items)
├── scrape_catalog.py     # Script used to scrape & build catalog.json
├── prepare_catalog.py    # Cleans and normalizes scraped data
├── check_api.py          # Diagnostic: validates API key + LLM call
├── test_retriever.py     # Tests BM25 retrieval with sample queries
├── test_api.py           # End-to-end test of /health and /chat
├── requirements.txt      # Python dependencies
├── Approach_Document.md  # Detailed design decisions and trade-offs
└── .venv/                # Virtual environment (not committed)
```

---

## Setup Instructions

### 1. Clone / Navigate to the Project

```bash
cd shl
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv .venv
```

Activate it:
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (CMD):** `.venv\Scripts\activate.bat`
- **Linux/Mac:** `source .venv/bin/activate`

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Get a Free OpenRouter API Key

1. Go to [https://openrouter.ai](https://openrouter.ai) and sign up (free, no credit card)
2. Navigate to **Keys** → **Create API Key**
3. Copy your key (starts with `sk-or-v1-...`)

### 5. Set the API Key

**Windows (PowerShell):**
```powershell
$env:OPENROUTER_API_KEY="sk-or-v1-your-key-here"
```

**Windows (CMD):**
```cmd
set OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

**Linux/Mac:**
```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
```

### 6. Verify the Setup

```bash
python check_api.py
```

Expected output:
```
Key found: sk-or-v1...xxxx (length=73)
OpenRouter response OK:
{"reply": "hello"}
```

---

## Running the Server

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at:
- **Base URL:** `http://localhost:8000`
- **Interactive Docs (Swagger):** `http://localhost:8000/docs`
- **Health Check:** `http://localhost:8000/health`

### Optional: Change the LLM Model

The default model is `openai/gpt-oss-20b:free` (~3–5s response time, benchmarked fastest free model).
The agent automatically falls back through `minimax/minimax-m2.5:free` → `openai/gpt-oss-120b:free` if rate-limited.
You can override with any free model from [openrouter.ai/models](https://openrouter.ai/models?q=:free):

```powershell
$env:OPENROUTER_MODEL="openai/gpt-oss-120b:free"
```

---

## API Reference

### `GET /health`

Health check endpoint.

**Response:**
```json
{ "status": "ok" }
```

---

### `POST /chat`

Main conversational endpoint. Accepts the full conversation history and returns the agent's next reply.

**Request Body:**
```json
{
  "messages": [
    { "role": "user", "content": "I need assessments for a Java developer" },
    { "role": "assistant", "content": "..." },
    { "role": "user", "content": "Mid-level, 3-5 years experience" }
  ]
}
```

**Constraints:**
- `messages` must be non-empty
- First message must have `role: "user"`
- Maximum 20 messages total; maximum 8 user turns

**Response:**
```json
{
  "reply": "Based on your requirements, here are my recommendations...",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

**Test Type Codes:**

| Code | Meaning |
|---|---|
| `A` | Ability & Aptitude |
| `K` | Knowledge & Skills |
| `P` | Personality & Behavior |
| `C` | Competencies |
| `B` | Biodata & Situational Judgment |
| `D` | Development & 360 |
| `E` | Assessment Exercises |
| `S` | Simulations |

**Error Responses:**

| Status | Reason |
|---|---|
| `400` | Empty messages, invalid role order, or exceeded 8 user turns |
| `500` | LLM or internal agent error |

---

## Testing

```bash
# Test BM25 retriever only (no API key needed)
python test_retriever.py

# Test API key + LLM connection
python check_api.py

# Full end-to-end API test (server must be running)
python test_api.py
```

---

## Deployment

Deploy to any platform that supports Python (Render, Railway, Fly.io, etc.).

### Production Start Command

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Environment Variables to Set

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | ✅ Yes | Your OpenRouter API key |
| `OPENROUTER_MODEL` | ❌ Optional | Override default model (default: `openai/gpt-oss-20b:free`) |

### Example: Deploy to Render

1. Push repo to GitHub
2. New Web Service → connect repo
3. **Build Command:** `pip install -r requirements.txt`
4. **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add `OPENROUTER_API_KEY` under **Environment Variables**

---

## Design Decisions

### Why BM25 over Vector Embeddings?

The SHL catalog is small (377 items) and **keyword-dense** — job titles and skills like "Java", "Angular", "OPQ32r", "Verify G+" require exact-match recall. BM25 explicitly rewards term frequency + inverse document frequency, making it more reliable than semantic embeddings for this domain. It also has:

- **Zero external dependencies** — pure Python + NumPy, no PyTorch/GPU required
- **Instant cold-start** — index builds in memory on server startup (~50ms)
- **No embedding API costs**

### Why OpenRouter Free Tier + Fallback Chain?

OpenRouter provides access to 24+ free LLMs through a single OpenAI-compatible API. A live parallel benchmark was run against all available free models to find the fastest responding ones:

| Model | Benchmark Time | JSON Quality |
|---|---|---|
| `openai/gpt-oss-20b:free` | ~3.1s ✅ | Perfect |
| `minimax/minimax-m2.5:free` | ~3.2s ✅ | Perfect |
| `openai/gpt-oss-120b:free` | ~4.2s ✅ | Perfect |
| 21 others (DeepSeek, Llama, Gemma, Qwen…) | 429 rate-limited ❌ | — |

The agent automatically cycles through the fallback chain on any 429 or 402 error. No credit card is required.

### Constraint-First Prompting

The system prompt places hard constraints (NEVER, ONLY) at the top, followed by the JSON schema, behavioral guidance, and dynamic catalog context. This ordering ensures constraints receive the highest attention weight from the LLM.

### Double-Pass Validation

Every LLM response goes through:
1. **JSON extraction** — regex-based stripping of markdown fences + brace-depth parsing
2. **Catalog validation** — every recommended `name` and `url` is cross-checked against `catalog.json`. Non-SHL URLs and hallucinated items are silently dropped.

This guarantees **zero hallucination leakage** in the API response.

---

> For detailed design rationale, trade-offs, and evaluation methodology, see [Approach_Document.md](./Approach_Document.md).
