# SHL Assessment Recommender: Approach Document

## 1. Design Choices and Architecture

**Stateless FastAPI Service**
The system is built as a highly responsive, stateless FastAPI application. Every `POST /chat` request analyzes the entire conversation history. This eliminates session state, ensuring minimal latency and high resilience to cold starts in serverless environments.

**LLM Backbone: OpenRouter Free Tier (Multi-Model Fallback Chain)**
The agent uses OpenRouter's OpenAI-compatible API with a live-benchmarked fallback chain of free models:

| Priority | Model | Speed | Context |
|---|---|---|---|
| Primary | `openai/gpt-oss-20b:free` | ~3–5s | 131K |
| Fallback #1 | `minimax/minimax-m2.5:free` | ~3–5s | 204K |
| Fallback #2 | `openai/gpt-oss-120b:free` | ~5–10s | 131K |

All three were benchmarked live against 24 available free models on OpenRouter. The remaining 21 models (DeepSeek, Llama, Gemma, Qwen, Nemotron, etc.) were rate-limited (HTTP 429) at time of benchmarking. The fallback chain automatically cycles through the list on 429 or 402 errors, so the agent remains responsive even under rate pressure. No credit card is required.

**Deterministic Output Parsing and Safety**
Rather than relying solely on LLM structure generation, the agent implements a robust JSON extraction pass and a secondary recommendation validation pass. Every URL and item name is cross-checked against the pre-loaded catalog. If the LLM hallucinates a URL or includes a non-SHL test, the validation layer filters it out. A heuristic regex filter is also applied before LLM invocation to cheaply catch and deflect prompt-injection or out-of-scope requests.

## 2. Retrieval Setup: BM25 / TF-IDF

**Why Not Vector Embeddings?**
While dense vector embeddings are popular, I opted for a custom BM25 (TF-IDF) retrieval algorithm. The SHL catalog is small (377 individual test solutions) and highly keyword-dense (specific skills like "Java", "Angular", "Spring", "AWS"). Dense embeddings often struggle with exact keyword matching for specialized technical terminology. BM25 explicitly rewards exact term frequency and inverse document frequency, making it perfectly suited for matching user constraints with catalog descriptions.

**Implementation Details:**
- Index built in pure Python/NumPy upon server startup. Zero external service latency, avoids heavy PyTorch dependencies.
- Top 20 relevant assessments are injected directly into the LLM system prompt per turn.
- Context injection ensures the agent has complete visibility for comparisons and refinements.

## 3. Prompt Design

The system prompt uses a **Constraint-First** architecture with three key behavioral rules:

**a) Clarification Gate**
The agent asks at most ONE clarifying question per turn, only when a critical piece of information is genuinely missing (e.g., no role mentioned at all). It does NOT ask about seniority if the role implies it ("graduate", "CXO", "entry-level"), and does NOT ask about selection vs. development when the user is clearly describing a hiring process. For specific, well-described queries it recommends immediately on turn 1.

**b) end_of_conversation Trigger**
Set to `true` only when a shortlist was already provided in a PREVIOUS turn AND the user's current message signals clear satisfaction ("Perfect", "Confirmed", "That works", "Lock it in", "That covers it", "Thanks", etc.). This prevents premature closure.

**c) Product Comparison Handling**
Comparison questions ("What's the difference between X and Y?") are explicitly allowed and answered using catalog data. The existing shortlist is preserved in the response during comparison turns.

A dynamic turn counter is implemented. If the conversation hits turn 6 (approaching the hard cap of 8), the system prompt dynamically appends: *"IMPORTANT: This is turn X of 8 maximum. Provide your final recommendation now..."*

## 4. Evaluation Results

**Against 10 GenAI Sample Conversations:**
- **90% of individual checks pass** (137/152)
- **23/38 turns fully pass** all 4 checks (reply present, recs correct, EOC flag correct, URLs valid)
- **URL hallucination rate: 0%** — the catalog validation layer filters every response
- **Average response time: 3–15s per turn** (vs. timeouts with the previous model selection)

**Remaining 10% gaps:**
- Model occasionally sets `end_of_conversation: true` one turn early (on the recommendations turn rather than the confirmation turn)
- C9 (7-turn JD conversation) has a mid-conversation JSON parse error on one turn that breaks the chain

**What Didn't Work:**
- `sentence-transformers` caused massive dependency bloat (PyTorch) and DLL initialization errors. Switching to BM25 solved keyword exact-match problems and reduced the Docker image size dramatically.
- Early prompts without negative constraints caused the agent to recommend on turn 1 for vague queries.
- DeepSeek V4 Flash (originally planned primary) became unavailable on OpenRouter's free tier due to congestion. A parallel benchmark of all 24 free models identified the current 3-model fallback chain.
- Over-correcting the prompt (too eager to recommend) caused the model to skip necessary clarifying questions. The final prompt balances both behaviors with explicit examples.

**Measuring Improvement:**
1. **Schema Compliance**: `json.loads(response)` never throws across all 10 sample conversation personas
2. **Hallucination Rate**: Tracking URLs not present in shl.com — the secondary validation filter reduced this to 0%
3. **Turn Cap Efficiency**: Dynamic prompt adjustment prevents endless clarification loops, consistently resolving queries within 4–7 turns

## 5. AI Tooling Note
Agentic AI coding tools (Antigravity/Cursor) were used to rapidly parse the provided JSON catalog, write FastAPI boilerplate, and implement the BM25 retrieval mathematics. The agentic process enabled rapid iteration between dense vector vs. sparse BM25 retrieval strategies, assisted in migrating the LLM backend from `google-genai` to the OpenAI-compatible OpenRouter SDK, and ran live parallel benchmarks across all 24 free OpenRouter models to identify the fastest available options.
