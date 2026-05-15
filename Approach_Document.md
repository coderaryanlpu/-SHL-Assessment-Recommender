# SHL Assessment Recommender: Approach Document

## 1. Design Choices and Architecture

**Stateless FastAPI Service**
The system is built as a highly responsive, stateless FastAPI application. Every `POST /chat` request analyzes the entire conversation history. This eliminates session state, ensuring minimal latency and high resilience to cold starts in serverless environments.

**LLM Backbone: DeepSeek V4 Flash via OpenRouter**
`deepseek/deepseek-v4-flash:free` accessed through [OpenRouter](https://openrouter.ai) (OpenAI-compatible API) is the reasoning engine. It offers exceptional speed via OpenRouter's routing infrastructure, critical for maintaining fluid multi-turn conversations within the strict 30-second API timeout. It demonstrates excellent instruction-following, reliably adhering to the rigid JSON output schema mandated by the evaluation harness. OpenRouter's free tier requires no credit card and provides access to multiple capable models (DeepSeek, Gemma, Llama, etc.) as fallbacks via a single `OPENROUTER_MODEL` environment variable.

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

The system prompt uses a **Constraint-First** architecture:
- **Negative Constraints** at the top: "NEVER recommend assessments not in the catalog", "NEVER recommend on the first turn if the query is vague"
- **Output Enforcement**: Strict JSON schema examples
- **Behavioral Guidance**: Specific strategies derived from the sample conversations (e.g., "For leadership roles include OPQ32r", "For senior technical hires include Verify G+")
- **Dynamic Context**: The `catalog_context` appended at the end receives the highest attention weight

A dynamic turn counter is implemented. If the conversation hits turn 6 (approaching the hard cap of 8), the system prompt dynamically appends: *"IMPORTANT: This is turn X of 8 maximum. Provide your final recommendation now..."*

## 4. Evaluation and What Didn't Work

**What Didn't Work:**
- `sentence-transformers` caused massive dependency bloat (PyTorch) and DLL initialization errors. Switching to BM25 solved keyword exact-match problems and reduced the Docker image size dramatically.
- Early prompts without negative constraints caused the agent to recommend on turn 1 for vague queries. Adding constraint-first ordering resolved this.

**Measuring Improvement:**
1. **Schema Compliance**: Asserting `json.loads(response)` never throws across all 10 sample conversation personas
2. **Hallucination Rate**: Tracking URLs not present in shl.com — the secondary validation filter reduced this to 0%
3. **Turn Cap Efficiency**: Dynamic prompt adjustment prevents endless clarification loops, consistently resolving queries within 4-6 turns

## 5. AI Tooling Note
Agentic AI coding tools (Antigravity/Cursor) were used to rapidly parse the provided JSON catalog, write FastAPI boilerplate, and implement the BM25 retrieval mathematics. The agentic process enabled rapid iteration between dense vector vs. sparse BM25 retrieval strategies, and assisted in migrating the LLM backend from `google-genai` to the OpenAI-compatible OpenRouter SDK.
