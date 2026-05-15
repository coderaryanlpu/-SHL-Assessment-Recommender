"""
FastAPI service — SHL Assessment Recommender
Endpoints:
  GET  /health  → {"status": "ok"}
  POST /chat    → {reply, recommendations, end_of_conversation}
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Literal

from agent import run_agent
from retriever import get_index  # pre-warm on startup

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for SHL catalog assessment selection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pre-warm index on startup ─────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    get_index()  # Load BM25 index into memory

# ── Schemas ───────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("messages cannot be empty")
        if len(v) > 20:
            raise ValueError("Too many messages (max 20)")
        # Must start with a user message
        if v[0].role != "user":
            raise ValueError("First message must be from user")
        return v

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Convert pydantic models to plain dicts for agent
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Check turn limit
    user_turns = sum(1 for m in messages if m["role"] == "user")
    if user_turns > 8:
        raise HTTPException(
            status_code=400,
            detail="Conversation exceeded maximum of 8 user turns"
        )

    try:
        result = run_agent(messages)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    return ChatResponse(
        reply=result["reply"],
        recommendations=[
            Recommendation(**r) for r in result.get("recommendations", [])
        ],
        end_of_conversation=result.get("end_of_conversation", False),
    )
