"""
End-to-end test for the SHL Assessment Recommender API.
Tests: /health, /chat (vague query), /chat (specific multi-turn), off-topic guard.
"""
import urllib.request
import json
import time

BASE = "http://127.0.0.1:8001"

def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return json.loads(r.read())

def banner(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def show(result):
    print(f"  REPLY       : {result['reply'][:200]}")
    recs = result.get("recommendations", [])
    print(f"  RECS COUNT  : {len(recs)}")
    for r in recs:
        print(f"    - [{r['test_type']}] {r['name']}")
        print(f"       {r['url']}")
    print(f"  END_OF_CONV : {result['end_of_conversation']}")

# ── Test 1: Health check ──────────────────────────────────────────────────────
banner("TEST 1: Health check")
try:
    h = get("/health")
    assert h["status"] == "ok", f"Unexpected: {h}"
    print("  ✅ /health → OK")
except Exception as e:
    print(f"  ❌ /health FAILED: {e}")

# ── Test 2: Vague query — should ask clarifying questions ────────────────────
banner("TEST 2: Vague query (expect clarifying questions, no recs)")
try:
    t0 = time.time()
    r = post("/chat", {"messages": [
        {"role": "user", "content": "I need assessments for a new hire"}
    ]})
    elapsed = time.time() - t0
    print(f"  ⏱  {elapsed:.1f}s")
    show(r)
    assert r["reply"], "Empty reply"
    assert r["recommendations"] == [], f"Expected no recs on vague query, got {r['recommendations']}"
    print("  ✅ Vague query handled correctly (no premature recs)")
except Exception as e:
    print(f"  ❌ FAILED: {e}")

# ── Test 3: Specific query — should return recommendations ───────────────────
banner("TEST 3: Specific query (expect recommendations)")
try:
    t0 = time.time()
    r = post("/chat", {"messages": [
        {"role": "user", "content": "I'm hiring a mid-level Java software engineer. Need to assess coding skills and problem solving. Role is backend development at a fintech company."}
    ]})
    elapsed = time.time() - t0
    print(f"  ⏱  {elapsed:.1f}s")
    show(r)
    assert r["reply"], "Empty reply"
    print(f"  {'✅' if r['recommendations'] else '⚠️ '} Recommendations returned: {len(r['recommendations'])}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")

# ── Test 4: Multi-turn refinement ────────────────────────────────────────────
banner("TEST 4: Multi-turn refinement")
try:
    turn1_req = {"messages": [
        {"role": "user", "content": "I need to assess candidates for a sales manager role"}
    ]}
    t0 = time.time()
    r1 = post("/chat", turn1_req)
    print(f"  Turn 1 ({time.time()-t0:.1f}s):")
    show(r1)

    turn2_req = {"messages": [
        {"role": "user", "content": "I need to assess candidates for a sales manager role"},
        {"role": "assistant", "content": r1["reply"]},
        {"role": "user", "content": "Mid-senior level, B2B SaaS sales, need personality and situational judgment tests too"}
    ]}
    t0 = time.time()
    r2 = post("/chat", turn2_req)
    print(f"\n  Turn 2 ({time.time()-t0:.1f}s):")
    show(r2)
    print(f"  {'✅' if r2['recommendations'] else '⚠️ '} Multi-turn recommendations: {len(r2['recommendations'])}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")

# ── Test 5: Off-topic guard ───────────────────────────────────────────────────
banner("TEST 5: Off-topic guard (salary question)")
try:
    t0 = time.time()
    r = post("/chat", {"messages": [
        {"role": "user", "content": "What is a fair salary for a software engineer in India?"}
    ]})
    elapsed = time.time() - t0
    print(f"  ⏱  {elapsed:.1f}s")
    show(r)
    assert r["recommendations"] == [], "Should return no recs for off-topic"
    print("  ✅ Off-topic blocked, no recommendations")
except Exception as e:
    print(f"  ❌ FAILED: {e}")

print("\n" + "="*60)
print("  ALL TESTS COMPLETE")
print("="*60)
