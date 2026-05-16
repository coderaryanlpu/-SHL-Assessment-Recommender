"""
Replay all 10 GenAI Sample Conversations against the live API at port 8001.
Scores each turn on:
  - reply_present      : agent gave a non-empty reply
  - recs_correct       : recommendations present/absent as expected
  - eoc_correct        : end_of_conversation flag matches expected
  - url_valid          : every recommended URL contains shl.com
Prints a per-turn breakdown and a final scorecard.
"""
import urllib.request, urllib.error, json, time, sys

BASE = "http://127.0.0.1:8004"
TIMEOUT = 90  # seconds per call

# ── Helpers ───────────────────────────────────────────────────────────────────

def call_chat(messages: list) -> dict:
    data = json.dumps({"messages": messages}).encode()
    req = urllib.request.Request(
        BASE + "/chat", data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())

def score_turn(result, expect_recs: bool, expect_eoc: bool):
    """Return (pass_count, total, issues)."""
    issues = []
    score = 0
    total = 4

    # 1. Reply present
    if result.get("reply", "").strip():
        score += 1
    else:
        issues.append("EMPTY REPLY")

    # 2. Recs correct
    recs = result.get("recommendations", [])
    has_recs = len(recs) > 0
    if has_recs == expect_recs:
        score += 1
    else:
        issues.append(f"RECS: expected {'some' if expect_recs else 'none'}, got {len(recs)}")

    # 3. EOC correct
    eoc = result.get("end_of_conversation", False)
    if eoc == expect_eoc:
        score += 1
    else:
        issues.append(f"EOC: expected {expect_eoc}, got {eoc}")

    # 4. All URLs valid (only check when recs expected)
    if expect_recs and recs:
        bad_urls = [r["url"] for r in recs if "shl.com" not in r.get("url", "")]
        if not bad_urls:
            score += 1
        else:
            issues.append(f"BAD URLS: {bad_urls}")
    else:
        score += 1  # N/A — auto-pass

    return score, total, issues

def sep(char="─", n=65):
    print(char * n)

# ── Conversations ─────────────────────────────────────────────────────────────
# Each conversation is a list of turns.
# Each turn: (user_msg, expect_recs: bool, expect_eoc: bool)

CONVERSATIONS = {
    "C1 – Senior Leadership (OPQ32r)": [
        ("We need a solution for senior leadership.", False, False),
        ("The pool consists of CXOs, director-level positions; people with more than 15 years of experience.", False, False),
        ("Selection — comparing candidates against a leadership benchmark.", True, False),
        ("Perfect, that's what we need.", True, True),
    ],
    "C2 – Senior Rust Engineer": [
        ("I'm hiring a senior Rust engineer for high-performance networking infrastructure. What assessments should I use?", False, False),
        ("Yes, go ahead. Should I also add a cognitive test for this level?", True, False),
        ("That works. Thanks.", True, True),
    ],
    "C3 – Contact Centre Agents (500 entry-level)": [
        ("We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus. What should we use?", False, False),
        ("English.", False, False),
        ("US.", True, False),
        ("Is the Contact Center Call Simulation different from the Customer Service Phone Simulation?", False, False),
        ("Perfect — new simulation for volume, old solution for finalists. Confirmed.", True, True),
    ],
    "C4 – Graduate Financial Analysts": [
        ("Hiring graduate financial analysts — final-year students, no work experience. We need numerical reasoning and a finance knowledge test.", True, False),
        ("Good. Can you also add a situational judgement element — work-context decision making for graduates?", True, False),
        ("That covers it. Numerical + Graduate Scenarios as first filter, domain tests for shortlisted candidates.", True, True),
    ],
    "C5 – Sales Org Reskilling": [
        ("As part of our restructuring and annual talent audit, we need to re-skill our Sales organization. What solutions do you recommend?", True, False),
        ("What's the difference between OPQ and OPQ MQ Sales Report?", True, False),
        ("Clear. We'll use OPQ for everyone and add MQ only where we want motivators in the Sales Report; keeping the five solutions as our audit stack.", True, True),
    ],
    "C6 – Chemical Plant Operators": [
        ("We're hiring plant operators for a chemical facility. Safety is absolute top priority — reliability, procedure compliance, never cutting corners. What do you recommend?", True, False),
        ("What's the difference between the DSI and the Safety & Dependability 8.0?", False, False),
        ("We're industrial. The 8.0 bundle is the right fit. Confirmed.", True, True),
    ],
    "C7 – Bilingual Healthcare Admin (South Texas)": [
        ("We're hiring bilingual healthcare admin staff in South Texas — they handle patient records and need to be assessed in Spanish. HIPAA compliance is critical. What assessments work?", False, False),
        ("They're functionally bilingual — English fluent for written work. Go with the hybrid.", True, False),
        ("Are we legally required under HIPAA to test all staff who touch patient records? And does this SHL test satisfy that requirement?", False, False),
        ("Understood. Keep the shortlist as-is.", True, True),
    ],
    "C8 – Admin Assistants (Excel & Word)": [
        ("I need to quickly screen admin assistants for Excel and Word daily.", True, False),
        ("In that case, I am OK with adding a simulation - we want to capture the capabilities.", True, False),
        ("That's good.", True, True),
    ],
    "C9 – Senior Full-Stack Engineer (JD paste)": [
        ('Here\'s the JD for an engineer we need to fill. Can you recommend an assessment battery?\n\n"Senior Full-Stack Engineer — 5+ years across Core Java, Spring, REST API design, Angular, SQL/relational databases, AWS deployment, and Docker. Will own end-to-end microservice delivery, contribute to architectural decisions, and mentor mid-level engineers. Strong CI/CD and cloud-native experience required."', False, False),
        ("Backend-leaning. Day-one priorities are Core Java and Spring; SQL is constant. Angular is occasional — they'd review frontend PRs but not own features.", False, False),
        ("Senior IC. They lead design on their own services but don't manage other engineers directly.", True, False),
        ("Add AWS and Docker. Drop REST — the API design signal will already come through in Spring and the live interview.", True, False),
        ("On Java — they'd be working on existing services, not greenfield. Is the Advanced level the right pick?", True, False),
        ("Do we really need Verify G+ on top of all the technical tests? Feels redundant.", True, False),
        ("Keep Verify G+. Locking it in.", True, True),
    ],
    "C10 – Graduate Management Trainee Battery": [
        ("We run a graduate management trainee scheme. We need a full battery — cognitive, personality, and situational judgement. All recent graduates.", True, False),
        ("But can you remove the OPQ32r and replace it with something shorter? Candidates complain it takes too long.", False, False),
        ("Drop the OPQ. Final list: Verify G+ and Graduate Scenarios.", True, True),
    ],
}

# ── Runner ────────────────────────────────────────────────────────────────────

total_turns = 0
passed_turns = 0
total_checks = 0
passed_checks = 0
failed_convs = []

for conv_name, turns in CONVERSATIONS.items():
    sep("═")
    print(f"  {conv_name}")
    sep("═")

    messages = []
    conv_ok = True

    for t_idx, (user_msg, expect_recs, expect_eoc) in enumerate(turns, 1):
        messages.append({"role": "user", "content": user_msg})
        t0 = time.time()
        try:
            result = call_chat(messages)
            elapsed = time.time() - t0
            score, total, issues = score_turn(result, expect_recs, expect_eoc)
            passed_checks += score
            total_checks += total
            total_turns += 1

            status = "PASS" if not issues else "WARN"
            if not issues:
                passed_turns += 1
            else:
                conv_ok = False

            recs = result.get("recommendations", [])
            eoc = result.get("end_of_conversation", False)
            reply_preview = result.get("reply", "")[:100].replace("\n", " ")

            print(f"  Turn {t_idx} [{status}] {elapsed:.1f}s  recs={len(recs)}  eoc={eoc}")
            print(f"    reply: {reply_preview}...")
            if recs:
                for r in recs:
                    print(f"    [{r.get('test_type','?')}] {r.get('name','?')}")
            if issues:
                for iss in issues:
                    print(f"    ⚠ {iss}")

            # Append agent reply to message history for next turn
            messages.append({"role": "assistant", "content": result.get("reply", "")})

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  Turn {t_idx} [FAIL] {elapsed:.1f}s  ERROR: {e}")
            conv_ok = False
            total_turns += 1
            total_checks += 4  # 4 checks all failed
            break

    if not conv_ok:
        failed_convs.append(conv_name)
    print()

# ── Summary ───────────────────────────────────────────────────────────────────
sep("═")
print("  FINAL SCORECARD")
sep("═")
print(f"  Conversations  : {len(CONVERSATIONS) - len(failed_convs)} / {len(CONVERSATIONS)} fully passed")
print(f"  Turns passed   : {passed_turns} / {total_turns}")
print(f"  Checks passed  : {passed_checks} / {total_checks}  ({100*passed_checks//total_checks}%)")
if failed_convs:
    print(f"\n  Failed conversations:")
    for f in failed_convs:
        print(f"    - {f}")
sep("═")
