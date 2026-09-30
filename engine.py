"""
Life-Event Signal Engine (Step 2 + Step 3 + Step 4 from the design doc, simplified).

For each customer: gathers their recent transactions, app events, and Kate
messages, sends them to an LLM to infer a life event + confidence + reasoning,
then applies a small rules layer to decide whether/how to act.

Output: predictions.csv with one row per customer.
NEVER reads ground_truth.csv -- that file is only for scoring afterwards (see score.py).

Setup:
  pip install openai python-dotenv
  Put OPENAI_API_KEY=... in a .env file (never commit it)

Usage:
  python engine.py
  python engine.py --limit 40      # quick test run on the first 40 customers
  python engine.py --only C0001    # just run Sophie, useful for debugging the prompt
"""

import os
import csv
import json
import argparse
from collections import defaultdict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"  # fast + cheap, good enough for structured extraction tonight

SYSTEM_PROMPT = """You are a life-event detection engine for a bank (KBC).
You are given one customer's recent transactions, in-app search/behaviour events,
and messages they sent to the bank's assistant "Kate".

Your job: decide if these signals, TAKEN TOGETHER, suggest the customer is going
through one of these life events: moving_house, new_baby, new_job,
starting_business, retirement_approaching, or none.

Rules:
- No single signal proves anything. A grocery trip or one odd purchase means nothing
  alone. Only flag an event when multiple signals point the same direction.
- NEVER infer an event from demographic fields alone (age, income band, city). Age
  is background context, not evidence. You must be able to point to specific
  transactions, app events, or Kate messages as the reason. For example, do not
  flag "retirement_approaching" just because a customer is older than 55 -- you
  need an explicit signal such as a pension simulation request, a search about
  pensions, a Kate message asking about retirement, or a transfer into pension
  savings.
- Be conservative. If the evidence is weak or ambiguous, use a low confidence score
  or return "none".
- Never infer sensitive events not in the list above (e.g. divorce, illness,
  bereavement) even if the data looks unusual -- if signals suggest something
  sensitive and NOT in the list, return event "none" with a note in reasoning.
- Output ONLY valid JSON, no markdown, matching this exact schema:

{
  "life_event": "moving_house | new_baby | new_job | starting_business | retirement_approaching | none",
  "confidence": <integer 0-100>,
  "stage": "early | in_transition | settled | n/a",
  "reasoning": "<one short sentence citing the specific signals used>",
  "signals_used": ["<short signal labels>"]
}
"""

# Bundles of candidate actions per event type -- Step 3 from the design doc.
# The engine can only pick from this menu; it never invents a product.
ACTION_MENU = {
    "moving_house": [
        {"action": "One-tap address update across bank, insurance, other KBC services", "priority": 1},
        {"action": "Re-quote / transfer home contents insurance to new address", "priority": 1},
        {"action": "Help set up energy & internet at new address", "priority": 2},
        {"action": "Show new monthly cost (rent+utilities) vs income", "priority": 2},
        {"action": "Offer temporary buffer for moving costs (only if balance is low)", "priority": 3},
    ],
    "new_baby": [
        {"action": "Check child benefit / Groeipakket registration status", "priority": 1},
        {"action": "Review or add child savings account", "priority": 2},
        {"action": "Review life/income protection insurance", "priority": 3},
    ],
    "new_job": [
        {"action": "Update employer/income details on file", "priority": 1},
        {"action": "Review salary account benefits eligible at new income level", "priority": 2},
        {"action": "Revisit savings/investment plan given new income", "priority": 3},
    ],
    "starting_business": [
        {"action": "Offer self-employed / business current account", "priority": 1},
        {"action": "Flag possible personal/business spending mix for bookkeeping", "priority": 2},
        {"action": "Introduce business insurance options", "priority": 3},
    ],
    "retirement_approaching": [
        {"action": "Offer full pension simulation with an advisor", "priority": 1},
        {"action": "Review pension savings contribution level", "priority": 2},
    ],
}


def load_table(path, key_field):
    rows = defaultdict(list)
    if not os.path.exists(path):
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row[key_field]].append(row)
    return rows


def format_customer_payload(cid, customer, txns, events, messages):
    lines = []
    lines.append(f"Customer {cid}: age {customer.get('age')}, city {customer.get('city')}, "
                  f"income band {customer.get('income_band')}, "
                  f"products held: {customer.get('products_held')}")
    lines.append("\nRecent transactions:")
    for t in txns[-40:]:  # cap so the prompt stays small
        lines.append(f"  {t['date']} | {t['amount']} | {t['merchant']} | {t['category']} | {t['city']}")
    lines.append("\nApp behaviour events:")
    for e in events:
        lines.append(f"  {e['timestamp']} | {e['event_type']} | {e['detail']}")
    lines.append("\nMessages to Kate:")
    for m in messages:
        lines.append(f"  {m['timestamp']} | \"{m['text']}\"")
    return "\n".join(lines)


def call_llm(payload_text):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": payload_text},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


# Grounding check: the LLM must be able to point to real evidence in the actual
# data, not just assert an event. Each event type needs at least 2 of its
# keywords to literally appear in the customer's transactions/app_events/kate
# messages, or we don't trust the detection -- no matter how confident the LLM
# claims to be. This catches hallucinated detections instead of just hoping the
# prompt prevents them.
EVENT_KEYWORDS = {
    "moving_house": ["moving compan", "verhuizingen", "ikea", "rent deposit",
                      "notary", "change address", "vastgoed", "paint shop",
                      "new energy supplier", "new contract", "furniture"],
    "new_baby": ["babypark", "dreambaby", "pharmacy co-pay", "groeipakket",
                 "maternity", "baby"],
    "new_job": ["new employer", "change employer", "business attire",
                "salary - new employer"],
    "starting_business": ["ondernemingsloket", "company registration",
                           "business setup", "self-employed", "business software"],
    "retirement_approaching": ["pension simulation", "mypension", "pension savings",
                                "retire"],
}


def verify_grounding(event, payload_text):
    """Returns True if the predicted event has real supporting evidence in the
    customer's actual data, not just an LLM assertion."""
    if event == "none":
        return True
    keywords = EVENT_KEYWORDS.get(event, [])
    text = payload_text.lower()
    hits = sum(1 for kw in keywords if kw in text)
    return hits >= 2


def decide_action(result, customer):
    """Step 3 + 4: rules layer on top of the LLM's raw inference."""
    event = result.get("life_event", "none")
    confidence = result.get("confidence", 0)

    if event == "none" or confidence < 75:
        return {"act": False, "channel": "none", "actions": [], "reason_suppressed":
                "confidence below 75% threshold" if event != "none" else "no event detected"}

    candidates = ACTION_MENU.get(event, [])
    top_actions = sorted(candidates, key=lambda a: a["priority"])[:2]

    # crude urgency heuristic for channel choice: moving_house + insurance gap = high urgency
    income_band = customer.get("income_band", "mid")
    if event == "moving_house" and confidence >= 85:
        channel = "voice_note"  # ElevenLabs -- high urgency, high value (uninsured new home)
    elif confidence >= 80:
        channel = "kate_message"  # proactive chat message
    else:
        channel = "passive_card"  # low-key in-app card only

    # Never push sales-y offers if this looks like a financially stressed customer.
    if income_band == "low" and event in ("starting_business", "new_job"):
        top_actions = [a for a in top_actions if "insurance" not in a["action"].lower()]

    return {"act": True, "channel": channel, "actions": [a["action"] for a in top_actions],
            "reason_suppressed": ""}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process the first N customers")
    parser.add_argument("--only", type=str, default=None, help="only process this one customer_id")
    args = parser.parse_args()

    customers = {}
    with open("customers.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            customers[row["id"]] = row

    txns = load_table("transactions.csv", "customer_id")
    events = load_table("app_events.csv", "customer_id")
    messages = load_table("kate_messages.csv", "customer_id")

    ids = list(customers.keys())
    if args.only:
        ids = [args.only]
    elif args.limit:
        ids = ids[: args.limit]

    out_rows = []
    for i, cid in enumerate(ids, 1):
        customer = customers[cid]
        payload = format_customer_payload(cid, customer, txns.get(cid, []),
                                           events.get(cid, []), messages.get(cid, []))
        try:
            result = call_llm(payload)
        except Exception as e:
            print(f"[{i}/{len(ids)}] {cid}: ERROR {e}")
            continue

        # Hard grounding check -- overrides the LLM if it can't be backed by real evidence
        grounded = verify_grounding(result.get("life_event", "none"), payload)
        if not grounded:
            result["life_event"] = "none"
            result["confidence"] = 0
            result["reasoning"] = "Suppressed: claimed event not backed by verifiable evidence in the data"

        decision = decide_action(result, customer)

        out_rows.append({
            "customer_id": cid,
            "predicted_event": result.get("life_event", "none"),
            "confidence": result.get("confidence", 0),
            "stage": result.get("stage", "n/a"),
            "reasoning": result.get("reasoning", ""),
            "signals_used": "; ".join(result.get("signals_used", [])),
            "act": decision["act"],
            "channel": decision["channel"],
            "recommended_actions": " | ".join(decision["actions"]),
            "reason_suppressed": decision["reason_suppressed"],
        })

        print(f"[{i}/{len(ids)}] {cid}: {result.get('life_event')} "
              f"({result.get('confidence')}%) -> act={decision['act']}, channel={decision['channel']}")

    with open("predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nDone. Wrote {len(out_rows)} predictions to predictions.csv")


if __name__ == "__main__":
    main()