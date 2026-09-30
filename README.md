# KBC Life-Event Signal Engine

**Tectonic Hackathon — KBC Challenge**

## What it does

Most banking personalization stops at "recommend a product." This is a proof of
concept for something different: an engine that notices when a customer is
going through a real life event — moving house, a new baby, a new job,
starting a business, retirement approaching — by combining weak signals across
transactions, in-app behaviour, and messages to KBC's assistant "Kate," and
then proactively helps at exactly the right moment, in the right way, for
that specific customer.

No single signal is ever enough. A grocery trip, one odd purchase, a single
search — none of these prove anything on their own. The engine only acts when
multiple independent signals agree, and it always states its confidence and
its reasoning.

## How it works

1. **Synthetic data generator** (`generate_synthetic_data.py`) builds a
   population of 400 fake customers with realistic transaction, app-event, and
   Kate-message histories. 49 of them have a real life event planted inside
   otherwise ordinary noise; the rest are pure everyday activity. The planted
   labels are kept in `ground_truth.csv`, which the engine never sees.
2. **Detection engine** (`engine.py`) sends each customer's actual data to an
   LLM (GPT-4o-mini), asking it to infer a life event, a confidence score, and
   its reasoning — never from demographics like age alone, only from real
   evidence in the data.
3. **Grounding check**: rather than trusting the LLM's claim at face value,
   the engine independently verifies that the claimed event is backed by
   specific keywords actually present in that customer's data (a moving
   company payment, a pension simulation request, etc.). If it isn't, the
   detection is overridden to "no event" regardless of the LLM's stated
   confidence. This caught and eliminated a real false-positive problem during
   development (see "What we learned" below).
4. **Decision + timing rules** (also in `engine.py`) decide whether to act
   (only above 75% confidence), which 1-2 actions to offer (never more), and
   which channel to use — a passive in-app card, a Kate message, or a voice
   note (ElevenLabs) for high-urgency, high-value moments.
5. **Scoring** (`score.py`) compares the engine's output against the hidden
   ground truth to measure real accuracy.
6. **Demo UI** (`dashboard.html`) shows Sophie's scripted moving-house
   scenario with a live signal-timeline and confidence curve, a
   same-event-different-treatment comparison across customers, a working
   feedback loop that visibly recalibrates signal weights, and a population
   dashboard reading the real prediction results.

## Results

On our 400-customer synthetic population (49 with a real planted event, 351
without):

- **49/49 real events correctly detected**
- **0 false positives**
- **0 missed events**
- 100% recall across all 5 event types

## What we learned (and why it matters for a bank)

Our first version of the engine had a real false-positive problem — it
initially flagged around 20% of ordinary customers with events that weren't
there, partly because of a data-generation bug (a customer's rent payment
provider was randomly changing every month in the test data, which looked
exactly like a moving signal) and partly because the LLM would sometimes
assert an event without strong enough evidence.

We fixed this two ways: cleaning the underlying data so it didn't contain
accidental fake signals, and — more importantly — adding an independent
verification layer that checks the LLM's claims against the actual data
before allowing the system to act, rather than trusting its stated confidence
alone. This is the kind of discipline a bank actually needs before trusting
an AI system with customer-facing decisions: verify, don't just trust.

## How to run it

```
pip install -r requirements.txt
```

Create a `.env` file with:
```
OPENAI_API_KEY=your-key-here
```

Then:
```
python generate_synthetic_data.py --n 400 --event-rate 0.12 --seed 42
python engine.py
python score.py
python -m http.server
```

Open `http://localhost:8000/dashboard.html` to view the demo UI.

## What's unfinished

- The UI is a demo/presentation layer, not a production app — Sophie's
  scenario and the customer-contrast examples are scripted for clarity rather
  than pulled live from the engine.
- The decision/timing rules layer is a simplified version of what a
  production system would need (e.g. no real suppression-history tracking
  across sessions).
- Only 5 life-event types are implemented; the design is meant to generalize
  to more as "plugins" on the same engine.
