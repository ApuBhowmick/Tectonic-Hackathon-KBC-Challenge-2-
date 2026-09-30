"""
Scores predictions.csv against ground_truth.csv.
Run this AFTER engine.py. This is the only script allowed to open ground_truth.csv.

Usage:
  python score.py
"""

import csv
from collections import defaultdict

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    preds = {r["customer_id"]: r for r in load_csv("predictions.csv")}
    truth = {r["customer_id"]: r for r in load_csv("ground_truth.csv")}

    correct = 0
    false_positive = 0
    false_negative = 0
    wrong_type = 0
    total_events = 0
    total = 0

    per_event = defaultdict(lambda: {"correct": 0, "total": 0})

    for cid, t in truth.items():
        if cid not in preds:
            continue
        total += 1
        true_event = t["planted_event_type"]
        pred_event = preds[cid]["predicted_event"]
        acted = preds[cid]["act"] == "True"

        if true_event != "none":
            total_events += 1
            per_event[true_event]["total"] += 1

        if true_event == "none" and pred_event != "none" and acted:
            false_positive += 1
        elif true_event != "none" and (pred_event == "none" or not acted):
            false_negative += 1
        elif true_event != "none" and pred_event != true_event:
            wrong_type += 1
        elif true_event == pred_event:
            correct += 1
            if true_event != "none":
                per_event[true_event]["correct"] += 1

    print(f"Total customers scored: {total}")
    print(f"Planted events: {total_events}")
    print(f"Correct detections (right event, acted): {sum(v['correct'] for v in per_event.values())}/{total_events}")
    print(f"False positives (flagged a 'none' customer): {false_positive}")
    print(f"False negatives (missed a real event): {false_negative}")
    print(f"Wrong event type: {wrong_type}")
    print()
    print("Per event-type recall:")
    for event, v in per_event.items():
        rate = v["correct"] / v["total"] if v["total"] else 0
        print(f"  {event}: {v['correct']}/{v['total']} ({rate:.0%})")


if __name__ == "__main__":
    main()
