import csv
import sys

def fmt(row, key):
    """One decimal, or N/A for pitchers who could not be scored."""
    val = row.get(key)
    return f"{float(val):.1f}" if val else "N/A"


rows = list(csv.DictReader(open(sys.argv[1])))
if rows:
    # Surprise+ leads: it is the scale that holds up over a single outing.
    # Deception+ is shown alongside but needs a season's worth of pitches.
    print("| Pitcher | Role | Pitches | Surprise+ | Deception+ | Status |")
    print("|---------|------|---------|-----------|------------|--------|")
    for r in rows:
        print(
            f"| {r['pitcher_name']} | {r['role']} | {r['n_pitches_test']} "
            f"| {fmt(r, 'surprise_plus')} | {fmt(r, 'deception_plus')} | {r['status']} |"
        )
else:
    print("No Orioles pitchers threw >= 10 pitches.")
