import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
if rows:
    print("| Pitcher | Role | Pitches | Deception+ | Status |")
    print("|---------|------|---------|----------|--------|")
    for r in rows:
        pplus = f"{float(r['deception_plus']):.1f}" if r.get('deception_plus') else "N/A"
        print(f"| {r['pitcher_name']} | {r['role']} | {r['n_pitches_test']} | {pplus} | {r['status']} |")
else:
    print("No Orioles pitchers threw >= 10 pitches.")
