"""
Query 3: read the full submissions record for CIK 0001045810 (NVIDIA CORP,
found via query 2) to get registrant-level facts: EIN, addresses, former
names, state of incorporation, tickers, and confirm the `lei` field is null.

Path walked:
  edgar/by-cik/0001045810/submissions/submissions.json
"""
import json

PATH = "edgar/by-cik/0001045810/submissions/submissions.json"

with open(
    "/var/folders/b1/n4bd3bq52gx_g1h4tfbr_s6m0000gn/T/ailoy-company-analysis/" + PATH
) as f:
    data = json.load(f)

fields = [
    "cik", "name", "tickers", "exchanges", "ein", "lei",
    "stateOfIncorporation", "sic", "sicDescription", "category",
    "fiscalYearEnd", "formerNames", "addresses",
]
for k in fields:
    print(k, ":", data.get(k))
