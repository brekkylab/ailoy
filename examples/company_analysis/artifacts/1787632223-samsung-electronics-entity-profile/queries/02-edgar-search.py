"""
EDGAR lookups for Samsung Electronics.

Run against the read-only mount at <mount>/edgar/.
"""
import json
from collections import Counter

ROOT = "/var/folders/b1/n4bd3bq52gx_g1h4tfbr_s6m0000gn/T/ailoy-company-analysis"

# 1. entityName search "Samsung Electronics" -- full text search over filings,
#    213 hits total, but only 4 distinct CIKs appear as filer/subject.
p1 = f"{ROOT}/edgar/search/entityName/Samsung Electronics/pages/page-001.json"
d1 = json.load(open(p1))
print("Step 1 total hits:", d1["hits"]["total"])
ciks = set()
names = {}
for h in d1["hits"]["hits"]:
    for c, n in zip(h["_source"]["ciks"], h["_source"]["display_names"]):
        ciks.add(c)
        names[c] = n
for c in ciks:
    print(" -", c, names[c])

# 2. entityName search "Samsung Electronics Co Ltd" -- same result set (213
#    hits), confirming these are full-text hits on filing documents that
#    mention the name, not a company-name index lookup.
p2 = f"{ROOT}/edgar/search/entityName/Samsung Electronics Co Ltd/pages/page-001.json"
d2 = json.load(open(p2))
print("\nStep 2 total hits:", d2["hits"]["total"])

# 3. The one CIK registered under Samsung Electronics' own name:
#    0000879316, "SAMSUNG ELECTRONICS CO LTD /FI"
p3 = f"{ROOT}/edgar/by-cik/0000879316/submissions/submissions.json"
d3 = json.load(open(p3))
core = {k: v for k, v in d3.items() if k != "filings"}
print("\nCIK 0000879316 core submission fields:")
print(json.dumps(core, indent=2, ensure_ascii=False))

recent = d3["filings"]["recent"]
forms = Counter(recent["form"])
print("\nForm-type counts across all filings on record:")
for k, v in forms.most_common():
    print(f"  {k}: {v}")
print("Earliest filing date:", min(recent["filingDate"]))
print("Latest filing date:", max(recent["filingDate"]))

# 4. company facts (XBRL) -- not available for this filer; the response is
#    an S3 NoSuchKey XML error rather than a JSON facts document.
p4 = f"{ROOT}/edgar/by-cik/0000879316/facts/facts.json"
raw4 = open(p4).read()
print("\nfacts.json raw response (not JSON facts, an error body):")
print(raw4)
