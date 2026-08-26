"""
GLEIF lookups for Samsung Electronics Co., Ltd.

Run against the read-only mount at
<mount>/gleif/  (mount root omitted per task instructions; see evidence.md
for the paths as read).

Each `read`/`ls` against a `pages/page-NNN.json` path is a live request to the
GLEIF API. This script documents the paths opened, in order, and what they
returned, re-reading the same JSON files from the mount.
"""
import json

ROOT = "/var/folders/b1/n4bd3bq52gx_g1h4tfbr_s6m0000gn/T/ailoy-company-analysis"

# 1. Exact legal name "Samsung Electronics Co., Ltd" narrowed by country=KR
#    -- returned 50 Korean entities named similarly, none of them Samsung
#    Electronics itself (GLEIF stores its legal name in Korean script).
p1 = f"{ROOT}/gleif/search/entity.legalName/Samsung Electronics Co., Ltd/entity.legalAddress.country/KR/pages/page-001.json"
d1 = json.load(open(p1))
print("Step 1 total hits (legalName exact, KR):", d1["meta"]["pagination"]["total"])
print("Samsung Electronics itself present?",
      any("samsung electronics" in r["attributes"]["entity"]["legalName"]["name"].lower()
          for r in d1["data"]))

# 2. Full-text search "Samsung Electronics" -- 81 hits, one page. Confirms
#    the Korean-script legal name record: LEI 9884007ER46L6N7EI764.
p2 = f"{ROOT}/gleif/search/fulltext/Samsung Electronics/pages/page-001.json"
d2 = json.load(open(p2))
print("\nStep 2 total hits (fulltext):", d2["meta"]["pagination"]["total"])
for rec in d2["data"]:
    if rec["attributes"]["lei"] == "9884007ER46L6N7EI764":
        print("Matched GLEIF record for the parent company:")
        print(json.dumps(rec, ensure_ascii=False, indent=2))

# 3. Ownership relationships as reported to GLEIF.
p3 = f"{ROOT}/gleif/search/ownedBy/9884007ER46L6N7EI764/pages/page-001.json"
d3 = json.load(open(p3))
print("\nDirect/ultimate children reported (ownedBy):", d3["meta"]["pagination"]["total"])
for rec in d3["data"]:
    print(" -", rec["attributes"]["lei"], rec["attributes"]["entity"]["legalName"]["name"],
          rec["attributes"]["entity"]["legalAddress"]["country"])

p4 = f"{ROOT}/gleif/search/owns/9884007ER46L6N7EI764/pages/page-001.json"
d4 = json.load(open(p4))
print("\nParent reported (owns):", d4["meta"]["pagination"]["total"])

# 4. Reporting exceptions -- why no parent is listed.
for rel in ["direct-parent-reporting-exception", "ultimate-parent-reporting-exception"]:
    p = f"{ROOT}/gleif/by-lei/9884007ER46L6N7EI764/{rel}/{rel}.json"
    d = json.load(open(p))
    print(f"\n{rel}:", d["data"]["attributes"]["reason"])

# 5. ISINs linked to the LEI.
p5 = f"{ROOT}/gleif/by-lei/9884007ER46L6N7EI764/isins/isins.json"
d5 = json.load(open(p5))
print("\nISINs:", [x["attributes"]["isin"] for x in d5["data"]])
