"""
Query 2: search EDGAR full text index by entity name "NVIDIA" and read the
first results page to recover the CIK behind the hits.

Path walked:
  edgar/search/entityName/NVIDIA/pages/page-001.json
"""
import json

PATH = "edgar/search/entityName/NVIDIA/pages/page-001.json"

with open(
    "/var/folders/b1/n4bd3bq52gx_g1h4tfbr_s6m0000gn/T/ailoy-company-analysis/" + PATH
) as f:
    data = json.load(f)

print("total hits (capped at 10000):", data["hits"]["total"]["value"])
ciks_seen = {}
for hit in data["hits"]["hits"]:
    src = hit["_source"]
    for cik, name in zip(src["ciks"], src["display_names"]):
        ciks_seen.setdefault(cik, name)

for cik, name in ciks_seen.items():
    print(cik, "->", name)
