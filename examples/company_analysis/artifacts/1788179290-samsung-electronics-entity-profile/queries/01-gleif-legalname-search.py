"""
GLEIF: search entity.legalName for "Samsung Electronics" (exact substring per GLEIF
matching rules). Narrows to 47 records covering Samsung Electronics' sales/manufacturing
subsidiaries and one unrelated fund (ProShares Ultra Samsung Electronics).
Path read: gleif/search/entity.legalName/Samsung Electronics/pages/page-001.json
"""
import json

path = "/var/folders/b1/n4bd3bq52gx_g1h4tfbr_s6m0000gn/T/ailoy-company-analysis/gleif/search/entity.legalName/Samsung Electronics/pages/page-001.json"
data = json.load(open(path))
print("total:", data["meta"]["pagination"]["total"])
for rec in data["data"]:
    a = rec["attributes"]
    e = a["entity"]
    print(a["lei"], "|", e["legalName"]["name"], "|", e["jurisdiction"], "|", e["category"], "|", e["status"])
