"""
GLEIF: fulltext search "Samsung Electronics Co Ltd" surfaces the Korean parent entity
(legal name in Korean script, with English alternative-language name), which the
legalName-exact search above missed because GLEIF's legalName field for this LEI is
recorded in Korean, not English.
Path read: gleif/search/fulltext/Samsung Electronics Co Ltd/pages/page-001.json
"""
import json

path = "/var/folders/b1/n4bd3bq52gx_g1h4tfbr_s6m0000gn/T/ailoy-company-analysis/gleif/search/fulltext/Samsung Electronics Co Ltd/pages/page-001.json"
data = json.load(open(path))
print("total:", data["meta"]["pagination"]["total"])
for rec in data["data"]:
    a = rec["attributes"]
    e = a["entity"]
    other = [o["name"] for o in e.get("otherNames", [])]
    print(a["lei"], "|", e["legalName"]["name"], "|", other, "|", e["jurisdiction"], "|", e["status"])
