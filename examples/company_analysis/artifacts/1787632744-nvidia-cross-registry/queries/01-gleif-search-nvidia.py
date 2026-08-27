"""
Query 1: search GLEIF by legal name "NVIDIA" and read the results page.

Path walked (mounted filesystem, read-only, live):
  gleif/search/entity.legalName/NVIDIA/pages/page-001.json

This is a free descent (naming the filter costs nothing) followed by one
paid read (opening `pages`).
"""
import json

PATH = "gleif/search/entity.legalName/NVIDIA/pages/page-001.json"

with open(
    "/var/folders/b1/n4bd3bq52gx_g1h4tfbr_s6m0000gn/T/ailoy-company-analysis/" + PATH
) as f:
    data = json.load(f)

print("total hits:", data["meta"]["pagination"]["total"])
for rec in data["data"]:
    a = rec["attributes"]
    print("-" * 60)
    print("LEI:", a["lei"])
    print("legalName:", a["entity"]["legalName"]["name"])
    print("category:", a["entity"]["category"])
    print("jurisdiction:", a["entity"]["jurisdiction"])
    print("status:", a["entity"]["status"])
    print("hqAddress:", a["entity"]["headquartersAddress"]["addressLines"],
          a["entity"]["headquartersAddress"]["city"],
          a["entity"]["headquartersAddress"]["region"])
