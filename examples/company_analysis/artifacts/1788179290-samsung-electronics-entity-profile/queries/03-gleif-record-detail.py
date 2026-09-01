"""
GLEIF: pull the full LEI record for Samsung Electronics Co., Ltd. (LEI
9884007ER46L6N7EI764), plus its reporting exceptions (why no parent LEI is filed),
ISINs, branches, and direct children.
Paths read:
  gleif/by-lei/9884007ER46L6N7EI764/record/record.json
  gleif/by-lei/9884007ER46L6N7EI764/direct-parent-reporting-exception/direct-parent-reporting-exception.json
  gleif/by-lei/9884007ER46L6N7EI764/ultimate-parent-reporting-exception/ultimate-parent-reporting-exception.json
  gleif/by-lei/9884007ER46L6N7EI764/isins/isins.json
  gleif/by-lei/9884007ER46L6N7EI764/branches/branches.json
  gleif/by-lei/9884007ER46L6N7EI764/direct-children/direct-children.json
  gleif/by-lei/9884007ER46L6N7EI764/ultimate-children/ultimate-children.json
"""
import json

base = "/var/folders/b1/n4bd3bq52gx_g1h4tfbr_s6m0000gn/T/ailoy-company-analysis/gleif/by-lei/9884007ER46L6N7EI764"

rec = json.load(open(f"{base}/record/record.json"))["data"]["attributes"]
print("LEI:", rec["lei"])
print("Legal name:", rec["entity"]["legalName"]["name"], rec["entity"]["legalName"]["language"])
print("Other names:", rec["entity"]["otherNames"])
print("Legal address:", rec["entity"]["legalAddress"])
print("Registered as (biz reg no.):", rec["entity"]["registeredAs"])
print("Registered at (authority id):", rec["entity"]["registeredAt"])
print("Jurisdiction:", rec["entity"]["jurisdiction"])
print("Legal form id:", rec["entity"]["legalForm"])
print("Entity status:", rec["entity"]["status"])
print("Entity creation date:", rec["entity"]["creationDate"])
print("Registration status:", rec["registration"]["status"], rec["registration"]["initialRegistrationDate"], rec["registration"]["lastUpdateDate"])
print("BIC:", rec["bic"])
print("S&P Global company id:", rec["spglobal"])

dpe = json.load(open(f"{base}/direct-parent-reporting-exception/direct-parent-reporting-exception.json"))["data"]["attributes"]
upe = json.load(open(f"{base}/ultimate-parent-reporting-exception/ultimate-parent-reporting-exception.json"))["data"]["attributes"]
print("Direct parent exception:", dpe["category"], dpe["reason"])
print("Ultimate parent exception:", upe["category"], upe["reason"])

isins = json.load(open(f"{base}/isins/isins.json"))["data"]
print("ISIN count:", len(isins), "sample:", isins[0]["attributes"]["isin"])

branches = json.load(open(f"{base}/branches/branches.json"))["data"]
for b in branches:
    a = b["attributes"]
    print("Branch:", a["lei"], a["entity"]["legalName"]["name"], a["entity"]["jurisdiction"])

children = json.load(open(f"{base}/direct-children/direct-children.json"))["data"]
print("Direct children count (this page):", len(children))
for c in children:
    a = c["attributes"]
    print(" -", a["lei"], a["entity"]["legalName"]["name"], a["entity"]["jurisdiction"])

uc = json.load(open(f"{base}/ultimate-children/ultimate-children.json"))
print("Ultimate children total (per GLEIF pagination):", uc["meta"]["pagination"]["total"])
