"""
EDGAR: full-text search for "Samsung Electronics" hits (213 total, capped at first
page) and the submissions.json for CIK 0000879316, the filer entry EDGAR carries for
Samsung's Korean parent (registered as a foreign private issuer under an old
alphabetic-suffix name, no ticker, no current XBRL facts).
Paths read:
  edgar/search/entityName/Samsung Electronics/pages/page-001.json
  edgar/by-cik/0000879316/submissions/submissions.json
  edgar/by-cik/0000879316/facts/facts.json
"""
import json
from collections import Counter

search_path = "/var/folders/b1/n4bd3bq52gx_g1h4tfbr_s6m0000gn/T/ailoy-company-analysis/edgar/search/entityName/Samsung Electronics/pages/page-001.json"
sd = json.load(open(search_path))
print("full text search total hits:", sd["hits"]["total"])
print("first hit display_names:", sd["hits"]["hits"][0]["_source"]["display_names"])

subs_path = "/var/folders/b1/n4bd3bq52gx_g1h4tfbr_s6m0000gn/T/ailoy-company-analysis/edgar/by-cik/0000879316/submissions/submissions.json"
data = json.load(open(subs_path))
print("CIK:", data["cik"])
print("name:", data["name"])
print("EIN (as recorded by EDGAR, not a Korean EIN):", data["ein"])
print("lei field (submissions.json):", data["lei"])
print("stateOfIncorporation:", data["stateOfIncorporation"], "-", data["stateOfIncorporationDescription"])
print("entityType:", data["entityType"])
print("tickers:", data["tickers"], "exchanges:", data["exchanges"])
print("addresses:", data["addresses"])
print("former names:", data["formerNames"])

recent = data["filings"]["recent"]
forms = recent["form"]
dates = recent["filingDate"]
print("total filings listed in 'recent':", len(forms))
print("form type counts:", Counter(forms))
print("earliest filing date:", min(dates), "latest filing date:", max(dates))

facts_path = "/var/folders/b1/n4bd3bq52gx_g1h4tfbr_s6m0000gn/T/ailoy-company-analysis/edgar/by-cik/0000879316/facts/facts.json"
print("facts.json content (expect S3 NoSuchKey => no XBRL company facts filed):")
print(open(facts_path).read())
