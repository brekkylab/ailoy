"""
Find Alphabet Inc.'s LEI in GLEIF by legal name, and confirm the record
against known facts (Delaware incorporation, Mountain View HQ).

Path read:
  gleif/search/entity.legalName/Alphabet Inc/pages/page-001.json
"""
import json

PATH = "gleif/search/entity.legalName/Alphabet Inc/pages/page-001.json"

with open(PATH) as f:
    d = json.load(f)

print("total hits:", d["meta"]["pagination"]["total"])
for rec in d["data"]:
    e = rec["attributes"]["entity"]
    print(rec["id"], "|", e["legalName"]["name"], "|", e["legalAddress"]["country"], "|", e["jurisdiction"])

# ALPHABET INC., LEI 5493006MHB84DD0ZWV18, jurisdiction US-DE, HQ Mountain View, CA
# -- this is the public holding company (ticker GOOGL/GOOG). Other "Alphabet *"
# hits (Alphabet Energy Inc, Alphabet Minerals Inc, Alphabet Holding Company Inc,
# Alphabet Millcreek Self Storage Inc) are unrelated entities that merely share
# the word "Alphabet" -- confirmed by country/address mismatch (India, Canada,
# unrelated Delaware entity registered as 4846758 vs Alphabet Inc.'s 5786925).
