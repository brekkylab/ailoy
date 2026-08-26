"""
Alphabet Inc. (LEI 5493006MHB84DD0ZWV18): read its own relationship files to find
whether GLEIF discloses a parent above it, and its direct/ultimate children.

Paths read:
  gleif/by-lei/5493006MHB84DD0ZWV18/direct-parent-reporting-exception/direct-parent-reporting-exception.json
  gleif/by-lei/5493006MHB84DD0ZWV18/ultimate-parent-reporting-exception/ultimate-parent-reporting-exception.json
  gleif/by-lei/5493006MHB84DD0ZWV18/direct-children/direct-children.json
  gleif/by-lei/5493006MHB84DD0ZWV18/ultimate-children/ultimate-children.json
  gleif/search/ownedBy/5493006MHB84DD0ZWV18/pages/page-001.json
  gleif/search/owns/5493006MHB84DD0ZWV18/pages/page-001.json
"""
import json

ROOT = "/var/folders/b1/n4bd3bq52gx_g1h4tfbr_s6m0000gn/T/ailoy-company-analysis/"
LEI = "5493006MHB84DD0ZWV18"

def load(rel):
    with open(ROOT + rel) as f:
        return json.load(f)

# 1. Does Alphabet report a parent?
exc = load(f"gleif/by-lei/{LEI}/direct-parent-reporting-exception/direct-parent-reporting-exception.json")
print("direct-parent exception:", exc["data"]["attributes"]["category"], "-", exc["data"]["attributes"]["reason"])

exc2 = load(f"gleif/by-lei/{LEI}/ultimate-parent-reporting-exception/ultimate-parent-reporting-exception.json")
print("ultimate-parent exception:", exc2["data"]["attributes"]["category"], "-", exc2["data"]["attributes"]["reason"])

# 2. owns (who owns Alphabet) via the search index -- should be empty
owns = load(f"gleif/search/owns/{LEI}/pages/page-001.json")
print("search/owns total (entities that own Alphabet):", owns["meta"]["pagination"]["total"])

# 3. ownedBy (who Alphabet owns) via the search index -- flat list, includes
# every entity anywhere below Alphabet in the consolidation chain
owned = load(f"gleif/search/ownedBy/{LEI}/pages/page-001.json")
print("search/ownedBy total (all entities Alphabet ultimately consolidates):", owned["meta"]["pagination"]["total"])

# 4. direct-children (first tier only)
dc = load(f"gleif/by-lei/{LEI}/direct-children/direct-children.json")
print("direct-children total (page 1 of", dc['meta']['pagination']['lastPage'], "pages):", dc["meta"]["pagination"]["total"])

# 5. ultimate-children (GLEIF's own "ultimate-children" relation - should match ownedBy)
uc = load(f"gleif/by-lei/{LEI}/ultimate-children/ultimate-children.json")
print("ultimate-children total:", uc["meta"]["pagination"]["total"])

owned_ids = {e["id"] for e in owned["data"]}
uc_ids_p1 = {e["id"] for e in uc["data"]}
print("ultimate-children page1 subset of ownedBy list:", uc_ids_p1.issubset(owned_ids))
