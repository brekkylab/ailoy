"""
For all 50 entities GLEIF lists under Alphabet's ownedBy/ultimate-children set,
fetch each one's direct-parent (or, where that is withheld, its
direct-parent-reporting-exception + ultimate-parent) to reconstruct which tier
of the tree each sits in, and confirm none of them has its own further
direct-children beyond what is already captured (i.e. the tree bottoms out).

Paths read (one per entity, all under gleif/by-lei/<LEI>/...):
  direct-parent/direct-parent.json                          (when disclosed)
  direct-parent-reporting-exception/direct-parent-reporting-exception.json (when withheld)
  ultimate-parent/ultimate-parent.json                       (to confirm root)
  direct-children/direct-children.json                       (to check for a further tier)
"""
import json, os

ROOT = "/var/folders/b1/n4bd3bq52gx_g1h4tfbr_s6m0000gn/T/ailoy-company-analysis/"
BASE = ROOT + "gleif/by-lei/"

owned_lei_names = {
"98450066EDFC0Y6C2A54":"GOOGLE ARGENTINA S.R.L.",
"984500D8D6D8C77L5B32":"Google Austria GmbH",
"213800WO2QK7HUL8R680":"GOOGLE BELGIUM",
"984500B5170E75C99056":"GOOGLE BRASIL INTERNET LTDA.",
"984500074CCB812A5D80":"GOOGLE CLOUD CANADA CORPORATION",
"984500B1C1A7S4898E77":"Google Switzerland GmbH",
"984500C04A37BB9B5215":"GOOGLE CHILE LIMITADA",
"529900H4SBHKV7XQXK22":"Google Germany GmbH",
"74370096HBTUWJRFIT29":"GLUI Engineering Oy",
"743700HGB1CRPBRA1K29":"Tuike Finland Oy",
"9845003T366A4EB2EE58":"GOOGLE CLOUD FRANCE",
"9845004B3AADDD5FB669":"GOOGLE FRANCE",
"6488U5HN5HSK59195V82":"GOOGLE PAYMENT LIMITED",
"64886BIOSLNV78184624":"GOOGLE UK LIMITED",
"6488R1WLKN077T974P96":"WIZ CLOUD LIMITED",
"9845003AA4F14A013C44":"GOOGLE PAYMENT IRELAND LIMITED",
"98450052CF14CFEB6435":"GOOGLE CLOUD EMEA LIMITED",
"549300YY0W0QP4SZAG19":"GOOGLE EUROPE, MIDDLE EAST AND AFRICA UNLIMITED COMPANY",
"1FN51LDHILXJ06W1QN20":"GOOGLE IRELAND HOLDINGS UNLIMITED COMPANY",
"YYPPRNO5HB304LHFVG31":"GOOGLE IRELAND LIMITED",
"335800LW63MGNBQTB908":"WAYMO IT SERVICES INDIA PRIVATE LIMITED",
"335800VPK652AEFF1O79":"GOOGLE IT SERVICES INDIA PRIVATE LIMITED",
"335800FMIGZ24EJ2XW80":"RAIDEN INFOTECH INDIA PRIVATE LIMITED",
"335800VQ73CQOLRRSK16":"SZS TECH PRIVATE LIMITED",
"335800APEHUAALHEZA12":"MANDIANT CYBERSECURITY PRIVATE LIMITED",
"335800SCE2XZHDWXCG42":"GOOGLE PAYMENT INDIA PRIVATE LIMITED",
"335800WBAET9Q971AT75":"GOOGLE CLOUD INDIA PRIVATE LIMITED",
"3358004GEBDX73EU6859":"GOOGLE INDIA DIGITAL SERVICES PRIVATE LIMITED",
"335800DFHCGEFFZXRA34":"GOOGLE CAPITAL ADVISORS INDIA PRIVATE LIMITED",
"335800FCLGQFJQJGHP87":"GOOGLE INFORMATION SERVICES INDIA PRIVATE LIMITED",
"3358009UB5HS2WQF6Y18":"GOOGLE CONNECT SERVICES INDIA PRIVATE LIMITED",
"335800P2W6AY6E8BEP06":"GOOGLE INDIA PRIVATE LIMITED",
"529900H6FVQK3TP02Z71":"GOOGLE CLOUD ITALY S.R.L.",
"6488JJ7LW4MS85G57298":"Google Payment Lithuania, UAB",
"984500F71C16CDC30C27":"GOOGLE CLOUD MEXICO S DE RL DE CV",
"984500DO4RF63E8D6604":"Google Netherlands B.V.",
"984500E14WF63DAD2C91":"GOOGLE POLAND SP Z O O",
"984500B8CC95C15F8Q65":"GOOGLE CLOUD POLAND SP Z O O",
"549300AYV236IWGM7312":"GOOGLE HOLDINGS PTE. LTD.",
"RXU43ANVI3MGLXA9UI89":"GOOGLE ASIA PACIFIC PTE. LTD.",
"549300KG6S007HTRLN18":"ALPHABET CAPITAL US LLC",
"984500639BT036CND679":"CHARLESTON ROAD REGISTRY INC.",
"984500DI9BZD41OF4781":"GOC INTERNATIONAL LLC",
"9845000BD467666E3930":"Design, LLC",
"9845003D44CCA145CC76":"FIREBASE, INC.",
"6488C2T08TXT755U3P86":"WIZ, INC.",
"549300XRQBENEKTBS153":"GOOGLE ENERGY LLC",
"549300Y7W34DK0WBPY51":"GOOGLE INTERNATIONAL LLC",
"549300SW82SFEORONS31":"ALPHABET CAPITAL US II LLC",
"7ZW8QJWVPR4P1J1KQY45":"GOOGLE LLC",
}

def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None

for lei, name in owned_lei_names.items():
    d = load(f"{BASE}{lei}/direct-parent/direct-parent.json")
    if d:
        parent = d["data"]["attributes"]["entity"]["legalName"]["name"]
        print(f"{name:52s} <- direct parent: {parent}")
    else:
        exc = load(f"{BASE}{lei}/direct-parent-reporting-exception/direct-parent-reporting-exception.json")
        reason = exc["data"]["attributes"]["reason"] if exc else "UNKNOWN"
        up = load(f"{BASE}{lei}/ultimate-parent/ultimate-parent.json")
        upname = up["data"]["attributes"]["entity"]["legalName"]["name"] if up else "?"
        print(f"{name:52s} <- direct parent WITHHELD (reason={reason}); ultimate parent on file: {upname}")

# Check whether any of the 50 has its own direct-children (a further tier)
print("\n-- checking for a further tier below any of the 50 --")
for lei, name in owned_lei_names.items():
    dc = load(f"{BASE}{lei}/direct-children/direct-children.json")
    if dc:
        n = dc["meta"]["pagination"]["total"]
        kids = ", ".join(e["attributes"]["entity"]["legalName"]["name"] for e in dc["data"])
        print(f"{name:52s} has {n} direct child(ren): {kids}")
