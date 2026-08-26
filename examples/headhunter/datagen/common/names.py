"""Invented people and companies, built by combining common pieces.

Each piece is an ordinary given name, surname, or word, so no combination points at a
real person. The denylist covers what slips through anyway.
"""

import random

# Combinations that would read as real. Only ones the generator can actually produce.
DENYLIST: frozenset[str] = frozenset(
    {
        # real companies
        "Naver", "Kakao", "Coupang", "Toss", "Line", "Samsung", "LG", "SK",
        "Google", "Meta", "Amazon", "Apple", "Microsoft", "Netflix",
        # real people
        "Jensen Huang", "Elon Musk", "Tim Cook", "Satya Nadella",
    }
)

_EN_FIRST = [
    "Alex", "Jordan", "Casey", "Riley", "Morgan", "Avery", "Quinn", "Rowan",
    "Sage", "Blake", "Devon", "Ellis", "Finley", "Harper", "Iris", "Jade",
    "Kai", "Lane", "Nova", "Orion", "Piper", "Reese", "Skyler", "Tatum",
]
_EN_LAST = [
    "Vance", "Whitlock", "Ashby", "Brennan", "Calloway", "Dunmore", "Ellery",
    "Fairbanks", "Grantham", "Hollis", "Ingram", "Jessup", "Kendrick",
    "Lockhart", "Merrick", "Norwood", "Osgood", "Pemberton", "Radcliffe",
    "Sheffield", "Thorne", "Underhill", "Voss", "Westbrook",
]
_KO_FIRST = [
    "지훈", "서연", "민준", "하은", "도윤", "수아", "예준", "지우", "시우", "채원",
    "주원", "은서", "건우", "다은", "현우", "하윤", "지호", "소율", "준서", "예은",
]
_KO_LAST = ["강", "고", "구", "남", "노", "도", "류", "명", "반", "설", "성", "심", "연", "위", "표", "하", "함", "현"]

# Uncommon roots, so a head+tail pair does not land on a real company.
_CO_HEAD = [
    "Finlogic", "Nordwind", "Quantile", "Arborline", "Cindershift", "Draftwell",
    "Emberpath", "Foldgate", "Glasswing", "Halcyon", "Ironvale", "Junipex",
    "Kelpstone", "Larkfield", "Mossbank", "Nightjar", "Oakhelm", "Peatmoor",
]
_CO_TAIL = ["Systems", "Labs", "Works", "Dynamics", "Networks", "Platform", "Technologies", ""]

# Suffixes the name drift adds. Strip these and it is the same company. `variants.py`
# must use the same list; this one is the source.
LEGAL_SUFFIXES: tuple[str, ...] = ("Inc.", "Corporation", "Co., Ltd.", "Ltd.", "LLC")


_JA_FIRST = [
    "陽翔", "結菜", "蓮", "葵", "湊", "凛", "悠真", "咲良", "大翔", "杏",
    "颯太", "芽依", "朝陽", "紬", "陽菜", "樹", "莉子", "翔太", "美咲", "健太",
]
_JA_LAST = [
    "佐藤", "鈴木", "高橋", "田中", "伊藤", "渡辺", "山本", "中村", "小林", "加藤",
    "吉田", "山田", "佐々木", "山口", "松本", "井上", "木村", "林",
]


def person(rng: random.Random, locale: str) -> tuple[str, str]:
    """`(given, family)`. `locale` is `"en"`, `"ko"`, or `"ja"`."""
    if locale == "ko":
        return rng.choice(_KO_FIRST), rng.choice(_KO_LAST)
    if locale == "ja":
        return rng.choice(_JA_FIRST), rng.choice(_JA_LAST)
    return rng.choice(_EN_FIRST), rng.choice(_EN_LAST)


def company(rng: random.Random) -> tuple[str, str]:
    """`(name, urn)`. The urn is derived from the name — see `urn_for`."""
    head = rng.choice(_CO_HEAD)
    tail = rng.choice(_CO_TAIL)
    name = f"{head} {tail}".strip()
    return name, urn_for(name)


def urn_for(name: str) -> str:
    """A stable urn for a company name. **The only evidence that two spellings are one
    company.**

    Two demands pull against each other: `Acme` / `Acme Inc.` / `Acme Corporation` need
    the same urn, while `Nordwind Systems` and `Nordwind Labs` need different ones. So
    only the legal suffix is stripped — `Systems` and `Labs` are part of the name.

    Keying on the first word alone collapses 144 companies into 18 urns, and the
    duplicate-profile trap is judged on "same company_urn, same dates".
    """
    import hashlib

    root = name
    for suffix in LEGAL_SUFFIXES:
        if root.endswith(suffix):
            root = root[: -len(suffix)].rstrip()
            break
    digest = hashlib.sha256(root.lower().encode()).hexdigest()[:8]
    return f"urn:li:organization:{int(digest, 16) % 9_000_000 + 1_000_000}"
