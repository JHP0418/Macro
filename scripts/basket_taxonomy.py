from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GAPS_EXTRACTED = ROOT / "data" / "gaps_etf_list_2026_05_09_extracted.csv"

BASKET_ORDER = [
    "해외지수",
    "해외섹터",
    "국내지수",
    "국내섹터",
    "FX및 원자재",
    "국내채권_종합",
    "국내채권_회사채",
    "해외채권_종합",
    "해외채권_회사채",
    "금리연계형 및 초단기채권",
]

OVERSEAS_INDEX_GROUPS = {
    "US broad equity",
    "US growth",
    "Global/Developed equity",
    "China/HK growth",
    "China equity",
    "India/EM",
    "Japan equity",
}

OVERSEAS_SECTOR_GROUPS = {
    "US semiconductor",
    "US dividend/defensive",
    "US cyclical/sector",
    "US REIT",
}

DOMESTIC_INDEX_GROUPS = {"Korea broad equity", "Korea growth"}
DOMESTIC_SECTOR_GROUPS = {"Korea semiconductor", "Korea IT", "Korea cyclical", "Korea value", "Korea defensive"}
FX_COMMODITY_GROUPS = {"FX cash", "Gold", "Commodity/Oil", "Oil"}
SHORT_RATE_GROUPS = {"Cash/short bonds"}
OVERSEAS_CORP_BOND_GROUPS = {"US IG bonds", "US high yield"}
OVERSEAS_AGG_BOND_GROUPS = {"US long bonds"}


def load_gaps_name_map(path: Path = GAPS_EXTRACTED) -> dict[str, str]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "code" not in frame.columns or "name" not in frame.columns:
        return {}
    out: dict[str, str] = {}
    for _, row in frame.iterrows():
        code = normalize_code(str(row.get("code", "")))
        name = str(row.get("name", "")).strip()
        if code and name:
            out[code] = name
            out[f"{code}.KS"] = name
    return out


def normalize_code(value: str) -> str:
    text = value.strip().upper()
    if text.startswith("A"):
        text = text[1:]
    if text.endswith(".KS"):
        text = text[:-3]
    return text


def enrich_asset_name(symbol: str, fallback: str, name_map: dict[str, str] | None = None) -> str:
    mapping = name_map if name_map is not None else load_gaps_name_map()
    code = normalize_code(symbol)
    return mapping.get(code, mapping.get(symbol, fallback))


def classify_basket(group: str, name: str = "", symbol: str = "") -> str:
    group = str(group or "").strip()
    name = str(name or "")
    if group in OVERSEAS_INDEX_GROUPS:
        return "해외지수"
    if group in OVERSEAS_SECTOR_GROUPS:
        return "해외섹터"
    if group in DOMESTIC_INDEX_GROUPS:
        return "국내지수"
    if group in DOMESTIC_SECTOR_GROUPS:
        return "국내섹터"
    if group in FX_COMMODITY_GROUPS:
        return "FX및 원자재"
    if group in SHORT_RATE_GROUPS:
        return "금리연계형 및 초단기채권"
    if group in OVERSEAS_CORP_BOND_GROUPS:
        return "해외채권_회사채"
    if group in OVERSEAS_AGG_BOND_GROUPS:
        if any(token in name for token in ["회사채", "하이일드", "우량회사"]):
            return "해외채권_회사채"
        return "해외채권_종합"
    if group == "Korea bonds":
        if any(token in name for token in ["회사채", "금융채"]):
            return "국내채권_회사채"
        if any(token in name for token in ["단기", "CD", "KOFR", "머니마켓", "통안"]):
            return "금리연계형 및 초단기채권"
        return "국내채권_종합"
    return "기타"


def basket_rank_key(basket: str) -> int:
    try:
        return BASKET_ORDER.index(str(basket))
    except ValueError:
        return len(BASKET_ORDER)
