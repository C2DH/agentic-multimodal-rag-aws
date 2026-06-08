import os
import re
import json
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# ---- Fast-fail timeouts so gunicorn never hangs ----
# connect timeout, read timeout
CONNECT_TIMEOUT = float(os.environ.get("EXTERNAL_CONNECT_TIMEOUT", "3.05"))
READ_TIMEOUT = float(os.environ.get("EXTERNAL_READ_TIMEOUT", "8"))
TIMEOUT = (CONNECT_TIMEOUT, READ_TIMEOUT)

USER_AGENT = os.environ.get(
    "EXTERNAL_USER_AGENT",
    "agentic-rag/1.0 (contact: admin@example.com)"
)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})

_TAG_RE = re.compile(r"<.*?>", re.DOTALL)


def _strip_html(s: str) -> str:
    return _TAG_RE.sub("", s or "").strip()


def _chunk(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]


def _get_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safe GET that never raises. Returns {} on errors.
    """
    try:
        r = SESSION.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


# ============================================================
# Wikipedia
# ============================================================

@lru_cache(maxsize=2048)
def wikipedia_search(query: str, limit: int = 5, lang: str = "en") -> List[Dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []

    api = WIKI_API if lang == "en" else f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": int(limit),
        "format": "json",
    }
    data = _get_json(api, params)
    hits = data.get("query", {}).get("search", []) or []
    out = []
    for h in hits:
        out.append({
            "title": h.get("title"),
            "pageid": h.get("pageid"),
            "snippet": _strip_html(h.get("snippet", "")),
        })
    return out


def _wikipedia_bulk_extracts_and_qids(titles: List[str], lang: str = "en") -> Dict[str, Dict[str, Any]]:
    titles = [t.strip() for t in titles if (t or "").strip()]
    if not titles:
        return {}

    api = WIKI_API if lang == "en" else f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "redirects": 1,
        "prop": "extracts|pageprops",
        "exintro": 1,
        "explaintext": 1,
        "titles": "|".join(titles[:50]),
    }
    data = _get_json(api, params)
    pages = (data.get("query", {}) or {}).get("pages", {}) or {}

    out: Dict[str, Dict[str, Any]] = {}
    for _, p in pages.items():
        title = p.get("title")
        if not title:
            continue
        qid = ((p.get("pageprops") or {}).get("wikibase_item")) if isinstance(p.get("pageprops"), dict) else None
        out[title] = {
            "title": title,
            "pageid": p.get("pageid"),
            "extract": (p.get("extract") or "").strip(),
            "qid": qid,
            "url": f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
        }
    return out


@lru_cache(maxsize=2048)
def wikipedia_summary(title: str, lang: str = "en") -> Optional[Dict[str, Any]]:
    title = (title or "").strip()
    if not title:
        return None
    d = _wikipedia_bulk_extracts_and_qids([title], lang=lang)
    # may have redirected title
    if d:
        # return first value
        return list(d.values())[0]
    return None


@lru_cache(maxsize=2048)
def wikipedia_qid_from_title(title: str, lang: str = "en") -> Optional[str]:
    s = wikipedia_summary(title, lang=lang)
    if not s:
        return None
    return s.get("qid")


# ============================================================
# Wikidata
# ============================================================

@lru_cache(maxsize=2048)
def wikidata_search(query: str, limit: int = 10, lang: str = "en") -> List[Dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []
    params = {
        "action": "wbsearchentities",
        "search": query,
        "language": lang,
        "limit": int(limit),
        "format": "json",
    }
    data = _get_json(WIKIDATA_API, params)
    hits = data.get("search", []) or []
    out = []
    for h in hits:
        out.append({
            "qid": h.get("id"),
            "label": h.get("label"),
            "description": h.get("description"),
            "url": f"https://www.wikidata.org/wiki/{h.get('id')}" if h.get("id") else None,
        })
    return out


def wikidata_labels(ids_csv: str, lang: str = "en") -> Dict[str, Dict[str, str]]:
    """
    Bulk labels/descriptions. Accepts "Q1,Q2,..." or "Q1|Q2|..."
    Never raises.
    """
    if not ids_csv:
        return {}
    ids = re.split(r"[,\|\s]+", ids_csv.strip())
    ids = [x for x in ids if x and x.startswith("Q")]
    if not ids:
        return {}

    out: Dict[str, Dict[str, str]] = {}
    for chunk in _chunk(ids, 40):
        params = {
            "action": "wbgetentities",
            "ids": "|".join(chunk),
            "props": "labels|descriptions",
            "languages": lang,
            "format": "json",
        }
        data = _get_json(WIKIDATA_API, params)
        ents = data.get("entities", {}) or {}
        for qid, ent in ents.items():
            lab = ((ent.get("labels") or {}).get(lang) or {}).get("value")
            desc = ((ent.get("descriptions") or {}).get(lang) or {}).get("value")
            if qid:
                out[qid] = {"label": lab or "", "description": desc or ""}
    return out


def wikidata_facts(qid: str, lang: str = "en") -> Optional[Dict[str, Any]]:
    """
    Best-effort facts. Never raises. Returns None on failure.
    """
    qid = (qid or "").strip()
    if not qid.startswith("Q"):
        return None

    params = {
        "action": "wbgetentities",
        "ids": qid,
        "props": "labels|descriptions|claims",
        "languages": lang,
        "format": "json",
    }
    data = _get_json(WIKIDATA_API, params)
    ent = (data.get("entities", {}) or {}).get(qid)
    if not ent:
        return None

    label = ((ent.get("labels") or {}).get(lang) or {}).get("value") or qid
    desc = ((ent.get("descriptions") or {}).get(lang) or {}).get("value") or ""

    # lightweight facts (avoid exploding into tons of extra label lookups)
    claims = ent.get("claims") or {}
    facts: Dict[str, Any] = {}

    def _get_first_string(prop: str) -> Optional[str]:
        arr = claims.get(prop) or []
        if not arr:
            return None
        mainsnak = (arr[0] or {}).get("mainsnak") or {}
        dv = mainsnak.get("datavalue") or {}
        v = dv.get("value")
        if isinstance(v, str):
            return v
        return None

    # P856 official website is usually a URL string
    website = _get_first_string("P856")
    if website:
        facts["official_website"] = website

    return {
        "qid": qid,
        "label": label,
        "description": desc,
        "url": f"https://www.wikidata.org/wiki/{qid}",
        "facts": facts,
    }


# ============================================================
# Enrichment
# ============================================================

def enrich_external(seeds: List[str], lang: str = "en") -> Dict[str, Any]:
    """
    Returns a dict (your code expects this!):
      {seeds_used: [...], wiki: [...], wikidata: [...], dropped: [...]}

    Designed to be fast + safe:
      - one Wikipedia bulk query for up to 50 titles
      - one Wikidata bulk query for the resulting QIDs
      - never blocks for long (short TIMEOUT)
    """
    seeds = [s.strip() for s in (seeds or []) if (s or "").strip()]
    seeds = list(dict.fromkeys(seeds))  # dedup keep order
    seeds = seeds[:50]

    if not seeds:
        return {"seeds_used": [], "wiki": [], "wikidata": [], "dropped": []}

    # Treat seeds as Wikipedia titles (agent validates titles before calling this)
    wiki_map = _wikipedia_bulk_extracts_and_qids(seeds, lang=lang)

    wiki_items: List[Dict[str, Any]] = []
    qids: List[str] = []
    dropped: List[str] = []

    for s in seeds:
        # find exact match or best by normalized title
        found = None
        if s in wiki_map:
            found = wiki_map[s]
        else:
            # fallback: try match by case-insensitive
            for t, v in wiki_map.items():
                if t.lower() == s.lower():
                    found = v
                    break

        if not found:
            dropped.append(s)
            continue

        wiki_items.append({
            "seed": s,
            "title": found.get("title"),
            "url": found.get("url"),
            "extract": found.get("extract") or "",
        })
        if found.get("qid"):
            qids.append(found["qid"])

    qids = list(dict.fromkeys([q for q in qids if q and q.startswith("Q")]))

    # Bulk labels/descriptions
    wd_items: List[Dict[str, Any]] = []
    if qids:
        labmap = wikidata_labels("|".join(qids), lang=lang) or {}
        for q in qids:
            m = labmap.get(q) or {}
            wd_items.append({
                "qid": q,
                "label": m.get("label") or q,
                "description": m.get("description") or "",
                "url": f"https://www.wikidata.org/wiki/{q}",
            })

    return {
        "seeds_used": [w.get("title") for w in wiki_items if w.get("title")],
        "wiki": wiki_items,
        "wikidata": wd_items,
        "dropped": dropped,
    }
