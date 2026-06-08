import os
import re
import json
import difflib
from functools import lru_cache
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import quote

from external_tools import enrich_external as _enrich_external_full
from external_tools import wikipedia_search as _wikipedia_search

# ---------------------------
# Routing thresholds
# ---------------------------
REPO_STRONG_SCORE = 0.62     # if top similarity >= this, repo evidence is likely relevant
REPO_MEDIUM_SCORE = 0.52     # mixed territory

MAX_REPO_MATCHES = 10
MAX_REPO_SNIPPET_CHARS = 1400

MAX_EXTERNAL_SEEDS = 8
MAX_EXTERNAL_ITEMS = 20

# Seeds to ignore if we ever fall back to text extraction
STOP_SEEDS = {
    "do", "does", "did", "done", "then", "these", "this", "that", "those",
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "it", "i", "you",
    "we", "they", "he", "she", "them", "our", "your", "my", "me",
}

# ============================================================
# Nova / Bedrock client (text + image via Converse)
# ============================================================

class NovaClient:
    """
    Bedrock Converse client.

    Uses (in this order):
      - env["BEDROCK_MODEL_ID"]
      - env["NOVA_MODEL_ID"]
      - OS env NOVA_MODEL_ID
      - OS env BEDROCK_MODEL_ID

    Region:
      - env["AWS_REGION"] or OS env AWS_REGION or "eu-north-1"
    """
    def __init__(self, env: Optional[Dict[str, Any]] = None):
        env = env or {}
        self.region = env.get("AWS_REGION") or os.environ.get("AWS_REGION", "eu-north-1")

        self.model_id = (
            (env.get("BEDROCK_MODEL_ID") or "").strip()
            or (env.get("NOVA_MODEL_ID") or "").strip()
            or os.environ.get("NOVA_MODEL_ID", "").strip()
            or os.environ.get("BEDROCK_MODEL_ID", "").strip()
        )

        self.max_tokens = int(env.get("NOVA_MAX_TOKENS") or os.environ.get("NOVA_MAX_TOKENS", "900"))
        self.temperature = float(env.get("NOVA_TEMPERATURE") or os.environ.get("NOVA_TEMPERATURE", "0.2"))

        self._client = None
        self._init_error = None
        try:
            import boto3  # type: ignore
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        except Exception as e:
            self._init_error = str(e)

    def available(self) -> bool:
        return bool(self._client and self.model_id)

    @staticmethod
    def _extract_text(resp: Dict[str, Any]) -> str:
        out = resp.get("output", {}).get("message", {}).get("content", []) or []
        for c in out:
            if "text" in c:
                return c["text"]
        return ""

    def generate_text(self, prompt: str) -> str:
        if not self.available():
            raise RuntimeError(f"Nova not configured (model_id missing or client init failed: {self._init_error}).")

        resp = self._client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": self.temperature},
        )
        return self._extract_text(resp)

    def generate_with_image(self, prompt: str, image_bytes: bytes, image_format: str = "png") -> str:
        if not self.available():
            raise RuntimeError(f"Nova not configured (model_id missing or client init failed: {self._init_error}).")

        fmt = (image_format or "png").lower()
        if fmt == "jpg":
            fmt = "jpeg"

        resp = self._client.converse(
            modelId=self.model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"text": prompt},
                    {"image": {"format": fmt, "source": {"bytes": image_bytes}}},
                ],
            }],
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": self.temperature},
        )
        return self._extract_text(resp)


# ============================================================
# Helpers
# ============================================================

def _wiki_url(title: str) -> str:
    t = (title or "").strip().replace(" ", "_")
    return f"https://en.wikipedia.org/wiki/{quote(t)}"

def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _guess_image_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext in {"jpg", "jpeg", "png", "webp"}:
        return ext
    return "png"

def _read_image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def _safe_json_from_text(txt: str) -> Optional[Any]:
    if not txt:
        return None
    txt = txt.strip()
    try:
        return json.loads(txt)
    except Exception:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", txt, re.DOTALL)
    if not m:
        return None
    blob = m.group(1)
    try:
        return json.loads(blob)
    except Exception:
        return None

def _clean_seed(s: str) -> Optional[str]:
    s = _norm_space(s)
    if not s:
        return None
    if len(s) < 3:
        return None
    if s.lower() in STOP_SEEDS:
        return None
    if re.fullmatch(r"[0-9]+", s):
        return None
    return s

def _dedup_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        x = (x or "").strip()
        if not x:
            continue
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def _score_from_match(m: Dict[str, Any]) -> float:
    # Your results show score ~0.45..0.55
    for k in ("score", "similarity", "sim"):
        if k in m and isinstance(m[k], (int, float)):
            return float(m[k])
    # sometimes distance is used (lower is better); we can try to invert if present
    if "distance" in m and isinstance(m["distance"], (int, float)):
        d = float(m["distance"])
        # heuristic: if distance is <=1, treat similarity as (1-d)
        if 0.0 <= d <= 1.0:
            return 1.0 - d
    return 0.0

def _title_from_match(m: Dict[str, Any]) -> str:
    return (
        m.get("pdf_title")
        or m.get("book_title")
        or m.get("title")
        or m.get("source")
        or "Repository item"
    )

def _page_from_match(m: Dict[str, Any]) -> Optional[int]:
    for k in ("page", "page_num", "page_number"):
        v = m.get(k)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
    return None

def _snippet_from_match(m: Dict[str, Any]) -> str:
    return (
        m.get("chunk_current")
        or m.get("chunk")
        or m.get("text")
        or m.get("snippet")
        or ""
    )

def _repo_context(matches: List[Dict[str, Any]], limit: int = 8) -> str:
    lines = []
    for m in (matches or [])[:limit]:
        score = _score_from_match(m)
        title = _title_from_match(m)
        page = _page_from_match(m)
        snip = _snippet_from_match(m)[:260].strip()
        p = f"p.{page}" if page is not None else "p.?"
        lines.append(f"- score={score:.4f} {title} ({p}) :: {snip}")
    return "\n".join(lines)

def _repo_evidence_text(matches: List[Dict[str, Any]], limit: int = 6) -> str:
    blocks = []
    for m in (matches or [])[:limit]:
        title = _title_from_match(m)
        page = _page_from_match(m)
        snip = _snippet_from_match(m).strip()
        snip = snip[:MAX_REPO_SNIPPET_CHARS]
        p = f"{page}" if page is not None else "?"
        blocks.append(f"[{title} — page {p}]\n{snip}\n")
    return "\n".join(blocks).strip()

def _is_image_question(q: str) -> bool:
    """
    This MUST catch 'Do you know what this image is about?' etc.
    """
    ql = (q or "").lower()
    triggers = [
        "image", "picture", "photo", "photograph", "screenshot",
        "logo", "badge", "emblem", "crest", "coat of arms", "flag",
        "what is this", "what's this", "identify", "who is this", "where is this",
        "what is in this", "what is on this",
        "about this image", "about this picture",
    ]
    return any(t in ql for t in triggers)

@lru_cache(maxsize=256)
def _wiki_best_title(seed: str) -> Optional[str]:
    seed = (seed or "").strip()
    if not seed:
        return None
    try:
        hits = _wikipedia_search(seed, limit=5) or []
    except Exception:
        return None
    if not hits:
        return None
    best = hits[0].get("title")
    return best.strip() if isinstance(best, str) else None


# ============================================================
# External enrichment (image-first)
# ============================================================

def _vision_identify(client: NovaClient, question: str, image_path: str) -> Tuple[Dict[str, Any], List[str]]:
    """
    Returns:
      - vision_debug dict
      - seeds list (best_guess + candidates)
    """
    dbg: Dict[str, Any] = {"vision_used": False, "raw": "", "parsed": None}
    if not (client and client.available() and image_path and os.path.exists(image_path)):
        return dbg, []

    img_bytes = _read_image_bytes(image_path)
    img_fmt = _guess_image_format(image_path)

    prompt = (
        "You are identifying what is shown in an uploaded image.\n"
        "Return ONLY JSON.\n\n"
        "Schema:\n"
        "{\n"
        '  "best_guess": "Most specific Wikipedia title",\n'
        '  "candidates": ["up to 8 other plausible Wikipedia titles"],\n'
        '  "type": "one of: logo|crest|flag|person|place|document|other",\n'
        '  "confidence": 0.0,\n'
        '  "notes": "short, optional"\n'
        "}\n\n"
        "Rules:\n"
        "- Be SPECIFIC. If it is a sports team badge, use the team name (e.g., 'Luxembourg national football team'),\n"
        "  NOT just the country name.\n"
        "- If it is a federation logo, use the federation name.\n"
        "- Avoid generic single words.\n\n"
        f"User question: {question}\n"
    )

    try:
        out = client.generate_with_image(prompt, img_bytes, img_fmt)
        dbg["vision_used"] = True
        dbg["raw"] = (out or "")[:2000]
        parsed = _safe_json_from_text(out) or {}
        dbg["parsed"] = parsed
    except Exception as e:
        dbg["error"] = str(e)
        return dbg, []

    seeds: List[str] = []
    if isinstance(dbg["parsed"], dict):
        bg = _clean_seed(str(dbg["parsed"].get("best_guess") or ""))
        if bg:
            seeds.append(bg)
        cands = dbg["parsed"].get("candidates") or []
        if isinstance(cands, list):
            for c in cands[:8]:
                cs = _clean_seed(str(c or ""))
                if cs:
                    seeds.append(cs)

    seeds = _dedup_keep_order(seeds)
    # Validate seeds to canonical wiki titles where possible (helps quality)
    validated: List[str] = []
    for s in seeds[:MAX_EXTERNAL_SEEDS]:
        bt = _wiki_best_title(s)
        validated.append(bt or s)

    validated = _dedup_keep_order(validated)
    return dbg, validated


def _parse_enrich_dict(en: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    external_tools.enrich_external returns dict:
      { seeds_used:[], wiki:[{title, summary:{extract,...}}], wikidata:[...], dropped:[...] }
    We convert into template-friendly lists.
    """
    wiki_items: List[Dict[str, Any]] = []
    wd_items: List[Dict[str, Any]] = []

    for w in (en.get("wiki") or []):
        if not isinstance(w, dict):
            continue
        title = (w.get("title") or "").strip()
        summ = w.get("summary") or {}
        extract = ""
        url = ""
        if isinstance(summ, dict):
            extract = (summ.get("extract") or summ.get("summary") or "").strip()
            url = (summ.get("url") or "").strip()
        if not url and title:
            url = _wiki_url(title)
        if title:
            wiki_items.append({"title": title, "extract": extract, "url": url})

    for wd in (en.get("wikidata") or []):
        if not isinstance(wd, dict):
            continue
        qid = (wd.get("qid") or wd.get("id") or "").strip()
        label = (wd.get("label") or wd.get("title") or "").strip()
        desc = (wd.get("description") or "").strip()
        url = (wd.get("url") or "").strip()
        if not url and qid:
            url = f"https://www.wikidata.org/wiki/{qid}"
        if qid or label:
            wd_items.append({"qid": qid, "label": label, "description": desc, "url": url})

    return wiki_items[:MAX_EXTERNAL_ITEMS], wd_items[:MAX_EXTERNAL_ITEMS]


def external_search_by_image(question: str, image_path: str, env: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (wiki_items, wd_items, debug)
    """
    client = NovaClient(env)
    debug: Dict[str, Any] = {"mode": "image_first", "vision": {}, "seeds_used": [], "enrich": {}}

    vision_dbg, seeds = _vision_identify(client, question, image_path)
    debug["vision"] = vision_dbg
    debug["seeds_used"] = seeds

    if not seeds:
        return [], [], debug

    # Call your existing enrich_external (dict return)
    try:
        en = _enrich_external_full(seeds)
        if not isinstance(en, dict):
            en = {}
    except Exception as e:
        debug["enrich"]["error"] = str(e)
        return [], [], debug

    debug["enrich"]["dict_keys"] = list(en.keys())
    debug["enrich"]["dropped"] = en.get("dropped", [])
    debug["enrich"]["seeds_used"] = en.get("seeds_used", [])

    wiki_items, wd_items = _parse_enrich_dict(en)
    return wiki_items, wd_items, debug


def external_search_by_text(question: str, evidence_text: str, env: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fallback when there is no image (or no vision).
    Keep it conservative to avoid garbage seeds.
    """
    debug: Dict[str, Any] = {"mode": "text_first", "seeds_used": [], "enrich": {}}
    blob = f"{question}\n{evidence_text}"
    # Extract capitalized phrases up to 4 words
    cands = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", blob)
    seeds = []
    for c in cands:
        cs = _clean_seed(c)
        if cs:
            seeds.append(cs)
    seeds = _dedup_keep_order(seeds)[:MAX_EXTERNAL_SEEDS]
    debug["seeds_used"] = seeds

    if not seeds:
        return [], [], debug

    try:
        en = _enrich_external_full(seeds)
        if not isinstance(en, dict):
            en = {}
    except Exception as e:
        debug["enrich"]["error"] = str(e)
        return [], [], debug

    debug["enrich"]["dict_keys"] = list(en.keys())
    wiki_items, wd_items = _parse_enrich_dict(en)
    return wiki_items, wd_items, debug


# ============================================================
# Main agent entrypoint (imported by app.py)
# ============================================================

def run_agent(
    question: str,
    image_path: str = "",
    retriever: Any = None,
    env: Optional[Dict[str, Any]] = None,
    matches: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    env = env or {}
    question = question or ""
    image_path = image_path or ""

    # 1) Get repository matches (if not passed)
    if matches is None and retriever is not None and image_path:
        try:
            top_k = int(env.get("TOP_K", 10))
            matches = retriever.search_image(image_path, top_k=top_k) or []
        except Exception:
            matches = []
    matches = matches or []
    top_matches = matches[:MAX_REPO_MATCHES]

    # 2) Router decision
    repo_scores = [_score_from_match(m) for m in top_matches]
    repo_max = max(repo_scores) if repo_scores else 0.0
    image_q = _is_image_question(question)

    if repo_max >= REPO_STRONG_SCORE and not image_q:
        route = "repository"
    elif image_path and image_q:
        route = "external"
    elif repo_max >= REPO_MEDIUM_SCORE:
        route = "mixed"
    else:
        route = "external" if image_path else "mixed"

    router_debug = {
        "route": route,
        "repo_max_score": repo_max,
        "repo_strong_threshold": REPO_STRONG_SCORE,
        "repo_medium_threshold": REPO_MEDIUM_SCORE,
        "image_question": image_q,
    }

    # 3) Build contexts
    repo_ctx = _repo_context(top_matches, limit=8)
    repo_evidence = _repo_evidence_text(top_matches, limit=6)

    wiki_items: List[Dict[str, Any]] = []
    wd_items: List[Dict[str, Any]] = []
    ext_debug: Dict[str, Any] = {"mode": "none", "seeds_used": []}

    if route in {"external", "mixed"} and image_path:
        wiki_items, wd_items, ext_debug = external_search_by_image(question, image_path, env)
    elif route in {"mixed"}:
        wiki_items, wd_items, ext_debug = external_search_by_text(question, repo_evidence, env)

    # 4) Answer generation (IMPORTANT: include the image so it never says “without seeing it”)
    client = NovaClient(env)
    answer_text = ""
    used = route

    prompt = (
        "You are an agentic RAG assistant answering a question about an uploaded image.\n\n"
        "You have TWO evidence sources:\n"
        "A) Repository matches (visual similarity). These can be unrelated. Treat them as reliable ONLY if the top score is high.\n"
        f"   - top_score = {repo_max:.4f}\n"
        f"   - strong_if >= {REPO_STRONG_SCORE:.2f}\n\n"
        "B) External context (Wikipedia/Wikidata) seeded from image understanding. This can also be wrong if identification is uncertain.\n\n"
        "Your job:\n"
        "- Decide what the image is actually showing.\n"
        "- Use the repository evidence ONLY if it clearly supports the same identity/topic.\n"
        "- If repository matches are weak, say they are likely unrelated and rely on the image + external context.\n"
        "- Do NOT say you cannot see the image.\n"
        "- Keep the answer short and specific.\n\n"
        f"User question:\n{question}\n\n"
        "Repository matches (may be unrelated):\n"
        f"{repo_ctx or '(none)'}\n\n"
        "Repository text excerpts:\n"
        f"{repo_evidence or '(none)'}\n\n"
        "External context (may be unrelated):\n"
        + "\n".join([f"- {w.get('title')}: {w.get('extract','')[:180]}" for w in (wiki_items or [])[:8]])
        + "\n\n"
        "Return format:\n"
        "Answer: <one paragraph>\n"
        "Evidence used: <repository|external|mixed> (one line)\n"
        "Confidence: <low|medium|high> (one word)\n"
    )

    try:
        if client.available() and image_path and os.path.exists(image_path):
            img_bytes = _read_image_bytes(image_path)
            img_fmt = _guess_image_format(image_path)
            answer_text = client.generate_with_image(prompt, img_bytes, img_fmt).strip()
        elif client.available():
            answer_text = client.generate_text(prompt).strip()
        else:
            answer_text = (
                "Nova is not configured. Set BEDROCK_MODEL_ID or NOVA_MODEL_ID.\n"
                f"Router: {router_debug}\n"
            )
    except Exception as e:
        answer_text = f"(Nova error: {e})"

    # 5) Return to templates
    return {
        "answer_text": answer_text,
        "answer": answer_text,

        "top": top_matches,
        "matches": top_matches,

        "entities_detected": ext_debug.get("seeds_used", []),
        "entities": ext_debug.get("seeds_used", []),

        "wikipedia_items": wiki_items,
        "wikipedia": wiki_items,

        "wikidata_items": wd_items,
        "wikidata": wd_items,

        "external_debug": ext_debug,
        "router_debug": router_debug,
    }
