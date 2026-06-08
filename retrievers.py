import os
import re
import numpy as np
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from elasticsearch import Elasticsearch


def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


class RetrieverBundle:
    def __init__(self, env: dict):
        self.env = env

        # Qdrant + ES
        self.qdrant = QdrantClient(host=env["QDRANT_HOST"], port=int(env["QDRANT_PORT"]))
        self.es = Elasticsearch(env["ES_URL"], basic_auth=(env["ES_USER"], env["ES_PASSWORD"]))

        # CLIP
        device = env.get("CLIP_DEVICE", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.processor = CLIPProcessor.from_pretrained(env["CLIP_MODEL_NAME"])
        self.model = CLIPModel.from_pretrained(env["CLIP_MODEL_NAME"]).to(self.device)
        self.model.eval()

    # ---------- CLIP embedding ----------
    def embed_image_file(self, image_path: str) -> np.ndarray:
        pil_img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)  # (1,512)
        vec = feats[0].detach().cpu().numpy().astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec

    # ---------- Qdrant search ----------
    def qdrant_search(self, vec: np.ndarray, top_k: int):
        qvec = vec.tolist()

        # Support both client APIs
        if hasattr(self.qdrant, "query_points"):
            res = self.qdrant.query_points(
                collection_name=self.env["QDRANT_COLLECTION"],
                query=qvec,
                limit=top_k,
                with_payload=True,
            )
            return res.points

        return self.qdrant.search(
            collection_name=self.env["QDRANT_COLLECTION"],
            query_vector=qvec,
            limit=top_k,
            with_payload=True,
        )

    # ---------- ES helpers ----------
    def _book_title_candidates(self, doc_title: str = "", filename: str = "") -> list:
        cands = []

        def add(x: str):
            x = (x or "").strip()
            if x and x not in cands:
                cands.append(x)

        if doc_title:
            add(doc_title)
            add(doc_title + ".pdf")
            add(doc_title.replace(" ", "_"))
            add(doc_title.replace(" ", "_") + ".pdf")
            add(doc_title.replace(" ", "-"))
            add(doc_title.replace(" ", "-") + ".pdf")

        if filename:
            base = os.path.splitext(filename)[0]
            base_cut = re.split(r"(_p\d+|_page\d+|_pg\d+|-p\d+|-page\d+|-pg\d+)", base, maxsplit=1)[0]
            add(base_cut)
            add(base_cut + ".pdf")
            add(base_cut.replace("_", " "))
            add(base_cut.replace("_", " ") + ".pdf")
            add(base_cut.replace("-", " "))
            add(base_cut.replace("-", " ") + ".pdf")
            add(base + ".pdf")
            add(base.replace("_", " ") + ".pdf")

        return cands

    def _es_page_query(self, book_title_candidates: list, page: int, size: int = 8) -> dict:
        should = []
        for t in book_title_candidates:
            if not t:
                continue
            should.append({"term": {"book_title.keyword": t}})
            should.append({"match_phrase": {"book_title": t}})

        return {
            "size": size,
            "_source": ["chunk_text", "book_title", "page_number", "pdf_rel_path"],
            "query": {
                "bool": {
                    "filter": [{"terms": {"page_number": [page, str(page)]}}],
                    "should": should,
                    "minimum_should_match": 1 if should else 0,
                }
            },
        }

    def fetch_page_window_chunks(self, doc_title: str, page: int, filename: str = "", window: int = 1) -> dict:
        out = {
            "prev": "",
            "current": "",
            "next": "",
            "matched_book_title": "",
            "candidates": [],
            "pdf_rel_path": "",
        }

        if page is None or page < 0:
            return out

        candidates = self._book_title_candidates(doc_title or "", filename or "")
        out["candidates"] = candidates

        targets = {"prev": page - window, "current": page, "next": page + window}
        for key, p in targets.items():
            if p is None or p < 0:
                continue

            try:
                body = self._es_page_query(candidates, p, size=8)
                resp = self.es.search(index=self.env["ES_INDEX"], body=body)
                hits = resp.get("hits", {}).get("hits", []) or []

                texts = []
                for h in hits:
                    src = h.get("_source", {}) or {}

                    if not out["matched_book_title"]:
                        out["matched_book_title"] = (src.get("book_title") or "").strip()

                    if not out["pdf_rel_path"]:
                        prp = (src.get("pdf_rel_path") or "").strip()
                        if prp:
                            out["pdf_rel_path"] = prp

                    t = (src.get("chunk_text") or "").replace("\n", " ").strip()
                    if t:
                        texts.append(t)

                merged = " ".join(texts).strip()
                max_chars = int(self.env["MAX_CHARS_PER_PAGE"])
                if len(merged) > max_chars:
                    merged = (merged[:max_chars] + "…").strip()

                out[key] = merged

            except Exception as e:
                out[key] = f"(error fetching text: {e})"

        return out

    # ---------- One-call pipeline used by agent & UI ----------
    def search_image(self, image_path: str, top_k: int):
        vec = self.embed_image_file(image_path)
        points = self.qdrant_search(vec, top_k=top_k)

        results = []
        for p in points:
            payload = getattr(p, "payload", None) or {}

            img_path = payload.get("path")
            filename = payload.get("filename")
            doc_title = payload.get("doc")
            page_i = _safe_int(payload.get("page"))

            chunks = self.fetch_page_window_chunks(
                doc_title=doc_title or "",
                page=page_i,
                filename=filename or "",
                window=int(self.env["PAGE_WINDOW"]),
            )

            results.append({
                "score": float(getattr(p, "score", 0.0)),
                "filename": filename,
                "page": page_i if page_i is not None else payload.get("page"),
                "doc": doc_title,
                "img_path": img_path,
                "chunk_prev": chunks["prev"],
                "chunk_current": chunks["current"],
                "chunk_next": chunks["next"],
                "pdf_rel_path": chunks.get("pdf_rel_path", ""),
                "matched_book_title": chunks.get("matched_book_title", ""),
                "candidates": chunks.get("candidates", []),
            })

        return results
