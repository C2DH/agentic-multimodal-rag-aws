import os
import uuid
import json
import logging
from dotenv import load_dotenv
from flask import Flask, request, render_template, send_from_directory, send_file, abort
from werkzeug.middleware.proxy_fix import ProxyFix

from retrievers import RetrieverBundle
from agent import run_agent

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

PREFIX = "/agentic-rag"


def safe_abs_under_prefix(prefix: str, path: str) -> str:
    prefix = os.path.abspath(prefix)
    path = os.path.abspath(path)
    if not path.startswith(prefix):
        raise ValueError("Path not allowed")
    return path


def safe_abs_pdf(root_dir: str, rel_path: str) -> str:
    root_dir = os.path.abspath(root_dir)
    rel_path = (rel_path or "").lstrip("/")
    abs_path = os.path.abspath(os.path.join(root_dir, rel_path))
    if not abs_path.startswith(root_dir + os.sep) and abs_path != root_dir:
        raise ValueError("Path traversal blocked")
    return abs_path


def load_env():
    return {
        # Qdrant
        "QDRANT_HOST": os.environ.get("QDRANT_HOST", "localhost"),
        "QDRANT_PORT": os.environ.get("QDRANT_PORT", "6333"),
        "QDRANT_COLLECTION": os.environ.get("QDRANT_COLLECTION", "radio_images"),
        "TOP_K": int(os.environ.get("TOP_K", "10")),

        # Elasticsearch
        "ES_URL": os.environ.get("ES_URL", "http://localhost:9200"),
        "ES_USER": os.environ.get("ES_USER", "elastic"),
        "ES_PASSWORD": os.environ.get("ES_PASSWORD", ""),
        "ES_INDEX": os.environ.get("ES_INDEX", "radio_luxembourg_books"),
        "PAGE_WINDOW": int(os.environ.get("PAGE_WINDOW", "1")),
        "MAX_CHARS_PER_PAGE": int(os.environ.get("MAX_CHARS_PER_PAGE", "900")),

        # CLIP
        "CLIP_MODEL_NAME": os.environ.get("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32"),
        "CLIP_DEVICE": os.environ.get("CLIP_DEVICE", "auto"),

        # Files
        "ALLOWED_IMAGE_PREFIX": os.environ.get("ALLOWED_IMAGE_PREFIX", "/home/ec2-user"),
        "PDF_ROOT_DIR": os.environ.get("PDF_ROOT_DIR", os.path.abspath(os.path.join(BASE_DIR, ".."))),

        # Bedrock / Nova
        "AWS_REGION": os.environ.get("AWS_REGION", "eu-north-1"),
        # Works with your agent.py (it checks BEDROCK_MODEL_ID first)
        "BEDROCK_MODEL_ID": os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0"),
        "NOVA_MODEL_ID": os.environ.get("NOVA_MODEL_ID", ""),
    }


def create_app():
    app = Flask(__name__)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

    env = load_env()
    retriever = RetrieverBundle(env)

    # -------------------
    # Routes (both / and /agentic-rag)
    # -------------------

    @app.route("/", methods=["GET"])
    @app.route(f"{PREFIX}/", methods=["GET"])
    def home():
        return render_template("index.html")

    @app.route("/health", methods=["GET"])
    @app.route(f"{PREFIX}/health", methods=["GET"])
    def health():
        return {"ok": True}

    @app.route("/search", methods=["POST"])
    @app.route(f"{PREFIX}/search", methods=["POST"])
    def search():
        if "image" not in request.files:
            return "No image uploaded", 400
        file = request.files["image"]
        if file.filename == "":
            return "Empty filename", 400

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".png", ".jpg", ".jpeg", ".webp", ".gif"]:
            ext = ".png"

        upload_name = f"{uuid.uuid4().hex}{ext}"
        upload_path = os.path.join(UPLOAD_DIR, upload_name)
        file.save(upload_path)

        results = retriever.search_image(upload_path, top_k=env["TOP_K"])
        return render_template("results.html", query_filename=upload_name, results=results)

    @app.route("/ask", methods=["POST"])
    @app.route(f"{PREFIX}/ask", methods=["POST"])
    def ask():
        if "image" not in request.files:
            return "No image uploaded", 400

        question = (request.form.get("question") or "").strip()
        if not question:
            return "Missing question", 400

        file = request.files["image"]
        if file.filename == "":
            return "Empty filename", 400

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".png", ".jpg", ".jpeg", ".webp", ".gif"]:
            ext = ".png"

        upload_name = f"{uuid.uuid4().hex}{ext}"
        upload_path = os.path.join(UPLOAD_DIR, upload_name)
        file.save(upload_path)

        out = run_agent(question=question, image_path=upload_path, retriever=retriever, env=env)

        try:
            app.logger.setLevel(logging.INFO)
            app.logger.info("AGENT_DEBUG %s", json.dumps(out.get("router_debug", {}), ensure_ascii=False))
            app.logger.info("AGENT_DEBUG %s", json.dumps(out.get("external_debug", {}), ensure_ascii=False))
        except Exception:
            pass

        return render_template(
            "answer.html",
            agent_result=out,
            question=question,
            query_filename=upload_name,
        )

    @app.route("/ask-from-query", methods=["POST"])
    @app.route(f"{PREFIX}/ask-from-query", methods=["POST"])
    def ask_from_query():
        question = (request.form.get("question") or "").strip()
        query_filename = (request.form.get("query_filename") or "").strip()
        if not question or not query_filename:
            return "Missing question or query_filename", 400

        upload_path = os.path.join(UPLOAD_DIR, query_filename)
        if not os.path.exists(upload_path):
            return "Query image not found", 404

        out = run_agent(question=question, image_path=upload_path, retriever=retriever, env=env)

        try:
            app.logger.setLevel(logging.INFO)
            app.logger.info("AGENT_DEBUG %s", json.dumps(out.get("router_debug", {}), ensure_ascii=False))
            app.logger.info("AGENT_DEBUG %s", json.dumps(out.get("external_debug", {}), ensure_ascii=False))
        except Exception:
            pass

        return render_template(
            "answer.html",
            agent_result=out,
            question=question,
            query_filename=query_filename,
        )

    @app.route("/uploads/<filename>")
    @app.route(f"{PREFIX}/uploads/<filename>")
    def uploaded_file(filename):
        return send_from_directory(UPLOAD_DIR, filename)

    @app.route("/img")
    @app.route(f"{PREFIX}/img")
    def serve_indexed_image():
        path = request.args.get("path", "")
        if not path:
            return "Missing path", 400

        if not path.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
            return "Not allowed", 403

        try:
            allowed_prefix = env["ALLOWED_IMAGE_PREFIX"]
            abs_path = safe_abs_under_prefix(allowed_prefix, path)
        except Exception:
            abort(403)

        if not os.path.exists(abs_path):
            return "Not found", 404

        directory = os.path.dirname(abs_path)
        fname = os.path.basename(abs_path)
        return send_from_directory(directory, fname)

    @app.route("/pdf")
    @app.route(f"{PREFIX}/pdf")
    def serve_pdf():
        rel = request.args.get("rel", "")
        if not rel:
            return "Missing rel", 400
        if not rel.lower().endswith(".pdf"):
            return "Not allowed", 403

        try:
            abs_path = safe_abs_pdf(env["PDF_ROOT_DIR"], rel)
        except Exception:
            abort(403)

        if not os.path.exists(abs_path):
            return f"PDF not found on disk: {abs_path}", 404

        return send_file(abs_path, mimetype="application/pdf", as_attachment=False)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5010, debug=False, use_reloader=False)
