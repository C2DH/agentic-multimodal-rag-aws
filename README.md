# Agentic Multimodal RAG System for Radio Luxembourg Archive Retrieval

A cost-effective **Agentic Multimodal Retrieval-Augmented Generation (RAG)** prototype for searching and answering questions over the **Radio Luxembourg archive collection**. The system combines image-based retrieval, page-level text retrieval, optional external knowledge enrichment, and answer generation through Amazon Bedrock / Amazon Nova.

> This repository is based on a technical project report and supporting implementation notebooks. The system was designed as a practical archive-retrieval prototype rather than a production-ready commercial service.

---

## Project Overview

Historical archive collections often contain a mixture of scanned pages, photographs, logos, tables, advertisements, captions, and extracted text. Traditional keyword search is not enough when a user starts from an image or when document metadata is incomplete.

This project addresses that problem using a multimodal RAG pipeline:

1. Cropped images from Radio Luxembourg PDF documents are embedded with **CLIP ViT-B/32**.
2. Image vectors are stored in **Qdrant** for visual similarity search.
3. Extracted PDF text is indexed in **Elasticsearch** for page-level contextual retrieval.
4. A Flask web app allows users to upload an image and ask a question.
5. An agentic routing layer decides whether to answer using:
   - local archive evidence,
   - external Wikipedia/Wikidata evidence,
   - or a mixture of both.
6. Final answer generation is performed through **Amazon Bedrock** using **Amazon Nova**.

---

## Main Features

- Image upload and visual similarity search
- CLIP-based image embedding generation
- Qdrant vector database integration
- Elasticsearch page-window retrieval
- Agentic routing between local and external evidence
- Wikipedia and Wikidata enrichment
- Amazon Bedrock / Amazon Nova answer generation
- Flask-based web interface
- Notebook-based ingestion pipeline
- Evaluation using a manually reviewed 100-question image-based test set

---

## Repository Structure

| File | Purpose |
|---|---|
| `app.py` | Flask web application, routes, upload handling, PDF/image serving, and API entry points. |
| `agent.py` | Agentic reasoning logic, routing thresholds, Nova client, repository/external evidence handling, and final answer generation. |
| `retrievers.py` | CLIP image embedding, Qdrant visual search, Elasticsearch page-window retrieval, and combined retrieval pipeline. |
| `external_tools.py` | Wikipedia and Wikidata search/enrichment helpers. |
| `Extracting_Images_InsertingData.ipynb` | Notebook for extracting/cropping images and preparing ingestion data. |
| `Qdrant.ipynb` | Notebook for generating image vectors and inserting them into Qdrant. |
| `Append.ipynb` | Notebook for appending/indexing text chunks into Elasticsearch. |
Recommended cleaned file names before publishing:

```text
agent.py
app.py
external_tools.py
retrievers.py
notebooks/Extracting_Images_InsertingData.ipynb
notebooks/Qdrant.ipynb
notebooks/Append.ipynb
docs/report.pdf
```

---

## Architecture

The system uses two complementary retrieval stores:

| Store | Content | Retrieval Type |
|---|---|---|
| **Qdrant** | 512-dimensional CLIP image embeddings and image metadata | Visual similarity search |
| **Elasticsearch** | Extracted page-level text chunks and document metadata | Lexical/contextual retrieval |

At query time, the user uploads an image and asks a question. The image is converted into a CLIP vector and searched against Qdrant. The matched image metadata is then used to retrieve related text from Elasticsearch. The agent evaluates the available evidence and optionally enriches the answer using Wikipedia/Wikidata before calling Amazon Nova through Bedrock.

---

## Technology Stack

| Component | Technology |
|---|---|
| Web application | Flask |
| Image embedding model | `openai/clip-vit-base-patch32` |
| Vector dimension | 512 |
| Vector database | Qdrant |
| Text search engine | Elasticsearch |
| Multimodal / LLM model | Amazon Nova via Amazon Bedrock |
| External knowledge | Wikipedia API and Wikidata API |
| Development environment | Jupyter Notebook |
| Deployment environment | AWS EC2, Amazon Linux 2023 |
| Programming language | Python |

---

## Environment Variables

Create a `.env` file locally or configure the variables in your server environment.

```bash
# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=radio_images
TOP_K=10

# Elasticsearch
ES_URL=http://localhost:9200
ES_USER=elastic
ES_PASSWORD=CHANGE_ME
ES_INDEX=radio_luxembourg_books
PAGE_WINDOW=1
MAX_CHARS_PER_PAGE=900

# CLIP
CLIP_MODEL_NAME=openai/clip-vit-base-patch32
CLIP_DEVICE=auto

# Files
ALLOWED_IMAGE_PREFIX=/home/ec2-user
PDF_ROOT_DIR=/home/ec2-user/rag-project

# AWS Bedrock / Nova
AWS_REGION=eu-north-1
BEDROCK_MODEL_ID=amazon.nova-lite-v1:0
NOVA_MODEL_ID=
NOVA_MAX_TOKENS=900
NOVA_TEMPERATURE=0.2
```

Do **not** commit the real `.env` file to GitHub.

---

## Installation

The project was tested on an AWS EC2 instance running Amazon Linux 2023.

```bash
sudo yum update -y
sudo yum install -y git python3 python3-pip unzip

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install flask python-dotenv pillow numpy torch torchvision
pip install transformers qdrant-client elasticsearch boto3 requests
```

You also need running Qdrant and Elasticsearch services. The original deployment also used Jupyter Notebook for ingestion/evaluation workflows and Nginx as a reverse proxy.

---

## Running the Application

Start the Flask app:

```bash
python app.py
```

The application listens on:

```text
http://0.0.0.0:5010
```

`0.0.0.0` means the Flask server listens on all available network interfaces of the VM. External access still depends on firewall rules, EC2 security groups, and reverse-proxy configuration.

The deployed project also supported the URL prefix:

```text
/agentic-rag
```

---

## Main Routes

| Route | Purpose |
|---|---|
| `/` and `/agentic-rag/` | Render home page |
| `/health` | Health check |
| `/search` | Upload an image and return visually similar repository images |
| `/ask` | Upload an image and question, then run the agentic RAG workflow |
| `/ask-from-query` | Ask a new question using a previously uploaded query image |
| `/uploads/<filename>` | Serve uploaded query images |
| `/img?path=...` | Safely serve indexed images |
| `/pdf?rel=...` | Safely serve referenced PDF files |

---

## Data Ingestion Pipeline

The ingestion workflow prepares two stores:

### Image ingestion into Qdrant

1. Read PDF or scanned document pages.
2. Detect or manually crop relevant images.
3. Save cropped images to disk.
4. Generate a CLIP image embedding for each cropped image.
5. Normalize the vector.
6. Insert the vector and metadata payload into Qdrant.

Expected Qdrant payload:

```json
{
  "path": "/path/to/cropped/image.png",
  "filename": "cropped_image_name.png",
  "page": 12,
  "doc": "Document or book title"
}
```

### Text ingestion into Elasticsearch

1. Extract text from PDF pages or OCR output.
2. Clean text at page level.
3. Create chunks.
4. Store `book_title`, `page_number`, `chunk_text`, and `pdf_rel_path`.
5. Insert records into Elasticsearch.

Expected Elasticsearch document:

```json
{
  "book_title": "Document title",
  "page_number": 12,
  "chunk_text": "Text extracted from the page...",
  "pdf_rel_path": "relative/path/to/document.pdf"
}
```

---

## Agentic Routing Logic

The routing layer evaluates the quality of local repository evidence and decides how the answer should be generated.

| Condition | Route | Interpretation |
|---|---|---|
| Strong repository match, usually score >= 0.62 | Repository | Mainly use local archive evidence from Qdrant and Elasticsearch |
| Medium repository match, usually score >= 0.52 | Mixed | Combine local evidence with external evidence when useful |
| Image-identification or public-knowledge question | External or Mixed | Use Amazon Nova, Wikipedia, and Wikidata for enrichment |
| Weak or unreliable repository match | External | Avoid over-trusting weak local matches |

These thresholds were selected empirically during development and should be tuned for larger datasets.

---

## Evaluation Summary

The system was evaluated using a manually reviewed dataset of 100 image-based questions.

| Evidence Group | Correct | Partial | Incorrect | Strict Accuracy | Weighted Accuracy |
|---|---:|---:|---:|---:|---:|
| Local Repository | 30 | 28 | 12 | 42.86% | 62.86% |
| Wikipedia / Wikidata | 20 | 8 | 2 | 66.67% | 80.00% |
| Overall | 50 | 36 | 14 | 50.00% | 68.00% |

Strict accuracy counts only correct answers. Weighted accuracy gives full credit to correct answers and half credit to partial answers.

The evaluation should be interpreted as an initial practical assessment, not a final benchmark. The dataset was built manually because no gold-standard dataset existed for this archive collection.

---
