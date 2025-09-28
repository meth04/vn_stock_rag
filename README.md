# VN Stock Advisor + RAG-Anything integration

This project is a scaffold to integrate a VN stock analysis pipeline with RAG-Anything
(for provenance-backed research & QA). It computes technical indicators, runs strategies,
generates reports, and connects to RAG-Anything for ingestion & provenance.

Prereqs:
 - Python 3.9+
 - Install dependencies: pip install -e .  (or pip install -r requirements.txt)
 - Optional: install and run RAG-Anything (https://github.com/HKUDS/RAG-Anything). See their README. :contentReference[oaicite:2]{index=2}

Env:
 - Copy `.env.example` to `.env` and set GEMINI_API_KEY, SERPER_API_KEY, RAG_ANYTHING_ENDPOINT if you use them.

Usage:
 - CLI: python -m vn_stock_advisor.main --symbol FPT --price-csv data/FPT.csv --rag-endpoint http://localhost:8080

Notes on RAG integration:
 - This project provides a RagClient adapter (src/vn_stock_advisor/rag_integration.py).
 - Depending on how you deploy RAG-Anything (local server vs CLI), update RAG_ANYTHING_ENDPOINT or ensure `raganything` package/CLI is available.
 - See https://github.com/HKUDS/RAG-Anything for setup & best practices. :contentReference[oaicite:3]{index=3}
