---
title: Synthetic architecture summary
source: https://example.invalid/diagrams/architecture.svg
valid_from: 2026-01-01
---
Architecture
============

This Markdown summary describes a fictional diagram. The API sends document
embeddings to a local PostgreSQL database; Ollama provides local embeddings.

Storage
-------

Documents are mounted read only. Database files persist independently.
