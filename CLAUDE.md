# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KBSkills is a CLI-based local knowledge base system that combines GraphRAG (via LightRAG + Google Gemini) with a skills-driven thinking framework to generate structured outlines from ingested knowledge. The primary language is Chinese for user-facing output and skill definitions.

## Common Commands

```bash
# Install in development mode
pip install -e .

# CLI entry point
kbskills --help

# Initialize configuration (prompts for Gemini API key, creates directories)
kbskills init

# Ingest documents from local directory and/or URL file
kbskills ingest --dir KBase/docs --urls KBase/urls.txt

# Show knowledge base status (entity/relation counts, storage size)
kbskills status

# List available skills, show details, or test matching
kbskills skills list
kbskills skills show first_principles
kbskills skills match "某个主题"

# Generate an outline for a topic (main workflow)
kbskills query "主题" --mode hybrid --output output/result.md
```

## Architecture

### Pipeline: Query → Outline (5-step agent in `agent/topic_agent.py`)

1. **Topic Decomposition** — LLM breaks topic into 3-5 sub-topics with search queries
2. **Skill Matching** — Scores topic against skill triggers (domain embedding similarity 50%, keyword 30%, intent regex 20%)
3. **Knowledge Retrieval** — Queries LightRAG graph for each sub-topic (modes: naive/local/global/hybrid)
4. **Skill-Guided Analysis** — If skills matched, applies their thinking framework steps via LLM
5. **Concern Identification + Outline Generation** — Identifies key concerns from KB, generates hierarchical markdown

### Ingestion Pipeline (`ingestion/pipeline.py`)

Three phases: local file loading → URL content fetching → LightRAG graph insertion. Each document is prepended with `[Source: ...]` metadata before insertion.

- **File types**: PDF (PyMuPDF), DOCX, PPTX, Excel, CSV, text/markdown, images (Gemini Vision)
- **URL types**: Web pages (httpx + html2text, special Wikipedia handler), YouTube (transcript API with yt-dlp fallback), audio files (Gemini transcription)

### Skills System (`skills/` directory + `src/kbskills/skills/`)

Skills are YAML files defining thinking frameworks with:
- `metadata.trigger` — domains, keywords, intent_patterns, threshold for matching
- `thinking_framework.steps` — sequential prompts with `{topic}` placeholder
- `tools` — output format templates (e.g., assumption matrices, logic trees)
- `output_requirements` — required sections and style guidelines

Skill matching uses Gemini embeddings for domain cosine similarity combined with keyword and regex scoring.

### Key Modules

| Module | Purpose |
|---|---|
| `cli.py` | Click-based CLI with 7 commands (init, ingest, status, skills list/show/match, query) |
| `config.py` | Layered config: defaults → `~/.kbskills/config.json` → `.kbskills.json` → env vars (`KBSKILLS_*`) |
| `knowledge/graph_builder.py` | LightRAG singleton wrapper with Gemini LLM/embedding backends (embedding dim: 3072) |
| `agent/prompts.py` | Three prompt templates: TOPIC_DECOMPOSITION, CONCERN_IDENTIFICATION, OUTLINE_GENERATION |
| `skills/matcher.py` | SkillMatcher with weighted multi-factor scoring |
| `skills/executor.py` | Helper functions that inject skill guidance into LLM prompts |

### Data Layout

- `data/raw/` — ingested raw files
- `data/graph/` — LightRAG graph artifacts (graphml, JSON KV stores, vector DB JSON)
- `skills/` — user-facing skill YAML definitions
- `output/` — generated markdown outlines (`{topic}_{timestamp}.md`)
- `KBase/` — sample knowledge base source (docs + urls.txt)

## Technical Notes

- LightRAG is async-first; `_run_async()` helper bridges sync CLI calls to async operations
- The RAG instance is cached as a module-level singleton (`_rag_instance`)
- Build system uses Hatch (`hatchling`); package source is at `src/kbskills/`
- Requires Python ≥ 3.11; LLM backend is Google Gemini (default model: gemini-2.5-pro)
