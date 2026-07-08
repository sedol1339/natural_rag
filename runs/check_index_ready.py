"""Verify that a downloaded index is laid out correctly and that answers can be
generated *without* re-indexing.

Two levels of checking:

1. File-presence check (offline, always runs): confirms the expected index
   files exist under ``<output-dir>/<dataset_name>/index`` — the exact path the
   run scripts (``ragu_run.py`` / ``lightrag_run.py``) compute — and, for RAGU,
   reports the graph size and the vector dimension stored in the index.

2. ``--probe`` (online, optional): actually loads the pipeline with indexing
   disabled and generates a short answer for the first question. This is the
   real "with an index present, everything works" guarantee. It needs the live
   embedder (Docker) and LLM (vsellm) — configured via the same env vars as the
   run scripts (OPENAI_BASE_URL, OPENAI_API_KEY, EMBED_BASE_URL, EMBED_MODEL,
   EMBEDDING_DIM, LLM_MODEL).

Examples
--------
    # offline structure check
    python runs/check_index_ready.py --engine ragu \
        --dataset-path datasets/bioasq --output-dir generated/ragu_bioasq

    # full probe (needs Docker embedder + vsellm key)
    python runs/check_index_ready.py --engine ragu \
        --dataset-path datasets/2wikimultihopqa/2wikimultihopqa.json \
        --output-dir generated/ragu_2wikimultihopqa \
        --probe --short-answers
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from natural_rag.dataset import RAGDataset
from natural_rag.paths import resolve_run_dirs

# Files RAGU persists into the index folder (see ragu DEFAULT_FILENAMES).
RAGU_REQUIRED = [
    "knowledge_graph.gml",
    "vdb_entity.json",
    "vdb_relation.json",
    "vdb_chunk.json",
    "kv_chunks.json",
]
RAGU_OPTIONAL = ["kv_community.json", "kv_community_summary.json"]

# LightRAG working_dir hallmarks (names vary a bit across versions).
LIGHTRAG_EXPECTED = [
    "graph_chunk_entity_relation.graphml",
    "vdb_entities.json",
    "vdb_relationships.json",
    "vdb_chunks.json",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", required=True, choices=["ragu", "lightrag"])
    p.add_argument("--dataset-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--index-model", default=os.environ.get("INDEX_MODEL"),
                   help="Builder-model label subdirectory (e.g. gpt-oss-20b), if used when answering.")
    p.add_argument("--make-dirs", action=argparse.BooleanOptionalAction, default=False,
                   help="Create the index/answers dirs (so you can drop index files in), then check.")
    p.add_argument("--probe", action=argparse.BooleanOptionalAction, default=False,
                   help="Generate an answer for question 0 (needs live embedder + LLM).")
    p.add_argument("--short-answers", action=argparse.BooleanOptionalAction, default=True)
    # Probe connection settings (mirror the run scripts' env defaults).
    p.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.vsellm.ru/v1"))
    p.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--embed-base-url", default=os.environ.get("EMBED_BASE_URL", "http://127.0.0.1:8080/v1"))
    p.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"))
    p.add_argument("--embed-model", default=os.environ.get("EMBED_MODEL", "/data"))
    p.add_argument("--embedding-dim", type=int, default=int(os.environ.get("EMBEDDING_DIM", "768")))
    p.add_argument("--query-mode", default=os.environ.get("QUERY_MODE", "local"))
    p.add_argument("--query-engine", default=os.environ.get("QUERY_ENGINE", "local"))
    return p.parse_args()


def _vdb_dim(vdb_json: Path) -> int | None:
    """RAGU nano-vdb files store 'embedding_dim' and a flat 'matrix'."""
    try:
        data = json.loads(vdb_json.read_text())
    except Exception:
        return None
    for key in ("embedding_dim", "dim"):
        if isinstance(data.get(key), int):
            return data[key]
    return None


def check_files(engine: str, index_dir: Path) -> bool:
    print(f"Index dir: {index_dir}")
    if not index_dir.is_dir():
        print("  MISSING: index directory does not exist.")
        return False

    present = {p.name for p in index_dir.iterdir() if p.is_file()}
    ok = True

    if engine == "ragu":
        for name in RAGU_REQUIRED:
            mark = "OK  " if name in present else "MISS"
            if name not in present:
                ok = False
            print(f"  [{mark}] {name}")
        for name in RAGU_OPTIONAL:
            if name in present:
                print(f"  [OK  ] {name} (optional)")
        # Report graph size and embedding dim.
        gml = index_dir / "knowledge_graph.gml"
        if gml.exists():
            try:
                import networkx as nx  # ragu dependency
                g = nx.read_gml(gml)
                print(f"  graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
            except Exception as e:
                print(f"  graph: could not read ({e})")
        dim = _vdb_dim(index_dir / "vdb_entity.json")
        if dim is not None:
            print(f"  vdb_entity embedding_dim = {dim}  "
                  f"(answer embedder MUST match this dim; expected 768 for gte-multilingual-base)")
    else:  # lightrag
        for name in LIGHTRAG_EXPECTED:
            mark = "OK  " if name in present else "warn"
            if name not in present:
                ok = False
            print(f"  [{mark}] {name}")
        extra = sorted(present - set(LIGHTRAG_EXPECTED))
        if extra:
            print(f"  other files present: {extra}")
        if not present:
            print("  MISSING: index directory is empty.")

    return ok


def probe(args: argparse.Namespace, index_dir: Path) -> bool:
    print("\nProbe: loading pipeline with indexing DISABLED and answering Q0 ...")
    dataset = RAGDataset.load_auto(args.dataset_path)
    if not dataset.questions:
        print("  no questions in dataset.")
        return False
    question = dataset.questions[0]

    if not args.openai_api_key:
        print("  ERROR: OPENAI_API_KEY required for probe.")
        return False

    if args.engine == "ragu":
        from ragu.models.llm import CachedAsyncOpenAI, LLMOpenAI
        from ragu.models.embedder import EmbedderOpenAI
        from natural_rag.pipelines.ragu_pipelines import RAGUPipeline

        llm_client = CachedAsyncOpenAI(base_url=args.openai_base_url, api_key=args.openai_api_key)
        embed_client = CachedAsyncOpenAI(base_url=args.embed_base_url, api_key=args.openai_api_key)
        pipeline = RAGUPipeline(
            language="english",
            index_dir=index_dir,
            builder_llm=LLMOpenAI(llm_client, args.llm_model),
            assistant_llm=LLMOpenAI(llm_client, args.llm_model),
            embedder=EmbedderOpenAI(embed_client, args.embed_model, dim=args.embedding_dim, batch_size=32),
            query_engine=args.query_engine,
            short_answers=args.short_answers,
        )
        # NOTE: build_index() is intentionally NOT called -> uses the on-disk graph.
        answer, _ = pipeline.generate_answer(question.text)
    else:  # lightrag
        import numpy as np
        from openai import AsyncOpenAI
        from lightrag.llm.openai import openai_complete_if_cache
        from lightrag.utils import EmbeddingFunc
        from natural_rag.pipelines.lightrag_pipelines import LightRAGPipeline

        async def embed(texts):
            client = AsyncOpenAI(base_url=args.embed_base_url, api_key=args.openai_api_key)
            resp = await client.embeddings.create(input=texts, model=args.embed_model)
            return np.array([d.embedding for d in resp.data])

        async def llm(prompt, system_prompt=None, history_messages=None, **kw):
            return await openai_complete_if_cache(
                args.llm_model, prompt, system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key=args.openai_api_key, base_url=args.openai_base_url, **kw)

        pipeline = LightRAGPipeline(
            working_dir=index_dir,
            llm_model_func=lambda prompt, system_prompt=None, history_messages=None, **kw: llm(
                prompt, system_prompt, history_messages, **kw),
            embedding_func=EmbeddingFunc(embedding_dim=args.embedding_dim, max_token_size=8192,
                                         func=lambda texts: embed(texts)),
            query_mode=args.query_mode,
            short_answers=args.short_answers,
        )
        answer, _ = pipeline.generate_answer(question.text)
        pipeline.close()

    print(f"  Q: {question.text}")
    print(f"  gold: {question.reference_answers[:1]}")
    print(f"  answer: {answer!r}")
    ok = bool(answer and answer.strip())
    print("  PROBE OK" if ok else "  PROBE FAILED (empty answer)")
    return ok


def main() -> int:
    args = parse_args()
    index_dir, answers_dir, _ = resolve_run_dirs(args.output_dir, args.dataset_path, args.index_model)
    if args.make_dirs:
        index_dir.mkdir(parents=True, exist_ok=True)
        answers_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created:\n  {index_dir}\n  {answers_dir}\n")
    files_ok = check_files(args.engine, index_dir)
    if not files_ok:
        print("\nStructure check FAILED — place the index files as listed above.")
        if not args.probe:
            return 1
    if args.probe:
        return 0 if probe(args, index_dir) else 2
    print("\nStructure check OK. Re-run with --probe (and live services) for a full answer test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
