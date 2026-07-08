"""Shared helpers for resolving run output directories.

All run/eval scripts compute their index and answer directories the same way so
that answers can be regenerated against a downloaded index without re-indexing.

Layout::

    <output-dir>/<dataset_name>[/<index_model>]/index
    <output-dir>/<dataset_name>[/<index_model>]/answers

The optional ``<index_model>`` segment labels the LLM used to *build* the graph
(e.g. ``gpt-oss-20b``), so graphs from different builder models can coexist.
"""
from __future__ import annotations

import re
from pathlib import Path


def sanitize_model_name(name: str) -> str:
    """Turn a model id into a safe single path segment.

    Slashes and other separators (e.g. ``openai/gpt-oss-20b``) collapse to
    underscores so the result is one directory level.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def dataset_name_from_path(dataset_path: str | Path) -> str:
    """Dataset name = file stem for single-file datasets, else the dir name."""
    dataset_path = Path(dataset_path)
    return dataset_path.stem if dataset_path.is_file() else dataset_path.name


def resolve_run_dirs(
    output_dir: str | Path,
    dataset_path: str | Path,
    index_model: str | None = None,
) -> tuple[Path, Path, Path]:
    """Return ``(index_dir, answers_dir, base_dir)`` for a run.

    ``index_model`` (if given) is sanitized and inserted between the dataset
    name and the ``index``/``answers`` leaves.
    """
    base = Path(output_dir) / dataset_name_from_path(dataset_path)
    if index_model:
        base = base / sanitize_model_name(index_model)
    return base / "index", base / "answers", base
