"""Convert regenerated answers into the input JSON expected by
GraphRAG-Benchmark's ``Evaluation/generation_eval.py``.

Reads the dataset (for questions + gold answers) and the per-question answer
files written by the run scripts (``<output-dir>/<dataset_name>/answers/<i>.txt``),
then emits a JSON array where each item matches the benchmark schema::

    {
      "id": ..., "question": ..., "source": ...,
      "context": [...], "evidence": ...,
      "question_type": "Fact Retrieval" | "Complex Reasoning" | ...,
      "generated_answer": ..., "ground_truth": ...
    }

For bioasq/musique/2wikimultihopqa the gold answers are short, so
``question_type`` maps to metrics ``rouge_score`` + ``answer_correctness``;
``context``/``evidence`` are unused by those metrics and default to empty.

Example
-------
    python runs/build_graphrag_bench_input.py \
        --dataset-path datasets/bioasq \
        --output-dir generated/ragu_bioasq --source bioasq
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from natural_rag.dataset import RAGDataset
from natural_rag.paths import dataset_name_from_path, resolve_run_dirs

# Default question_type per dataset (only affects which metrics run; bioasq is
# factoid, the multi-hop sets are reasoning — both -> rouge + answer_correctness).
DEFAULT_QUESTION_TYPE = {
    "bioasq": "Fact Retrieval",
    "musique": "Complex Reasoning",
    "2wikimultihopqa": "Complex Reasoning",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True,
                   help="The generated/<engine>_<dataset> dir used when answering.")
    p.add_argument("--index-model", default=None,
                   help="Builder-model label subdirectory (e.g. gpt-oss-20b), if used when answering.")
    p.add_argument("--source", default=None, help="Corpus name (defaults to dataset name).")
    p.add_argument("--question-type", default=None,
                   help="Override question_type for all items.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSON path (defaults to <output-dir>/<dataset>/graphrag_bench_input.json).")
    p.add_argument("--skip-missing", action=argparse.BooleanOptionalAction, default=False,
                   help="Skip questions whose answer file is missing (default: error).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset_name = dataset_name_from_path(args.dataset_path)
    source = args.source or dataset_name
    question_type = args.question_type or DEFAULT_QUESTION_TYPE.get(dataset_name, "Complex Reasoning")

    _, answers_dir, base_dir = resolve_run_dirs(
        args.output_dir, args.dataset_path, args.index_model
    )
    out_path = args.out or (base_dir / "graphrag_bench_input.json")

    dataset = RAGDataset.load_auto(args.dataset_path)

    items: list[dict] = []
    missing: list[int] = []
    for i, q in enumerate(dataset.questions):
        answer_path = answers_dir / f"{i}.txt"
        if not answer_path.exists():
            missing.append(i)
            if args.skip_missing:
                continue
            raise FileNotFoundError(
                f"Missing answer file {answer_path}. Run the *_run.py with "
                f"--no-build-index --answer first, or pass --skip-missing."
            )
        items.append({
            "id": str(q.metadata.get("id", i)),
            "question": q.text,
            "source": source,
            "context": [],
            "evidence": "",
            "question_type": question_type,
            "generated_answer": answer_path.read_text().strip(),
            "ground_truth": q.reference_answers[0] if q.reference_answers else "",
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2))

    print(f"Wrote {len(items)} items -> {out_path}")
    print(f"  source={source!r}  question_type={question_type!r}")
    if missing:
        print(f"  WARNING: {len(missing)} answer files missing (indices e.g. {missing[:10]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
