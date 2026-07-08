import asyncio
import dataclasses
import inspect
from pathlib import Path
from typing import Any, Callable

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

from natural_rag.pipelines import RAGPipeline
from natural_rag.data import Document


# Brief-answer settings used when reference answers are short (bioasq, musique,
# 2wikimultihopqa). `response_type` is interpolated into LightRAG's answer
# prompt; `user_prompt` (newer LightRAG only) adds an explicit instruction.
SHORT_ANSWER_RESPONSE_TYPE = (
    "a single short phrase: a name, number, date, or a short "
    "semicolon-separated list, with no explanation"
)
SHORT_ANSWER_USER_PROMPT = (
    "Answer as briefly as possible. Output ONLY the direct answer (a single "
    "entity, name, number, date, or a short semicolon-separated list). No "
    "explanations, no full sentences, no markdown, no restating the question."
)

# LightRAG's built-in rag_response prompt hard-codes a "### References" section
# and encourages long, structured answers. QueryParam.response_type/user_prompt
# do NOT remove that, so for short-answer benchmarks we replace the prompt
# template outright. Only {context_data} is referenced (str.format ignores the
# other kwargs LightRAG passes, e.g. response_type/user_prompt).
SHORT_RAG_RESPONSE_PROMPT = """---Role---

You answer the user query using ONLY the information in the provided Context.

---Instructions---

- Output ONLY the direct answer: a single entity, name, number, date, or a short
  semicolon-separated list of such items.
- No explanations, no full sentences, no Markdown, no headings.
- Do NOT output any references, citations, sources, or bibliography section, and
  do not output any heading. Output nothing after the answer itself.
- If the answer is not in the Context, output exactly:
  I don't know.

---Context---

{context_data}
"""


def apply_short_answer_prompts() -> None:
    """Monkeypatch LightRAG's answer templates to terse, reference-free ones.

    Safe no-op if the installed LightRAG exposes prompts differently.
    """
    try:
        from lightrag.prompt import PROMPTS  # type: ignore
    except Exception:
        return
    for key in ("rag_response", "naive_rag_response"):
        if key in PROMPTS:
            PROMPTS[key] = SHORT_RAG_RESPONSE_PROMPT


def _supported_query_param(**kwargs: Any) -> QueryParam:
    """Build a QueryParam passing only fields the installed version supports."""
    valid = {f.name for f in dataclasses.fields(QueryParam)}
    filtered = {k: v for k, v in kwargs.items() if k in valid and v is not None}
    return QueryParam(**filtered)


class LightRAGPipeline(RAGPipeline):
    def __init__(
        self,
        working_dir: str | Path,
        llm_model_func: Callable,
        embedding_func: EmbeddingFunc,
        tokenizer: Any = None,
        query_mode: str = "hybrid",
        addon_params: dict[str, Any] | None = None,
        max_parallel_insert: int | None = None,
        short_answers: bool = False,
        response_type: str | None = None,
        user_prompt: str | None = None,
        enable_llm_cache: bool = True,
        enable_rerank: bool = False,
    ):
        self.query_mode = query_mode
        self.enable_rerank = enable_rerank
        # Brief-answer controls. Explicit args win; otherwise `short_answers`
        # enables the built-in terse settings.
        self.response_type = response_type or (
            SHORT_ANSWER_RESPONSE_TYPE if short_answers else None
        )
        self.user_prompt = user_prompt or (
            SHORT_ANSWER_USER_PROMPT if short_answers else None
        )
        if short_answers:
            apply_short_answer_prompts()

        self._loop = asyncio.new_event_loop()
        try:
            self._previous_loop = asyncio.get_event_loop()
        except Exception:
            self._previous_loop = None

        asyncio.set_event_loop(self._loop)

        rag_kwargs: dict[str, Any] = {
            'working_dir': str(working_dir),
            'llm_model_func': llm_model_func,
            'embedding_func': embedding_func,
            'tokenizer': tokenizer,
        }
        lightrag_parameters = inspect.signature(LightRAG).parameters
        if addon_params is not None and 'addon_params' in lightrag_parameters:
            rag_kwargs['addon_params'] = addon_params
        if (
            max_parallel_insert is not None
            and 'max_parallel_insert' in lightrag_parameters
        ):
            rag_kwargs['max_parallel_insert'] = max_parallel_insert
        # Disable LightRAG's LLM response cache so answers are regenerated fresh
        # with the current prompt (the downloaded index ships a large
        # kv_store_llm_response_cache.json that would otherwise return stale
        # answers built with the old prompt).
        if 'enable_llm_cache' in lightrag_parameters:
            rag_kwargs['enable_llm_cache'] = enable_llm_cache

        self._rag = LightRAG(**rag_kwargs)

        self._run(self._rag.initialize_storages())

        try:
            from lightrag.kg.shared_storage import initialize_pipeline_status
            self._run(initialize_pipeline_status())
        except ImportError:
            pass

    def _run(self, coro):
        asyncio.set_event_loop(self._loop)
        return self._loop.run_until_complete(coro)

    async def _cancel_pending_tasks(self):
        current_task = asyncio.current_task(loop=self._loop)
        pending_tasks = [
            task
            for task in asyncio.all_tasks(self._loop)
            if task is not current_task and not task.done()
        ]
        if not pending_tasks:
            return
        for task in pending_tasks:
            task.cancel()
        await asyncio.gather(*pending_tasks, return_exceptions=True)

    # def build_index(self, documents: list[Document]):
    #     for doc in documents:
    #         if doc.text:
    #             self._run(self._rag.ainsert(doc.text))

    def build_index(self, documents: list[Document]):
        docs = [doc for doc in documents if doc.text]
        texts = [doc.text for doc in docs]
        ids = [doc.id for doc in docs]

        self._run(self._rag.ainsert(texts, ids=ids))

    def generate_answer(self, question: str) -> tuple[str, Any]:
        context_param = _supported_query_param(
            mode=self.query_mode,
            only_need_context=True,
            enable_rerank=self.enable_rerank,
        )
        context = self._run(self._rag.aquery(question, param=context_param))

        answer_param = _supported_query_param(
            mode=self.query_mode,
            response_type=self.response_type,
            user_prompt=self.user_prompt,
            enable_rerank=self.enable_rerank,
        )
        answer = self._run(self._rag.aquery(question, param=answer_param))

        return answer, context

    def close(self):
        try:
            self._run(self._rag.finalize_storages())
        except Exception:
            pass
        finally:
            if hasattr(self, "_loop") and not self._loop.is_closed():
                try:
                    self._run(self._cancel_pending_tasks())
                    self._run(self._loop.shutdown_asyncgens())
                except Exception:
                    pass
                self._loop.close()

            try:
                asyncio.set_event_loop(self._previous_loop)
            except Exception:
                try:
                    asyncio.set_event_loop(None)
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            if hasattr(self, "_loop") and not self._loop.is_closed():
                self.close()
        except Exception:
            pass
