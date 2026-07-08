import asyncio
import inspect
import sys
from pathlib import Path
from typing import Any

from natural_rag.dataset import Document
from natural_rag.pipelines import RAGPipeline

from ragu import (
    SimpleChunker,
    SmartSemanticChunker,
    KnowledgeGraph,
    BuilderArguments,
    Settings,
    QueryPlanEngine,
    TwoStageArtifactsExtractorLLM,
)
from ragu.common.logger import logger as ragu_logger
from ragu.common.prompts.messages import ChatMessages, UserMessage
from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.models.embedder import Embedder
from ragu.models.llm import LLM
from ragu.search_engine.local_search import LocalSearchEngine
from ragu.search_engine.naive_search import NaiveSearchEngine


# Brief-answer prompt used when reference answers are short (e.g. bioasq,
# musique, 2wikimultihopqa). Mirrors DEFAULT_RESPONSE_ONLY_PROMPT's jinja
# variables ({{ query }}, {{ context }}, {{ language }}) but forces a terse,
# extractive answer so the answer length matches the short gold answers.
SHORT_ANSWER_PROMPT = """
**Goal**
Answer the query as briefly as possible using the context.

**Instructions**
1. Output ONLY the direct answer: a single entity, name, number, date, or a
   short semicolon-separated list of such items.
2. No explanations, no full sentences, no restating of the question, no
   markdown, no citations.
3. If the answer is not supported by the context, output exactly: I don't know.

Query: {{ query }}
Context: {{ context }}

Provide the answer in the following language: {{ language }}
"""


class RAGUPipeline(RAGPipeline):
    def __init__(
        self,
        language: str,
        index_dir: str | Path,
        builder_llm: LLM,
        assistant_llm: LLM,
        embedder: Embedder,
        use_llm_summarization: bool = True,
        vectorize_chunks: bool = True,
        query_use_summary: bool = False,
        query_use_chunks: bool = False,
        chunker_name: str = "smart",
        simple_chunk_size: int = 2048,
        simple_chunk_overlap: int = 0,
        retrieval_top_k: int = 20,
        qa_top_k: int = 20,
        tokenizer_backend: str = "tiktoken",
        tokenizer_llm_name: str = "gpt-4o",
        tokenizer_embedder_name: str = "gpt-4o",
        query_engine: str = "local",
        short_answers: bool = False,
    ):
        ragu_logger.remove()
        ragu_logger.add(sys.stdout, level="DEBUG")

        self.language = language
        self.index_dir = index_dir
        self.builder_llm = builder_llm
        self.assistant_llm = assistant_llm
        self.embedder = embedder
        self.query_use_summary = query_use_summary
        self.query_use_chunks = query_use_chunks
        self.retrieval_top_k = retrieval_top_k
        self.qa_top_k = qa_top_k

        Settings.storage_folder = self.index_dir
        Settings.language = self.language

        if chunker_name == "simple":
            self.chunker = SimpleChunker(
                max_chunk_size=simple_chunk_size,
                overlap=simple_chunk_overlap,
            )
        elif chunker_name == "smart":
            self.chunker = SmartSemanticChunker()
        else:
            raise ValueError(f"Unsupported RAGU chunker: {chunker_name}")

        BIOASQ_ENTITY_TYPES = [
            "DiseaseOrDisorder",
            "SymptomOrFinding",
            "DrugOrChemical",
            "GeneOrProtein",
            "BiologicalProcess",
            "Anatomy",
            "CellOrOrganism",
            "MedicalProcedureOrTest",
            "TherapeuticIntervention",
            "MeasurementOrOutcome",
            "StudyOrEvidence",
            "Other",
        ]

        BIOASQ_RELATION_TYPES = [
            "treats",
            "prevents",
            "causes",
            "associated_with",
            "risk_factor_for",
            "symptom_of",
            "adverse_effect_of",
            "inhibits",
            "activates",
            "regulates",
            "expressed_in",
            "located_in",
            "part_of",
            "interacts_with",
            "biomarker_for",
            "diagnoses",
            "measures",
            "administered_by",
            "has_dose_or_route",
            "studied_in",
            "improves_outcome",
            "worsens_outcome",
            "compared_with",
            "other",
        ]

        self.artifact_extractor = TwoStageArtifactsExtractorLLM(
            llm=self.builder_llm,
            do_entity_validation=False,
            do_relation_validation=False,
            # entity_types=BIOASQ_ENTITY_TYPES,
            # relation_types=BIOASQ_RELATION_TYPES
        )

        self.builder_settings = BuilderArguments(
            use_llm_summarization=use_llm_summarization,
            vectorize_chunks=vectorize_chunks,
            make_community_summary=False,
        )

        # Pass only kwargs the installed RAGU version's KnowledgeGraph accepts
        # (its signature has drifted across versions: tokenizer/embedder-token
        # args were removed).
        kg_kwargs: dict[str, Any] = {
            "llm": self.builder_llm,
            "embedder": self.embedder,
            "chunker": self.chunker,
            "artifact_extractor": self.artifact_extractor,
            "builder_settings": self.builder_settings,
            "language": self.language,
            "embedder_token_limit": 8000,
            "tokenizer_backend": tokenizer_backend,
            "tokenizer_llm_name": tokenizer_llm_name,
            "tokenizer_embedder_name": tokenizer_embedder_name,
        }
        kg_params = inspect.signature(KnowledgeGraph.__init__).parameters
        self.knowledge_graph = KnowledgeGraph(
            **{k: v for k, v in kg_kwargs.items() if k in kg_params}
        )

        self.query_engine = query_engine

        def _short_instruction() -> RAGUInstruction:
            return RAGUInstruction(
                messages=ChatMessages.from_messages(
                    [UserMessage(content=SHORT_ANSWER_PROMPT)]
                ),
                pydantic_model=str,
                description="Brief extractive answer prompt for short-answer benchmarks.",
            )

        if query_engine == "naive":
            # Naive = plain vector RAG over chunks (uses the same RAGU index:
            # chunk vector DB + chunk KV). Graph/entities are not traversed.
            naive_search = NaiveSearchEngine(
                self.assistant_llm,
                self.knowledge_graph,
                self.embedder,
                tokenizer_model="gpt-4o-mini",
                language=self.language,
            )
            if short_answers:
                naive_search.update_prompt("naive_search", _short_instruction())
            self.search_engine = naive_search
        else:
            local_search = LocalSearchEngine(
                self.assistant_llm,
                self.knowledge_graph,
                self.embedder,
                tokenizer_model="gpt-4o-mini",
                language=self.language,
            )
            self.local_search = local_search
            if short_answers:
                local_search.update_prompt("local_search", _short_instruction())
            self.search_engine = QueryPlanEngine(local_search) if query_engine == "query_plan" else local_search

    def build_index(self, documents: list[Document]):
        docs = [doc.text for doc in documents if doc.text]
        asyncio.run(self.knowledge_graph.build_from_docs(docs))

    async def _answer(self, question: str, max_retries: int = 3) -> tuple[str, Any]:
        # NaiveSearchEngine.a_query does not accept `use_chunks`; only pass it
        # for the graph engines (local / query_plan).
        qa_kwargs = {} if self.query_engine == "naive" else {"use_chunks": True}
        context: Any = None
        for attempt in range(max_retries):
            try:
                context = await self.search_engine.a_search(question, top_k=self.retrieval_top_k)
                response = await self.search_engine.a_query(question, top_k=self.qa_top_k, **qa_kwargs)
            except Exception as exc:
                # A single bad question (e.g. provider content-filter 400, which
                # is not retryable) must not crash the whole batch. Log and
                # return a refusal placeholder so the run continues.
                ragu_logger.warning(
                    f"Answer generation failed ({type(exc).__name__}: {exc}); "
                    f"returning 'I don't know.' placeholder for this question."
                )
                return "I don't know.", context
            if response.response and response.response.strip():
                return response.response, context
            ragu_logger.warning(f"Empty answer on attempt {attempt+1}/{max_retries}, retrying...")
        return "NO ANSWER OBTAINED", context

    def generate_answer(self, question: str) -> tuple[str, Any]:
        return asyncio.run(self._answer(question))

    async def a_generate_answer(self, question: str) -> tuple[str, Any]:
        """Async entry point so a caller can answer many questions inside a
        single event loop. Using this (instead of per-question ``asyncio.run``
        via :meth:`generate_answer`) keeps RAGU's shared HTTP client and rate
        limiters bound to one loop, avoiding cross-loop reuse warnings and the
        spurious ``APIConnectionError`` retries that come with it.
        """
        return await self._answer(question)