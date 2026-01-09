from __future__ import annotations
from pathlib import Path
import re
from typing import Any, Self
import warnings

import yaml
from pydantic import BaseModel, Field, model_validator


class Document(BaseModel):
    """A document in a RAG corpus."""

    id: str = Field(description=(
        'A unique string identifier for a document. If all the documents have'
        ' unique titles, `id` can be equal to the `.title`. Should be a valid'
        ' filepath without extension (that is, may contain slash "/"'
        ' characters).'
    ))

    title: str | None = Field(default=None, description=(
        'A document title, may be None if document is untitled.'
    ))

    text: str | None = Field(default=None, description=(
        'A full text. May be omitted if, for example, no pre-defined PDF'
        ' parsing is given for the dataset. For documents with images, one can'
        ' subclass `Document` with additional fields.'
    ))

    source: bytes | None = Field(default=None, description=(
        'A source in the original format. For example, PDF bytes or HTML. May'
        ' be omitted, typically in cases when the `text` is given.'
    ))
    
    source_ext: str = Field(default='', description=(
        'An extension for the `.source` as file, starting from dot, if'
        ' specified.'
    ))

    metadata: dict[str, Any] = Field(default_factory=dict, description=(
        'Any additional metadata.'
    ))


class SourceLoc(BaseModel):
    """A location of relevant information in a corpus. If the relevant
    information is scattered across muitple documents, it can be
    represented as a list of `SourceLoc`.
    """
    
    doc_id: str = Field(description=(
        'A unique string identifier for a document.'
    ))
    
    loc: list[tuple[int, int] | str] | None = Field(
        default=None,
        description=(
            'A list of relevant spans. A relevans span is represented either'
            ' as start and end positions in characters, or as a start'
            ' substring (until end of sentence), or as a heading for a section'
            ' in markdown (should be enclosed here in [brackets]).'
        )
    )


class Question(BaseModel):
    """A question for a RAG corpus, supplemented with data to evaluate
    responses (such as the reference answer).
    """

    text: str = Field(description=(
        'A question text.'
    ))
    
    reference_answer: str | None = Field(default=None, description=(
        'A ground truth answer, if provided.'
    ))
    
    eval_rules: list[LLMJudgeEvalCheck] | None = Field(
        default=None,
        description=(
            'A list of `LLMJudgeEvalCheck` if the question supports'
            ' fine-grained evaluation based on a set of rules. The LLM judge'
            ' evaluates the answer according to each of the rules.'
        )
    )
    
    relevant: list[SourceLoc] | None = Field(default=None, description=(
        'Locations of the relevant information in the corpus, if provided.'
    ))
  
    metadata: dict[str, Any] = Field(default_factory=dict, description=(
        'Any additional metadata. For example, may have "tags" field with a'
        ' list of tags for the question. May also contain "comment" field'
        ' which is an optional commentary about the question. The comment may'
        ' explain why the qustion have some tag, or why it lacks the ground'
        ' truch answer etc. Is for practioners to read. By default is also'
        ' used by LLM-as-judge.'
    ))


class LLMJudgeEvalCheck(BaseModel):
    """A single rule for LLM judge to evaluate the answer granularly."""

    score: int = Field(description=(
        'A score given if the `check` is passed. May pe positive if the check'
        ' represents a good answer, negative if the check represents something'
        ' wrong that should not be mentioned, or zero if the check represents'
        ' some additional info in the answer, that is not required.'
    ))
    
    check: str = Field(description=(
        'An instruction for LLM-as-judge, typically in form of "mention'
        ' <something> in the answer". Should be unambiguous, so that the LLM'
        ' judge can easily tell "yes" or "no": is the check passed or not.'
    ))
  
    metadata: dict[str, Any] = Field(default_factory=dict, description=(
        'Any additional metadata. May contain "comment" field in the same way'
        ' as the `Question` may contain it. By default is also used by'
        ' LLM-as-judge.'
    ))


class RAGDataset(BaseModel):
    """A RAG corpus plus a set of questions.
    
    Can be saved to disk or loaded. On a disk, docs are represented as
    .md or .txt files, and questions are represented as .yaml files,
    to enable easy viewing and editing on disk.
    """
    
    documents: dict[str, Document] = Field(
        default_factory=dict,
        description=(
            'A mapping from document ID to a document.'
        )
    )
    
    questions: list[Question] = Field(
        default_factory=list[Question],
        description=(
            'A list of questions.'
        )
    )
    
    @model_validator(mode='after')
    def check_ids_exist(self) -> Self:
        for q in self.questions:
            if q.relevant is not None:
                for source in q.relevant:
                    assert source.doc_id in self.documents, (
                        f'doc ID {source.doc_id} mentioned as relevant'
                        ' for a question, but not in the documents'
                    )
        return self
    
    @classmethod
    def load_from_dir(
        cls,
        dir: str | Path,
        load_texts: bool = True,
        load_sources: bool = False,
        qa_path: str = 'questions.yaml',
        corpus_info_path: str = 'corpus_info.yaml',
        texts_dir: str = 'docs',
        sources_dir: str = 'sources',
    ) -> Self:
        """Loads from a dicectory organized as follows:
        
        - {dir}/corpus_info.yaml contains info about documents.
        - {dir}/questions.yaml contains questions (and optionally
          answers).
        - {dir}/docs/{id}.md (or .txt) contain documents
          (only the documents mentioned in corpus.yaml will be loaded).
          Will try to load .md, then .txt.
        - {dir}/sources/{id}{ext} optionally contain documents
          as sources, such as PDF or HTML (only the documents mentioned
          in corpus.yaml will be loaded). The extension is a field
          in the `Document` class.
          
        Such a structure enables easy viewing and editing on disk.
        
        By default, will load `Document` objects with `.text` and
        `.source` field filled, if the corresponding files exist. If the
        corpus is way too large, may skip loading texts with
        `load_texts=False`. May load sources with `load_sources=False`.
        """
        
        ROOT_DIR = Path(dir)
        
        qa_yaml_data = yaml.safe_load(
            (ROOT_DIR / qa_path).read_text()
        )
        assert isinstance(qa_yaml_data, list)
        qa = [
            Question.model_validate(entry)
            for entry in qa_yaml_data # pyright: ignore[reportUnknownVariableType]
        ]
        
        corpus_yaml_data = yaml.safe_load(
            (ROOT_DIR / corpus_info_path).read_text()
        )
        assert isinstance(corpus_yaml_data, list)
        docs = [
            Document.model_validate(entry)
            for entry in corpus_yaml_data # pyright: ignore[reportUnknownVariableType]
        ]
        
        if load_texts:
            for doc in docs:
                if (path := ROOT_DIR / texts_dir / f'{doc.id}.md').exists():
                    doc.text = path.read_text()
                elif (path := ROOT_DIR / texts_dir / f'{doc.id}.txt').exists():
                    doc.text = path.read_text()
                else:
                    warnings.warn(
                        f'Cannot find {doc.id} in "{texts_dir}" dir.'
                        ' Will not add it to `.documents`.'
                    )
        
        if load_sources:
            for doc in docs:
                rel_path = f'{doc.id}{doc.source_ext}'
                if (path := ROOT_DIR / sources_dir / rel_path).exists():
                    doc.source = path.read_bytes()
        
        return cls(
            documents={doc.id: doc for doc in docs},
            questions=qa,
        )
    
    def report_stats(self) -> str:
        report_lines: list[str] = []
        
        doc_sizes_ascending = sorted([
            (doc.id, len(doc.text))
            for doc in self.documents.values()
            if doc.text is not None
        ], key=lambda x: x[1])
        
        joint_text = '\n'.join([
            doc.text
            for doc in self.documents.values()
            if doc.text is not None
        ])
        
        report_lines.append(f'Total documents: {len(self.documents)}')
        report_lines.append(
            'Shortest documents in symbols:'
            f' {[size for _id, size in doc_sizes_ascending[:5]]}'
        )
        report_lines.append(
            'Longest documents in symbols:'
            f' {[size for _id, size in doc_sizes_ascending[::-1][:5]]}'
        )

        report_lines.append(f'Symbols: {len(joint_text)}')
        report_lines.append(f'Words: {len(re.findall(r'\w+', joint_text))}')
        report_lines.append(
            f'Pages (assuming 1800 chars/page): {len(joint_text) // 1800}'
        )
        
        report_lines.append(f'Total questions: {len(self.questions)}')
        
        if len(self.questions):
            n_relevant_per_question = [
                len(q.relevant or []) for q in self.questions
            ]
            all_relevant_doc_ids: list[str] = sum([
                ([r.doc_id for r in (q.relevant or [])])
                for q in self.questions
            ], []) # type: ignore
            docs_without_questions = (
                {doc.id for doc in self.documents.values()}
                - set(all_relevant_doc_ids)
            )
            report_lines.append(
                'Min relevant docs per question:'
                f' {min(n_relevant_per_question)}'
            )
            report_lines.append(
                'Max relevant docs per question:'
                f' {max(n_relevant_per_question)}'
            )
            report_lines.append(
                'N docs without questions:'
                f' {len(docs_without_questions)}'
            )
        
        return '\n'.join(report_lines)
    
    def dump_to_dir(
        self,
        dir: str | Path,
        qa_path: str = 'questions.yaml',
        corpus_info_path: str = 'corpus_info.yaml',
        texts_dir: str = 'docs',
        sources_dir: str = 'sources',
    ):
        """Saves the dataset to the directory in the format described in
        `.load_from_dir()`.
        """
        
        ROOT_DIR = Path(dir)
        ROOT_DIR.mkdir(exist_ok=True, parents=True)
        
        documents_to_save: list[Document] = []
        for doc in self.documents.values():
            doc = doc.model_copy()
            if doc.text is not None:
                (ROOT_DIR / texts_dir).mkdir(exist_ok=True, parents=True)
                (ROOT_DIR / texts_dir / f'{doc.id}.md').write_text(doc.text)
                doc.text = None
            if doc.source is not None:
                rel_path = f'{doc.id}{doc.source_ext}'
                (ROOT_DIR / sources_dir).mkdir(exist_ok=True, parents=True)
                (ROOT_DIR / sources_dir / rel_path).write_bytes(doc.source)
                doc.text = None
            documents_to_save.append(doc)
        
        docs_as_dicts = [
            doc.model_dump(mode='json', exclude_defaults=True)
            for doc in documents_to_save
        ]
        (ROOT_DIR / corpus_info_path).write_text(
            yaml.dump(docs_as_dicts, sort_keys=False)
        )
        
        qa_as_dicts = [
            qa.model_dump(mode='json', exclude_defaults=True)
            for qa in self.questions
        ]
        (ROOT_DIR / qa_path).write_text(
            yaml.dump(qa_as_dicts, sort_keys=False)
        )