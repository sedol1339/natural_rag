from __future__ import annotations
import os
import base64
import json

from neomodel import (
    config, StructuredNode, StringProperty, JSONProperty, 
    IntegerProperty, ArrayProperty, RelationshipTo, RelationshipFrom, 
    StructuredRel, UniqueIdProperty, db
)

from natural_rag.dataset import RAGDataset
from natural_rag.data import Document, Question


class NeoRelSourceLoc(StructuredRel):
    """
    Represents the SourceLoc connection between a Question and a Document.
    The `loc` field is stored on the relationship itself.
    """
    # Storing complex list[tuple|str] as JSON because Neo4j 
    # properties handle primitives (int/string) arrays but not mixed/nested arrays.
    loc = JSONProperty(default=None)


class NeoDocument(StructuredNode):
    """Neo4j representation of Document."""
    
    # Maps to Document.id
    doc_id = StringProperty(unique_index=True, required=True)
    
    title = StringProperty()
    text = StringProperty()
    
    # Storing bytes in Neo4j is possible but usually not recommended for large blobs.
    # We will store this as a Base64 encoded string.
    source_b64 = StringProperty() 
    
    source_ext = StringProperty(default='')
    metadata = JSONProperty(default={})

    # Relationships
    dataset = RelationshipFrom('NeoRAGDataset', 'CONTAINS_DOC')


class NeoLLMJudgeEvalCheck(StructuredNode):
    """Neo4j representation of LLMJudgeEvalCheck."""
    
    uid = UniqueIdProperty()
    score = IntegerProperty(required=True)
    check = StringProperty(required=True)
    metadata = JSONProperty(default={})
    
    # Relationships
    question = RelationshipFrom('NeoQuestion', 'HAS_EVAL_CHECK')


class NeoQuestion(StructuredNode):
    """Neo4j representation of Question."""
    
    uid = UniqueIdProperty()
    text = StringProperty(required=True)
    
    # Stores list[str]
    reference_answers = ArrayProperty(StringProperty(), default=[])
    
    metadata = JSONProperty(default={})

    # Relationships
    eval_rules = RelationshipTo('NeoLLMJudgeEvalCheck', 'HAS_EVAL_CHECK')
    
    # This maps to 'relevant' in the Pydantic model. 
    # Connects Question -> Document with the specific location properties.
    relevant_docs = RelationshipTo('NeoDocument', 'REFERENCES', model=NeoRelSourceLoc)
    
    dataset = RelationshipFrom('NeoRAGDataset', 'CONTAINS_QUESTION')


class NeoRAGDataset(StructuredNode):
    """Root node to group an imported dataset."""
    
    uid = UniqueIdProperty()
    name = StringProperty(default='Imported Dataset')
    created_at = StringProperty() # ISO format timestamp suggested

    # Relationships
    documents = RelationshipTo('NeoDocument', 'CONTAINS_DOC')
    questions = RelationshipTo('NeoQuestion', 'CONTAINS_QUESTION')


from datetime import datetime


def ingest_rag_dataset_to_neo4j(dataset: RAGDataset, dataset_name: str = "RAG Corpus") -> str:
    """
    Converts a Pydantic RAGDataset into Neo* neomodel objects and persists them 
    to the Neo4j database wrapped in a transaction.
    """
    
    # Using a transaction is faster and safer for bulk operations
    with db.transaction:
        # 1. Create the Dataset Node
        neo_dataset = NeoRAGDataset(
            name=dataset_name, 
            created_at=datetime.utcnow().isoformat()
        ).save()
        
        print(f"Created Dataset Node: {neo_dataset.uid}")
        
        # 2. Process Documents
        # We assume dataset.documents is a dict {id: Document}
        # We rely on doc.id, but ensure consistency with the key
        
        # Cache created neo documents to link questions later: {doc_id_str: NeoDocument}
        doc_cache = {}
        
        for doc_key, doc_obj in dataset.documents.items():
            
            # Handle source bytes -> base64 string
            source_encoded = None
            if doc_obj.source:
                try:
                    source_encoded = base64.b64encode(doc_obj.source).decode('utf-8')
                except Exception:
                    pass # Or handle error logging

            # Create NeoDocument
            neo_doc = NeoDocument(
                doc_id=doc_obj.id,
                title=doc_obj.title,
                text=doc_obj.text,
                source_b64=source_encoded,
                source_ext=doc_obj.source_ext,
                metadata=doc_obj.metadata
            ).save()
            
            # Link to Dataset
            neo_dataset.documents.connect(neo_doc)
            
            doc_cache[doc_obj.id] = neo_doc
            
        print(f"Ingested {len(doc_cache)} documents.")

        # 3. Process Questions
        for q_obj in dataset.questions:
            
            # Create NeoQuestion
            neo_q = NeoQuestion(
                text=q_obj.text,
                reference_answers=q_obj.reference_answers,
                metadata=q_obj.metadata
            ).save()
            
            # Link to Dataset
            neo_dataset.questions.connect(neo_q)
            
            # 3a. Handle Eval Rules (LLMJudgeEvalCheck)
            if q_obj.eval_rules:
                for rule in q_obj.eval_rules:
                    neo_rule = NeoLLMJudgeEvalCheck(
                        score=rule.score,
                        check=rule.check,
                        metadata=rule.metadata
                    ).save()
                    
                    neo_q.eval_rules.connect(neo_rule)
            
            # 3b. Handle Relevant Locations (SourceLoc)
            if q_obj.relevant:
                for loc_obj in q_obj.relevant:
                    target_doc_id = loc_obj.doc_id
                    
                    if target_doc_id in doc_cache:
                        target_neo_doc = doc_cache[target_doc_id]
                        
                        # Connect Question to Document with 'relevant' properties
                        # neomodel separates relationship properties into a dict
                        rel_props = {
                            'loc': loc_obj.loc # This will be serialized to JSON by JSONProperty
                        }
                        
                        neo_q.relevant_docs.connect(target_neo_doc, rel_props)
                    else:
                        print(f"Warning: Question refers to missing document ID {target_doc_id}")

        print(f"Ingested {len(dataset.questions)} questions.")

    return neo_dataset.uid