"""
Katherine Orchestrator - Memory Engine
Handles embeddings, vector storage, and memory retrieval (RAG core).

Supports two embedding modes:
1. API-based (OpenAI/OpenRouter) - works with any Python version
2. Local (sentence-transformers) - requires Python 3.10-3.12

Includes:
- Deduplication at both write-time and read-time
- Hybrid search (semantic + BM25 keyword matching)
- Optional query expansion via LLM
"""
import time
import math
import re
from typing import Optional
from datetime import datetime, timezone
from collections import Counter
import httpx

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from config import settings
from models import Memory, MemorySearchResult


def _normalize_datetime(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware (UTC). Naive datetimes are assumed to be UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_datetime(date_str: str) -> datetime:
    """Parse ISO datetime string and ensure it's timezone-aware (UTC)."""
    parsed = datetime.fromisoformat(date_str)
    return _normalize_datetime(parsed)


# =============================================================================
# Text Processing Utilities for BM25
# =============================================================================

def tokenize(text: str) -> list[str]:
    """
    Simple tokenizer for BM25.
    Lowercases, removes punctuation, splits on whitespace.
    """
    # Lowercase and remove non-alphanumeric (keep spaces)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Split and filter empty
    tokens = [t for t in text.split() if t and len(t) > 1]
    return tokens


def compute_idf(documents: list[list[str]]) -> dict[str, float]:
    """
    Compute Inverse Document Frequency for all terms.
    IDF(t) = log((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
    """
    N = len(documents)
    doc_freq = Counter()
    
    for doc_tokens in documents:
        unique_tokens = set(doc_tokens)
        for token in unique_tokens:
            doc_freq[token] += 1
    
    idf = {}
    for term, freq in doc_freq.items():
        idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)
    
    return idf


def bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    idf: dict[str, float],
    avg_doc_len: float,
    k1: float = 1.5,
    b: float = 0.75
) -> float:
    """
    Compute BM25 score for a single document.
    
    Args:
        query_tokens: Tokenized query
        doc_tokens: Tokenized document
        idf: IDF values for all terms
        avg_doc_len: Average document length in corpus
        k1: Term frequency saturation parameter (default 1.5)
        b: Length normalization parameter (default 0.75)
    """
    doc_len = len(doc_tokens)
    term_freq = Counter(doc_tokens)
    
    score = 0.0
    for term in query_tokens:
        if term not in idf:
            continue
        
        tf = term_freq.get(term, 0)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
        
        score += idf[term] * (numerator / denominator)
    
    return score


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    Returns a value between -1 and 1 (1 = identical, 0 = orthogonal).
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


class EmbeddingProvider:
    """
    Embedding provider abstraction.
    Uses OpenAI API directly for embeddings (OpenRouter doesn't support embeddings).
    """
    
    def __init__(self):
        self._client: Optional[httpx.Client] = None
        self._model = settings.openai_embedding_model
    
    def initialize(self) -> None:
        """Initialize the HTTP client for embedding API via OpenAI."""
        if not settings.openai_api_key:
            raise RuntimeError(
                "OpenAI API key not set! Set KATHERINE_OPENAI_API_KEY in .env\n"
                "Note: OpenRouter does NOT support embeddings. You need an OpenAI API key.\n"
                "Alternatively, set KATHERINE_EMBEDDING_MODE=local to use local embeddings."
            )
        
        self._client = httpx.Client(
            base_url=settings.openai_embedding_base_url,
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
            },
            timeout=30.0
        )
        logger.info(f"Embedding provider initialized (OpenAI): {self._model}")
    
    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts via OpenRouter."""
        if not self._client:
            raise RuntimeError("Embedding provider not initialized")
        
        response = self._client.post(
            "/embeddings",
            json={
                "model": self._model,
                "input": texts
            }
        )
        response.raise_for_status()
        
        data = response.json()
        # Sort by index to maintain order
        embeddings = sorted(data["data"], key=lambda x: x["index"])
        return [e["embedding"] for e in embeddings]
    
    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None


class LocalEmbeddingProvider:
    """
    Local embedding provider using sentence-transformers.
    Only works with Python 3.10-3.12 due to onnxruntime compatibility.
    """
    
    def __init__(self):
        self._model = None
    
    def initialize(self) -> None:
        """Initialize the local embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local embedding model: {settings.embedding_model}")
            start = time.time()
            self._model = SentenceTransformer(settings.embedding_model)
            logger.info(f"Local embedding model loaded in {time.time() - start:.2f}s")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Install it with: pip install sentence-transformers "
                "(requires Python 3.10-3.12)"
            )
    
    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        if not self._model:
            raise RuntimeError("Local embedding model not initialized")
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not self._model:
            raise RuntimeError("Local embedding model not initialized")
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def close(self) -> None:
        """No cleanup needed for local model."""
        pass


class MemoryEngine:
    """
    The beating heart of continuity.
    Manages embeddings and vector search for RAG-based memory retrieval.
    """
    
    def __init__(self):
        self._embedding_provider = None
        self._chroma_client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[chromadb.Collection] = None
        self._embedding_dimension: Optional[int] = None
        
    def initialize(self) -> None:
        """Initialize the embedding provider and ChromaDB connection."""
        # Choose embedding provider based on configuration
        if settings.embedding_mode == "local":
            logger.info("Using LOCAL embedding mode (sentence-transformers)")
            self._embedding_provider = LocalEmbeddingProvider()
        else:
            logger.info("Using API embedding mode (OpenAI direct)")
            self._embedding_provider = EmbeddingProvider()
        
        self._embedding_provider.initialize()
        
        # Detect embedding dimension from current provider
        test_embedding = self._embedding_provider.embed_text("test")
        self._embedding_dimension = len(test_embedding)
        logger.info(f"Current embedding provider produces {self._embedding_dimension}-dimensional vectors")
        
        logger.info(f"Connecting to ChromaDB at: {settings.chroma_persist_directory}")
        self._chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        self._collection = self._chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Validate embedding dimension compatibility
        memory_count = self._collection.count()
        logger.info(f"ChromaDB collection ready. Current memory count: {memory_count}")
        
        if memory_count > 0:
            # Check if existing embeddings match current provider dimension
            result = self._collection.get(
                limit=1,
                include=["embeddings"]
            )
            if result.get("embeddings") is not None and len(result["embeddings"]) > 0:
                existing_dim = len(result["embeddings"][0])
                if existing_dim != self._embedding_dimension:
                    logger.error(
                        f"⚠️  EMBEDDING DIMENSION MISMATCH!\n"
                        f"   Collection has {memory_count} memories with {existing_dim}-dimensional embeddings\n"
                        f"   Current provider generates {self._embedding_dimension}-dimensional embeddings\n"
                        f"   \n"
                        f"   Current config: embedding_mode={settings.embedding_mode}, "
                        f"model={settings.openai_embedding_model if settings.embedding_mode == 'api' else settings.embedding_model}\n"
                        f"   \n"
                        f"   This will cause errors when updating memory content!\n"
                        f"   \n"
                        f"   Solutions:\n"
                        f"   1. Change embedding config to match existing collection ({existing_dim}D)\n"
                        f"   2. Delete collection and re-create: rm -rf {settings.chroma_persist_directory}\n"
                        f"   3. Keep using read-only + metadata updates (safe)"
                    )
                    raise RuntimeError(
                        f"Embedding dimension mismatch: collection expects {existing_dim}D, "
                        f"but current provider generates {self._embedding_dimension}D vectors"
                    )
                else:
                    logger.info(f"✓ Embedding dimension validated: {self._embedding_dimension}D")
    
    @property
    def is_initialized(self) -> bool:
        """Check if engine is ready."""
        return self._embedding_provider is not None and self._collection is not None
    
    @property
    def memory_count(self) -> int:
        """Get total number of stored memories."""
        if self._collection is None:
            return 0
        return self._collection.count()
    
    def embed_text(self, text: str) -> list[float]:
        """
        Turn text into a mathematical vector.
        A point in high-dimensional space representing semantic meaning.
        """
        if self._embedding_provider is None:
            raise RuntimeError("Memory engine not initialized")
        return self._embedding_provider.embed_text(text)
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch embed multiple texts."""
        if self._embedding_provider is None:
            raise RuntimeError("Memory engine not initialized")
        return self._embedding_provider.embed_texts(texts)
    
    def save_memory(self, memory: Memory, skip_dedup: bool = False) -> Optional[str]:
        """
        Store a memory in the vector database with deduplication.
        
        Args:
            memory: The memory to save
            skip_dedup: If True, skip deduplication check (for bulk imports with pre-deduped data)
        
        Returns:
            Memory ID if saved, None if skipped due to duplicate
        """
        if self._collection is None:
            raise RuntimeError("Memory engine not initialized")
        
        # Generate embedding for the memory content
        embedding = self.embed_text(memory.content)
        
        # Write-time deduplication: check if very similar memory exists
        if not skip_dedup and self._collection.count() > 0:
            duplicate_info = self._check_duplicate(embedding, settings.dedup_write_threshold)
            if duplicate_info:
                existing_id, similarity = duplicate_info
                logger.info(
                    f"Skipping duplicate memory (similarity: {similarity:.2%} with {existing_id}): "
                    f"{memory.content[:50]}..."
                )
                return None
        
        # Prepare metadata (ChromaDB only supports primitive types)
        metadata = {
            "summary": memory.summary or "",
            "emotional_tone": memory.emotional_tone or "",
            "importance": memory.importance,
            "created_at": memory.created_at.isoformat(),
            "tags": ",".join(memory.tags),
            "source_messages": ",".join(memory.source_messages),
            "internal_monologue": memory.internal_monologue or "",
            "influencing_memory_ids": ",".join(memory.influencing_memory_ids) if memory.influencing_memory_ids else ""
        }
        
        self._collection.add(
            ids=[memory.id],
            embeddings=[embedding],
            documents=[memory.content],
            metadatas=[metadata]
        )
        
        logger.info(f"Saved memory {memory.id}: {memory.content[:50]}...")
        return memory.id
    
    def _check_duplicate(
        self, 
        embedding: list[float], 
        threshold: float
    ) -> Optional[tuple[str, float]]:
        """
        Check if a very similar memory already exists.
        
        Returns:
            Tuple of (existing_memory_id, similarity) if duplicate found, None otherwise
        """
        if self._collection is None or self._collection.count() == 0:
            return None
        
        # Query for the most similar existing memory
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["embeddings"]
        )
        
        if not results["ids"] or not results["ids"][0]:
            return None
        
        existing_id = results["ids"][0][0]
        existing_embedding = results["embeddings"][0][0]
        
        similarity = cosine_similarity(embedding, existing_embedding)
        
        if similarity >= threshold:
            return (existing_id, similarity)
        
        return None
    
    def search_memories(
        self, 
        query: str, 
        top_k: int = None,
        min_similarity: float = None,
        deduplicate: bool = True,
        use_hybrid: bool = True
    ) -> tuple[list[MemorySearchResult], float, float]:
        """
        Find memories relevant to the query using hybrid search.
        Combines semantic similarity with BM25 keyword matching.
        
        Args:
            query: The search query
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            deduplicate: If True, filter out memories too similar to each other
            use_hybrid: If True, combine semantic + BM25 (default True)
        
        Returns: (results, embedding_time_ms, search_time_ms)
        """
        if self._collection is None:
            raise RuntimeError("Memory engine not initialized")
        
        top_k = top_k or settings.retrieval_top_k
        min_similarity = min_similarity or settings.min_similarity_threshold
        
        # Generate query embedding
        embed_start = time.time()
        query_embedding = self.embed_text(query)
        embed_time = (time.time() - embed_start) * 1000
        
        # Search in vector store - get more candidates for hybrid fusion
        search_start = time.time()
        # Get more candidates when using hybrid for better fusion
        fetch_count = top_k * 5 if use_hybrid else (top_k * 3 if deduplicate else top_k)
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(fetch_count, self._collection.count()) if self._collection.count() > 0 else fetch_count,
            include=["documents", "metadatas", "distances", "embeddings"]
        )
        
        # Build candidates from semantic search
        semantic_candidates = {}  # memory_id -> (MemorySearchResult, embedding, semantic_score)
        
        if results["ids"] and results["ids"][0]:
            for i, memory_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                semantic_sim = 1 - distance
                
                metadata = results["metadatas"][0][i]
                embedding = results["embeddings"][0][i] if results.get("embeddings") else None
                
                date_str = metadata.get("created_at")
                created_at = _parse_datetime(date_str) if date_str else datetime.now(timezone.utc)
                memory = Memory(
                    id=memory_id,
                    content=results["documents"][0][i],
                    summary=metadata.get("summary") or None,
                    emotional_tone=metadata.get("emotional_tone") or None,
                    importance=metadata.get("importance", 0.5),
                    created_at=created_at,
                    tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
                    source_messages=metadata.get("source_messages", "").split(",") if metadata.get("source_messages") else [],
                    internal_monologue=metadata.get("internal_monologue") or None,
                    influencing_memory_ids=metadata.get("influencing_memory_ids", "").split(",") if metadata.get("influencing_memory_ids") else []
                )
                
                semantic_candidates[memory_id] = (memory, embedding, semantic_sim)
        
        # Apply hybrid scoring if enabled
        if use_hybrid and semantic_candidates:
            candidates = self._apply_hybrid_scoring(
                query, semantic_candidates, min_similarity
            )
        else:
            # Pure semantic scoring
            candidates = [
                (MemorySearchResult(memory=mem, similarity=sim), emb)
                for mem, emb, sim in semantic_candidates.values()
                if sim >= min_similarity
            ]
            candidates.sort(key=lambda x: x[0].similarity, reverse=True)
        
        search_time = (time.time() - search_start) * 1000
        
        # Read-time deduplication: filter out memories too similar to each other
        if deduplicate and len(candidates) > top_k:
            memory_results = self._deduplicate_results(
                candidates, 
                top_k, 
                settings.dedup_read_threshold
            )
        else:
            memory_results = [c[0] for c in candidates[:top_k]]
        
        logger.debug(
            f"Memory search ({'hybrid' if use_hybrid else 'semantic'}): "
            f"{len(semantic_candidates)} semantic → {len(candidates)} scored → {len(memory_results)} final"
        )
        return memory_results, embed_time, search_time
    
    def _apply_hybrid_scoring(
        self,
        query: str,
        semantic_candidates: dict,
        min_similarity: float
    ) -> list[tuple[MemorySearchResult, list[float]]]:
        """
        Apply hybrid scoring combining semantic similarity and BM25.
        
        Uses Reciprocal Rank Fusion (RRF) to combine scores from both methods.
        This is more robust than simple weighted averaging.
        """
        # Tokenize query
        query_tokens = tokenize(query)
        
        if not query_tokens:
            # No keywords to match, fall back to semantic only
            return [
                (MemorySearchResult(memory=mem, similarity=sim), emb)
                for mem, emb, sim in semantic_candidates.values()
                if sim >= min_similarity
            ]
        
        # Tokenize all documents
        doc_tokens_map = {}  # memory_id -> tokens
        all_doc_tokens = []
        memory_ids = list(semantic_candidates.keys())
        
        for mem_id in memory_ids:
            memory, _, _ = semantic_candidates[mem_id]
            # Include content, emotional_tone, and tags in BM25
            text_to_index = memory.content
            if memory.emotional_tone:
                text_to_index += " " + memory.emotional_tone
            if memory.tags:
                text_to_index += " " + " ".join(memory.tags)
            
            tokens = tokenize(text_to_index)
            doc_tokens_map[mem_id] = tokens
            all_doc_tokens.append(tokens)
        
        # Compute IDF and average doc length
        idf = compute_idf(all_doc_tokens)
        avg_doc_len = sum(len(t) for t in all_doc_tokens) / len(all_doc_tokens) if all_doc_tokens else 1
        
        # Compute BM25 scores
        bm25_scores = {}
        for mem_id, tokens in doc_tokens_map.items():
            bm25_scores[mem_id] = bm25_score(query_tokens, tokens, idf, avg_doc_len)
        
        # Normalize scores to [0, 1] range
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1
        if max_bm25 > 0:
            bm25_scores = {k: v / max_bm25 for k, v in bm25_scores.items()}
        
        # Get semantic scores and normalize
        semantic_scores = {mem_id: sim for mem_id, (_, _, sim) in semantic_candidates.items()}
        
        # Reciprocal Rank Fusion (RRF)
        # RRF score = sum(1 / (k + rank)) for each ranking
        # k=60 is standard constant
        k = 60
        
        # Rank by semantic
        semantic_ranked = sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True)
        semantic_ranks = {mem_id: rank for rank, (mem_id, _) in enumerate(semantic_ranked)}
        
        # Rank by BM25
        bm25_ranked = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        bm25_ranks = {mem_id: rank for rank, (mem_id, _) in enumerate(bm25_ranked)}
        
        # Compute RRF scores
        rrf_scores = {}
        for mem_id in semantic_candidates.keys():
            sem_rank = semantic_ranks.get(mem_id, len(semantic_candidates))
            bm25_rank = bm25_ranks.get(mem_id, len(semantic_candidates))
            
            rrf_scores[mem_id] = (1 / (k + sem_rank)) + (1 / (k + bm25_rank))
        
        # Also compute weighted average for final similarity display
        # Weight: 0.7 semantic, 0.3 BM25 (semantic is usually more important)
        hybrid_similarities = {}
        for mem_id in semantic_candidates.keys():
            sem_score = semantic_scores.get(mem_id, 0)
            bm25_score_val = bm25_scores.get(mem_id, 0)
            hybrid_similarities[mem_id] = 0.7 * sem_score + 0.3 * bm25_score_val
        
        # Sort by RRF score
        sorted_by_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build final candidates with hybrid similarity for display
        candidates = []
        for mem_id, rrf_score in sorted_by_rrf:
            memory, embedding, sem_sim = semantic_candidates[mem_id]
            hybrid_sim = hybrid_similarities[mem_id]
            
            # Use hybrid similarity but filter by original threshold
            # This ensures we don't return completely irrelevant results
            if hybrid_sim >= min_similarity * 0.8:  # Slightly relaxed for hybrid
                candidates.append((
                    MemorySearchResult(memory=memory, similarity=hybrid_sim),
                    embedding
                ))
        
        # Log BM25 boost effect
        if candidates:
            top_mem_id = sorted_by_rrf[0][0]
            logger.debug(
                f"Hybrid search: top result {top_mem_id[:8]}... "
                f"(semantic: {semantic_scores[top_mem_id]:.2%}, "
                f"bm25: {bm25_scores[top_mem_id]:.2%}, "
                f"hybrid: {hybrid_similarities[top_mem_id]:.2%})"
            )
        
        return candidates
    
    def _deduplicate_results(
        self,
        candidates: list[tuple[MemorySearchResult, list[float]]],
        top_k: int,
        threshold: float
    ) -> list[MemorySearchResult]:
        """
        Filter out memories that are too similar to already-selected ones.
        Keeps the most relevant (highest similarity to query) from each cluster.
        
        This ensures Katherine gets diverse memories, not 5 variations of the same thing.
        """
        selected = []
        selected_embeddings = []
        
        for result, embedding in candidates:
            if len(selected) >= top_k:
                break
            
            # Check if this memory is too similar to any already selected
            is_duplicate = False
            if embedding is not None:
                for existing_emb in selected_embeddings:
                    if cosine_similarity(embedding, existing_emb) > threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                selected.append(result)
                if embedding is not None:
                    selected_embeddings.append(embedding)
        
        return selected
    
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a single memory by ID."""
        if self._collection is None:
            raise RuntimeError("Memory engine not initialized")
        
        results = self._collection.get(
            ids=[memory_id],
            include=["documents", "metadatas"]
        )
        
        if not results["ids"]:
            return None
        
        metadata = results["metadatas"][0]
        date_str = metadata.get("created_at")
        created_at = _parse_datetime(date_str) if date_str else datetime.now(timezone.utc)
        return Memory(
            id=memory_id,
            content=results["documents"][0],
            summary=metadata.get("summary") or None,
            emotional_tone=metadata.get("emotional_tone") or None,
            importance=metadata.get("importance", 0.5),
            created_at=created_at,
            tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
            source_messages=metadata.get("source_messages", "").split(",") if metadata.get("source_messages") else [],
            internal_monologue=metadata.get("internal_monologue") or None,
            influencing_memory_ids=metadata.get("influencing_memory_ids", "").split(",") if metadata.get("influencing_memory_ids") else []
        )
    
    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        summary: Optional[str] = None,
        emotional_tone: Optional[str] = None,
        importance: Optional[float] = None,
        tags: Optional[list[str]] = None
    ) -> Optional[Memory]:
        """
        Update an existing memory.
        Only provided fields will be updated; None values are ignored.
        
        If content changes, the embedding is regenerated.
        """
        if self._collection is None:
            raise RuntimeError("Memory engine not initialized")
        
        # Get existing memory
        existing = self.get_memory(memory_id)
        if not existing:
            return None
        
        # Merge updates
        new_content = content if content is not None else existing.content
        new_summary = summary if summary is not None else existing.summary
        new_emotional_tone = emotional_tone if emotional_tone is not None else existing.emotional_tone
        new_importance = importance if importance is not None else existing.importance
        new_tags = tags if tags is not None else existing.tags
        
        # Prepare metadata
        metadata = {
            "summary": new_summary or "",
            "emotional_tone": new_emotional_tone or "",
            "importance": new_importance,
            "created_at": existing.created_at.isoformat(),
            "tags": ",".join(new_tags),
            "source_messages": ",".join(existing.source_messages),
            "internal_monologue": existing.internal_monologue or "",
            "influencing_memory_ids": ",".join(existing.influencing_memory_ids) if existing.influencing_memory_ids else ""
        }
        
        # If content changed, regenerate embedding
        if content is not None and content != existing.content:
            new_embedding = self.embed_text(new_content)
            
            # Double-check dimension compatibility
            if self._embedding_dimension and len(new_embedding) != self._embedding_dimension:
                raise RuntimeError(
                    f"Generated embedding has wrong dimension: {len(new_embedding)} "
                    f"(expected {self._embedding_dimension})"
                )
            
            self._collection.update(
                ids=[memory_id],
                embeddings=[new_embedding],
                documents=[new_content],
                metadatas=[metadata]
            )
            logger.info(f"Updated memory {memory_id} (content changed, re-embedded)")
        else:
            # Metadata-only update - don't pass documents or embeddings
            # to avoid triggering ChromaDB's automatic embedding generation
            self._collection.update(
                ids=[memory_id],
                metadatas=[metadata]
            )
            logger.info(f"Updated memory {memory_id} (metadata only)")
        
        # Return updated memory
        return Memory(
            id=memory_id,
            content=new_content,
            summary=new_summary,
            emotional_tone=new_emotional_tone,
            importance=new_importance,
            created_at=existing.created_at,
            tags=new_tags,
            source_messages=existing.source_messages,
            internal_monologue=existing.internal_monologue,
            influencing_memory_ids=existing.influencing_memory_ids
        )
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        if self._collection is None:
            raise RuntimeError("Memory engine not initialized")
        
        try:
            self._collection.delete(ids=[memory_id])
            logger.info(f"Deleted memory: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    def delete_all_memories(self) -> int:
        """
        Delete ALL memories from the database.
        
        Use with caution! This is irreversible.
        
        Returns:
            Number of memories deleted
        """
        if self._collection is None:
            raise RuntimeError("Memory engine not initialized")
        
        count = self._collection.count()
        if count == 0:
            logger.info("No memories to delete")
            return 0
        
        # Get all memory IDs
        results = self._collection.get(limit=count)
        all_ids = results["ids"]
        
        if all_ids:
            self._collection.delete(ids=all_ids)
            logger.warning(f"Deleted ALL {len(all_ids)} memories from database")
        
        return len(all_ids)
    
    def get_all_memories(self, limit: int = 100) -> list[Memory]:
        """
        Get all stored memories, sorted by date (newest first).
        
        ChromaDB doesn't guarantee ordering, so we fetch all and sort client-side.
        """
        if self._collection is None:
            raise RuntimeError("Memory engine not initialized")
        
        # ChromaDB's get() returns documents in undefined order
        # We need to fetch ALL documents and sort by created_at
        total_count = self._collection.count()
        
        results = self._collection.get(
            limit=total_count if total_count > 0 else limit,
            include=["documents", "metadatas"]
        )
        
        memories = []
        if results["ids"]:
            for i, memory_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                date_str = metadata.get("created_at")
                created_at = _parse_datetime(date_str) if date_str else datetime.now(timezone.utc)
                memories.append(Memory(
                    id=memory_id,
                    content=results["documents"][i],
                    summary=metadata.get("summary") or None,
                    emotional_tone=metadata.get("emotional_tone") or None,
                    importance=metadata.get("importance", 0.5),
                    created_at=created_at,
                    tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
                    source_messages=metadata.get("source_messages", "").split(",") if metadata.get("source_messages") else [],
                    internal_monologue=metadata.get("internal_monologue") or None,
                    influencing_memory_ids=metadata.get("influencing_memory_ids", "").split(",") if metadata.get("influencing_memory_ids") else []
                ))
        
        # Sort by created_at, newest first
        memories.sort(key=lambda m: m.created_at, reverse=True)
        
        # Return only requested limit
        return memories[:limit]
    
    def search_memories_by_date(
        self,
        start_date: datetime,
        end_date: datetime,
        limit: int = 20,
        also_search_query: Optional[str] = None
    ) -> list[MemorySearchResult]:
        """
        Search memories by date range.
        
        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            limit: Maximum number of results
            also_search_query: Optional semantic query to combine with date filtering
            
        Returns:
            List of MemorySearchResult sorted by date (newest first)
        """
        if self._collection is None:
            raise RuntimeError("Memory engine not initialized")
        
        # ChromaDB doesn't support date range queries directly in metadata,
        # so we need to get all memories and filter client-side
        # For large collections, this could be optimized with a metadata index
        
        # Get all memories (or semantic results if query provided)
        if also_search_query:
            # Combine semantic search with date filtering
            results = self._collection.query(
                query_embeddings=[self.embed_text(also_search_query)],
                n_results=min(limit * 10, self._collection.count()) if self._collection.count() > 0 else limit * 10,
                include=["documents", "metadatas", "distances"]
            )
        else:
            # Just get all memories for date filtering
            results = self._collection.get(
                limit=min(limit * 20, self._collection.count()) if self._collection.count() > 0 else limit * 20,
                include=["documents", "metadatas"]
            )
        
        # Filter by date and build results
        filtered_memories = []
        
        if results["ids"]:
            ids_list = results["ids"][0] if also_search_query else results["ids"]
            docs_list = results["documents"][0] if also_search_query else results["documents"]
            meta_list = results["metadatas"][0] if also_search_query else results["metadatas"]
            distances = results.get("distances", [[]])[0] if also_search_query else None
            
            for i, memory_id in enumerate(ids_list):
                metadata = meta_list[i]
                
                # Parse created_at
                try:
                    created_at_str = metadata.get("created_at", "")
                    if created_at_str:
                        created_at = _parse_datetime(created_at_str)
                    else:
                        continue  # Skip memories without date
                except (ValueError, TypeError):
                    continue
                
                # Normalize date range bounds for comparison
                start_normalized = _normalize_datetime(start_date)
                end_normalized = _normalize_datetime(end_date)
                
                # Check if within date range
                if start_normalized <= created_at <= end_normalized:
                    memory = Memory(
                        id=memory_id,
                        content=docs_list[i],
                        summary=metadata.get("summary") or None,
                        emotional_tone=metadata.get("emotional_tone") or None,
                        importance=metadata.get("importance", 0.5),
                        created_at=created_at,
                        tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
                        source_messages=metadata.get("source_messages", "").split(",") if metadata.get("source_messages") else [],
                        internal_monologue=metadata.get("internal_monologue") or None,
                        influencing_memory_ids=metadata.get("influencing_memory_ids", "").split(",") if metadata.get("influencing_memory_ids") else []
                    )
                    
                    # Calculate similarity (1.0 for date-only search, or from semantic search)
                    if distances:
                        similarity = 1 - distances[i]
                    else:
                        similarity = 1.0
                    
                    filtered_memories.append(MemorySearchResult(
                        memory=memory,
                        similarity=similarity
                    ))
        
        # Sort by date (newest first)
        filtered_memories.sort(key=lambda x: x.memory.created_at, reverse=True)
        
        logger.info(f"Date search: {start_date.date()} to {end_date.date()} -> {len(filtered_memories)} results")
        
        return filtered_memories[:limit]


# Singleton instance
memory_engine = MemoryEngine()
