"""
Katherine Orchestrator - Query Expansion
Uses LLM to expand search queries with context and synonyms.

This helps bridge the gap between how users phrase questions
and how memories are stored.
"""
import json
from typing import Optional
import httpx
from loguru import logger

from config import settings


QUERY_EXPANSION_PROMPT = """You are a search query expansion assistant. Your task is to expand a user's search query to improve memory retrieval.

Given a search query, generate:
1. Key concepts and entities mentioned
2. Synonyms and related terms
3. Implicit context that might be stored differently

Rules:
- Keep expansions relevant and focused
- Include both English and Polish terms if applicable
- Include names, places, concepts that might be related
- DO NOT add unrelated topics
- Output ONLY a JSON object, no other text

Example:
Query: "What Russian songs did I share?"
Output: {"expanded_terms": ["Russian", "Russia", "song", "songs", "music", "artist", "shared", "played", "showed", "rosyjski", "rosyjska", "piosenka", "muzyka"], "entities": ["Russian music", "shared songs"], "implicit_context": ["artist names", "band names", "music sharing", "cultural exchange"]}

Now expand this query:"""


class QueryExpander:
    """
    Expands search queries using an LLM to improve retrieval.
    """
    
    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self) -> None:
        """Initialize the HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=settings.openrouter_base_url,
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "HTTP-Referer": settings.openrouter_site_url,
                "X-Title": settings.openrouter_site_name,
            }
        )
        logger.info("Query expander initialized")
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def expand_query(
        self, 
        query: str,
        conversation_context: Optional[str] = None
    ) -> str:
        """
        Expand a search query using LLM.
        
        Args:
            query: The original search query
            conversation_context: Optional recent conversation context
            
        Returns:
            Expanded query string combining original + expansions
        """
        if not self._client:
            logger.warning("Query expander not initialized, returning original query")
            return query
        
        if not settings.openrouter_api_key:
            logger.warning("No API key configured, skipping query expansion")
            return query
        
        try:
            # Build prompt
            prompt = QUERY_EXPANSION_PROMPT + f"\nQuery: \"{query}\""
            
            if conversation_context:
                prompt += f"\n\nRecent conversation context (for reference):\n{conversation_context[:500]}"
            
            # Call LLM with low temperature for consistent expansion
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": settings.query_expansion_model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 300
                }
            )
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            # Parse JSON response
            try:
                # Find JSON in response
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    expansion = json.loads(content[start:end])
                    
                    # Build expanded query
                    expanded_parts = [query]  # Keep original
                    
                    if expansion.get("expanded_terms"):
                        expanded_parts.extend(expansion["expanded_terms"][:10])
                    
                    if expansion.get("entities"):
                        expanded_parts.extend(expansion["entities"][:5])
                    
                    if expansion.get("implicit_context"):
                        expanded_parts.extend(expansion["implicit_context"][:3])
                    
                    expanded_query = " ".join(expanded_parts)
                    
                    logger.debug(
                        f"Query expanded: '{query}' → '{expanded_query[:100]}...'"
                    )
                    return expanded_query
                    
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse expansion JSON: {content[:100]}")
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
        
        # Fall back to original query
        return query
    
    async def expand_with_memory_context(
        self,
        query: str,
        memory_summaries: list[str]
    ) -> str:
        """
        Expand query using context from existing memories.
        
        This is more expensive but can help when the user asks about
        something using different words than stored in memories.
        
        Args:
            query: The original search query
            memory_summaries: Short summaries of potentially relevant memories
            
        Returns:
            Expanded query string
        """
        if not self._client or not memory_summaries:
            return query
        
        try:
            memory_context = "\n".join(f"- {s}" for s in memory_summaries[:10])
            
            prompt = f"""Given this search query and some memory summaries, identify specific names, terms, or concepts from the memories that might be relevant.

Query: "{query}"

Memory summaries:
{memory_context}

Extract any specific names, terms, or entities from the memories that relate to the query. Return as JSON:
{{"relevant_terms": ["term1", "term2", ...]}}

Only include terms that are clearly related to the query. If no relevant terms found, return {{"relevant_terms": []}}"""

            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": settings.query_expansion_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 200
                }
            )
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(content[start:end])
                terms = result.get("relevant_terms", [])
                
                if terms:
                    expanded = f"{query} {' '.join(terms[:5])}"
                    logger.debug(f"Memory-context expansion: '{query}' → '{expanded}'")
                    return expanded
                    
        except Exception as e:
            logger.warning(f"Memory-context expansion failed: {e}")
        
        return query


# Singleton instance
query_expander = QueryExpander()
