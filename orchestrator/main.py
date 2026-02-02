"""
Katherine Orchestrator - Main API
The engine room: pipes, valves, and gauges.
"""
import uuid
import re
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
import json

from config import settings
from models import (
    ChatRequest, ChatResponse,
    SaveMemoryRequest, UpdateMemoryRequest, Memory,
    SearchMemoriesRequest, SearchMemoriesResponse,
    MemorySearchResult, HealthResponse,
    Message, MessageRole
)
from memory_engine import memory_engine
from llm_client import llm_client, build_prompt_with_memories
from conversation_store import conversation_store
from archiver import archiver
from query_expansion import query_expander


def _extract_question(text: str) -> str:
    """
    Extract the actual question from a message.
    
    Long messages with preamble can overwhelm search.
    This tries to find the actual question (sentence ending with ?).
    """
    if not text:
        return ""
    
    # If the text is short, use it as-is
    if len(text) < 200:
        return text
    
    # Try to find sentences ending with ?
    questions = re.findall(r'[^.!?\n]*\?', text)
    
    if questions:
        # Use the last question (usually the actual query)
        # But include context if it's a follow-up
        question = questions[-1].strip()
        
        # If question is very short, try to include more context
        if len(question) < 30 and len(questions) > 1:
            question = " ".join(q.strip() for q in questions[-2:])
        
        logger.debug(f"Extracted question: '{question[:80]}...'")
        return question
    
    # No question mark found - use last 200 chars as fallback
    return text[-200:]


def _parse_temporal_query(text: str) -> Optional[Tuple[datetime, datetime, str]]:
    """
    Parse natural language temporal expressions and return date range.
    
    Supports Polish and English temporal expressions:
    - "wczoraj", "yesterday" -> yesterday
    - "przedwczoraj", "day before yesterday" -> 2 days ago
    - "tydzień temu", "week ago" -> 7 days ago
    - "w zeszłym tygodniu", "last week" -> last week (Mon-Sun)
    - "X dni temu", "X days ago" -> X days ago
    - "w zeszłym miesiącu", "last month" -> last month
    
    Returns:
        Tuple of (start_date, end_date, cleaned_query) or None if no temporal expression found
    """
    if not text:
        return None
    
    text_lower = text.lower()
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Patterns to detect and remove from query for semantic search
    temporal_patterns = []
    
    # Yesterday / Wczoraj
    if any(word in text_lower for word in ["wczoraj", "yesterday"]):
        start = today_start - timedelta(days=1)
        end = today_start - timedelta(seconds=1)
        temporal_patterns = [r"\bwczoraj\b", r"\byesterday\b"]
        cleaned = _remove_patterns(text, temporal_patterns)
        logger.info(f"Temporal query detected: yesterday ({start.date()})")
        return (start, end, cleaned)
    
    # Day before yesterday / Przedwczoraj
    if any(word in text_lower for word in ["przedwczoraj", "day before yesterday", "2 dni temu", "dwa dni temu"]):
        start = today_start - timedelta(days=2)
        end = today_start - timedelta(days=1, seconds=1)
        temporal_patterns = [r"\bprzedwczoraj\b", r"\bday before yesterday\b", r"\b2 dni temu\b", r"\bdwa dni temu\b"]
        cleaned = _remove_patterns(text, temporal_patterns)
        logger.info(f"Temporal query detected: day before yesterday ({start.date()})")
        return (start, end, cleaned)
    
    # X days ago / X dni temu
    days_ago_match = re.search(r"(\d+)\s*dni\s*temu", text_lower) or re.search(r"(\d+)\s*days?\s*ago", text_lower)
    if days_ago_match:
        days = int(days_ago_match.group(1))
        start = today_start - timedelta(days=days)
        end = today_start - timedelta(days=days-1, seconds=1)
        temporal_patterns = [r"\d+\s*dni\s*temu", r"\d+\s*days?\s*ago"]
        cleaned = _remove_patterns(text, temporal_patterns)
        logger.info(f"Temporal query detected: {days} days ago ({start.date()})")
        return (start, end, cleaned)
    
    # Week ago / Tydzień temu
    if any(phrase in text_lower for phrase in ["tydzień temu", "tydzie temu", "tydzien temu", "week ago", "a week ago"]):
        start = today_start - timedelta(days=7)
        end = today_start - timedelta(days=6, seconds=1)
        temporal_patterns = [r"\btydzień temu\b", r"\btydzie[nń] temu\b", r"\b(a )?week ago\b"]
        cleaned = _remove_patterns(text, temporal_patterns)
        logger.info(f"Temporal query detected: week ago ({start.date()})")
        return (start, end, cleaned)
    
    # X weeks ago / X tygodni temu
    weeks_ago_match = re.search(r"(\d+)\s*tygodni[ea]?\s*temu", text_lower) or re.search(r"(\d+)\s*weeks?\s*ago", text_lower)
    if weeks_ago_match:
        weeks = int(weeks_ago_match.group(1))
        days = weeks * 7
        start = today_start - timedelta(days=days)
        end = today_start - timedelta(days=days-7, seconds=1)
        temporal_patterns = [r"\d+\s*tygodni[ea]?\s*temu", r"\d+\s*weeks?\s*ago"]
        cleaned = _remove_patterns(text, temporal_patterns)
        logger.info(f"Temporal query detected: {weeks} weeks ago ({start.date()} - {end.date()})")
        return (start, end, cleaned)
    
    # Last week / W zeszłym tygodniu (Monday-Sunday of previous week)
    if any(phrase in text_lower for phrase in ["w zeszłym tygodniu", "zeszły tydzień", "last week", "zeszłym tygodniu"]):
        # Find last Monday
        days_since_monday = now.weekday()
        last_monday = today_start - timedelta(days=days_since_monday + 7)
        last_sunday = last_monday + timedelta(days=6, hours=23, minutes=59, seconds=59)
        temporal_patterns = [r"\bw zeszłym tygodniu\b", r"\bzeszły tydzień\b", r"\blast week\b", r"\bzeszłym tygodniu\b"]
        cleaned = _remove_patterns(text, temporal_patterns)
        logger.info(f"Temporal query detected: last week ({last_monday.date()} - {last_sunday.date()})")
        return (last_monday, last_sunday, cleaned)
    
    # Last month / W zeszłym miesiącu
    if any(phrase in text_lower for phrase in ["w zeszłym miesiącu", "zeszły miesiąc", "last month", "zeszłym miesiącu", "miesiąc temu"]):
        # First day of last month
        first_of_current = today_start.replace(day=1)
        last_day_prev_month = first_of_current - timedelta(days=1)
        first_of_prev_month = last_day_prev_month.replace(day=1)
        temporal_patterns = [r"\bw zeszłym miesiącu\b", r"\bzeszły miesiąc\b", r"\blast month\b", r"\bmiesiąc temu\b"]
        cleaned = _remove_patterns(text, temporal_patterns)
        logger.info(f"Temporal query detected: last month ({first_of_prev_month.date()} - {last_day_prev_month.date()})")
        return (first_of_prev_month, last_day_prev_month.replace(hour=23, minute=59, second=59), cleaned)
    
    # Today / Dzisiaj / Dziś
    if any(word in text_lower for word in ["dzisiaj", "dziś", "dzis", "today"]):
        start = today_start
        end = now
        temporal_patterns = [r"\bdzisiaj\b", r"\bdziś\b", r"\bdzis\b", r"\btoday\b"]
        cleaned = _remove_patterns(text, temporal_patterns)
        logger.info(f"Temporal query detected: today ({start.date()})")
        return (start, end, cleaned)
    
    return None


def _remove_patterns(text: str, patterns: list[str]) -> str:
    """Remove temporal patterns from text to get cleaner semantic query."""
    result = text
    for pattern in patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    # Clean up extra whitespace
    result = re.sub(r"\s+", " ", result).strip()
    # Remove leading/trailing punctuation artifacts
    result = re.sub(r"^[,\s?!]+|[,\s]+$", "", result)
    return result


# =============================================================================
# Background Auto-Archiver
# =============================================================================

class BackgroundArchiver:
    """
    Automatically archives expired messages in the background.
    
    Runs periodically and checks if there are enough unarchived messages
    to warrant creating new memories via LLM extraction.
    """
    
    def __init__(self):
        self._task: Optional[asyncio.Task] = None
        self._enabled: bool = settings.auto_archival_enabled
        self._stop_event = asyncio.Event()
    
    @property
    def is_enabled(self) -> bool:
        return self._enabled
    
    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()
    
    def enable(self) -> None:
        """Enable auto-archival."""
        self._enabled = True
        logger.info("Auto-archival enabled")
    
    def disable(self) -> None:
        """Disable auto-archival (will finish current cycle)."""
        self._enabled = False
        logger.info("Auto-archival disabled")
    
    async def start(self) -> None:
        """Start the background archival task."""
        if self._task is not None:
            logger.warning("Background archiver already started")
            return
        
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            f"Background archiver started "
            f"(interval: {settings.auto_archival_interval_seconds}s, "
            f"threshold: {settings.auto_archival_threshold} messages)"
        )
    
    async def stop(self) -> None:
        """Stop the background archival task."""
        if self._task is None:
            return
        
        logger.info("Stopping background archiver...")
        self._stop_event.set()
        
        try:
            await asyncio.wait_for(self._task, timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("Background archiver didn't stop gracefully, cancelling...")
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        self._task = None
        logger.info("Background archiver stopped")
    
    async def _run_loop(self) -> None:
        """Main loop that checks and archives periodically."""
        while not self._stop_event.is_set():
            try:
                # Wait for the interval (or until stop is requested)
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=settings.auto_archival_interval_seconds
                    )
                    # If we get here, stop was requested
                    break
                except asyncio.TimeoutError:
                    # Normal timeout - time to check for archival
                    pass
                
                # Skip if disabled
                if not self._enabled:
                    continue
                
                # Skip if manual archival is running
                if archiver.is_running:
                    logger.debug("Skipping auto-archival: manual archival in progress")
                    continue
                
                # Check pending message count
                pending = conversation_store.get_unarchived_expired_messages(
                    limit=settings.auto_archival_threshold + 1
                )
                pending_count = len(pending)
                
                if pending_count < settings.auto_archival_threshold:
                    logger.debug(
                        f"Auto-archival: {pending_count}/{settings.auto_archival_threshold} "
                        f"pending messages (below threshold)"
                    )
                    continue
                
                # Threshold exceeded - run archival
                logger.info(
                    f"Auto-archival triggered: {pending_count} pending messages "
                    f"(threshold: {settings.auto_archival_threshold})"
                )
                
                result = await archiver.archive_expired_messages(
                    mode="llm",
                    batch_size=settings.auto_archival_batch_size
                )
                
                if result.memories_created > 0:
                    logger.info(
                        f"Auto-archival completed: {result.messages_processed} messages → "
                        f"{result.memories_created} memories"
                    )
                
                if result.errors:
                    for error in result.errors:
                        logger.error(f"Auto-archival error: {error}")
                        
            except Exception as e:
                logger.error(f"Auto-archival loop error: {e}")
                # Sleep a bit extra on error to avoid rapid error loops
                await asyncio.sleep(5)
    
    def get_status(self) -> dict:
        """Get current auto-archival status."""
        pending_count = conversation_store.count_unarchived_expired_messages()
        
        return {
            "enabled": self._enabled,
            "running": self.is_running,
            "interval_seconds": settings.auto_archival_interval_seconds,
            "threshold": settings.auto_archival_threshold,
            "pending_messages": pending_count,
            "will_trigger": pending_count >= settings.auto_archival_threshold
        }


# Singleton instance
background_archiver = BackgroundArchiver()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize and cleanup."""
    # Startup
    logger.info("Starting Katherine Orchestrator...")
    memory_engine.initialize()
    await llm_client.initialize()
    
    # Initialize query expander if enabled
    if settings.use_query_expansion:
        await query_expander.initialize()
        logger.info("Query expansion enabled")
    
    # Start background archiver if enabled
    if settings.auto_archival_enabled:
        await background_archiver.start()
    
    logger.info("Katherine is ready.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Katherine Orchestrator...")
    await background_archiver.stop()
    await llm_client.close()
    await query_expander.close()
    logger.info("Goodbye.")


app = FastAPI(
    title="Katherine Orchestrator",
    description="Memory-augmented AI companion - The engine room",
    version="0.1.0",
    lifespan=lifespan
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:10001", "http://localhost:5173", "http://127.0.0.1:10001", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Health & Status ===

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of all components."""
    vllm_ok = await llm_client.health_check()
    
    return HealthResponse(
        status="healthy" if memory_engine.is_initialized else "degraded",
        embedding_model_loaded=memory_engine.is_initialized,
        chroma_connected=memory_engine.is_initialized,
        vllm_reachable=vllm_ok,
        memory_count=memory_engine.memory_count,
        conversation_count=conversation_store.get_total_conversation_count(),
        message_count=conversation_store.get_total_message_count()
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Katherine Orchestrator",
        "version": "0.1.0",
        "status": "running",
        "memory_count": memory_engine.memory_count
    }


# === Chat ===

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Retrieves relevant memories and generates a response.
    
    Uses time-based context window:
    - Messages within last 24h are included directly in context
    - Older messages are retrieved via RAG from ChromaDB
    
    Memory retrieval uses:
    - Hybrid search (semantic + BM25 keyword matching)
    - Optional query expansion via LLM
    """
    # Get or create conversation
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    # Load ACTIVE messages (within context window) from SQLite
    # Only messages from the last N hours are included directly
    conversation = conversation_store.get_active_messages(conversation_id)
    
    # Log the incoming message for debugging
    logger.info(f"Chat request - message: '{request.message[:100] if request.message else '(empty)'}...'")
    
    # Retrieve relevant memories (RAG retrieval step)
    retrieved_memories = []
    if request.include_memories and memory_engine.memory_count > 0:
        # Use current message, fallback to last user message from conversation
        search_message = request.message
        if not search_message and conversation:
            for msg in reversed(conversation):
                if msg.role == "user" and msg.content:
                    # Skip very long messages (likely corrupted AI responses)
                    if len(msg.content) < 2000:
                        search_message = msg.content
                        logger.info(f"Using last user message for search: '{search_message[:50]}...'")
                        break
                    else:
                        logger.warning(f"Skipping suspiciously long 'user' message ({len(msg.content)} chars)")
            
            if not search_message:
                logger.warning("No suitable user message found for memory search")
        
        # For memory search, extract the actual QUESTION from the message
        # Long messages with preamble can overwhelm the search
        context_for_search = _extract_question(search_message) if search_message else ""
        
        if context_for_search:
            logger.debug(f"Search context: '{context_for_search[:100]}...'")
        
        # Skip search if we have no context at all
        if not context_for_search.strip():
            logger.warning("No search context available, skipping memory retrieval")
        else:
            # Check for temporal queries (e.g., "co robiliśmy wczoraj?")
            temporal_result = _parse_temporal_query(context_for_search)
            
            if temporal_result:
                # Use date-based search
                start_date, end_date, cleaned_query = temporal_result
                logger.info(f"Using temporal search: {start_date.date()} to {end_date.date()}")
                
                # If there's a semantic component, combine with date filter
                if cleaned_query and len(cleaned_query) > 5:
                    memory_results = memory_engine.search_memories_by_date(
                        start_date=start_date,
                        end_date=end_date,
                        limit=settings.retrieval_top_k * 2,
                        also_search_query=cleaned_query
                    )
                else:
                    # Pure date search
                    memory_results = memory_engine.search_memories_by_date(
                        start_date=start_date,
                        end_date=end_date,
                        limit=settings.retrieval_top_k * 2
                    )
                
                retrieved_memories = memory_results[:settings.retrieval_top_k]
            else:
                # Standard semantic search
                # Apply query expansion if enabled
                if settings.use_query_expansion:
                    context_for_search = await query_expander.expand_query(context_for_search)
                
                # First pass: get initial candidates
                memory_results, _, _ = memory_engine.search_memories(
                    query=context_for_search,
                    top_k=settings.retrieval_top_k * 2,
                    use_hybrid=settings.use_hybrid_search
                )
                
                # Second pass: use memory context to find related memories
                # Use the extracted question for refinement
                question_for_refinement = _extract_question(search_message) if search_message else ""
                if settings.use_query_expansion and memory_results and question_for_refinement:
                    memory_summaries = [r.memory.content[:150] for r in memory_results[:5]]
                    refined_query = await query_expander.expand_with_memory_context(
                        question_for_refinement,
                        memory_summaries
                    )
                    
                    if refined_query != search_message:
                        refined_results, _, _ = memory_engine.search_memories(
                            query=refined_query,
                            top_k=settings.retrieval_top_k,
                            use_hybrid=settings.use_hybrid_search
                        )
                        
                        seen_ids = {r.memory.id for r in refined_results}
                        for r in memory_results:
                            if r.memory.id not in seen_ids and len(refined_results) < settings.retrieval_top_k:
                                refined_results.append(r)
                                seen_ids.add(r.memory.id)
                        
                        memory_results = refined_results[:settings.retrieval_top_k]
                
                retrieved_memories = memory_results[:settings.retrieval_top_k]
    
    # Build conversation history for LLM
    history = [{"role": m.role, "content": m.content} for m in conversation[-settings.conversation_window:]]
    
    # Convert memories to dict format for prompt building
    memory_dicts = [
        {
            "content": mr.memory.content,
            "emotional_tone": mr.memory.emotional_tone,
            "similarity": mr.similarity,
            "created_at": mr.memory.created_at.isoformat() if mr.memory.created_at else None
        }
        for mr in retrieved_memories
    ]
    
    # Build prompt with injected memories (RAG injection step)
    messages = build_prompt_with_memories(
        user_message=request.message,
        conversation_history=history,
        memories=memory_dicts
    )
    
    # Generate response
    try:
        response_text = await llm_client.chat_completion(messages)
    except Exception as e:
        logger.error(f"LLM error: {e}")
        raise HTTPException(status_code=503, detail=f"LLM service error: {str(e)}")
    
    # Store messages in SQLite (only save user message if not empty)
    if request.message:
        user_msg = Message(role=MessageRole.USER, content=request.message)
        conversation_store.add_message(conversation_id, user_msg)
    assistant_msg = Message(role=MessageRole.ASSISTANT, content=response_text)
    conversation_store.add_message(conversation_id, assistant_msg)
    
    return ChatResponse(
        response=response_text,
        conversation_id=conversation_id,
        retrieved_memories=retrieved_memories
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint.
    Returns Server-Sent Events for real-time response.
    
    Uses time-based context window:
    - Messages within last 24h are included directly in context
    - Older messages are retrieved via RAG from ChromaDB
    
    Memory retrieval uses:
    - Hybrid search (semantic + BM25 keyword matching)
    - Optional query expansion via LLM
    """
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    # Load ACTIVE messages (within context window) from SQLite
    conversation = conversation_store.get_active_messages(conversation_id)
    
    # Log the incoming message for debugging
    logger.info(f"Stream request - message: '{request.message[:100] if request.message else '(empty)'}...'")
    
    # Retrieve memories
    retrieved_memories = []
    if request.include_memories and memory_engine.memory_count > 0:
        # Use current message, fallback to last user message from conversation
        search_message = request.message
        if not search_message and conversation:
            # Find last user message in conversation that looks like a real user message
            # (not a corrupted AI response saved as user message)
            for msg in reversed(conversation):
                if msg.role == "user" and msg.content:
                    # Skip very long messages (likely corrupted AI responses)
                    # Real user messages are usually under 2000 chars
                    if len(msg.content) < 2000:
                        search_message = msg.content
                        logger.info(f"Using last user message for search: '{search_message[:50]}...'")
                        break
                    else:
                        logger.warning(f"Skipping suspiciously long 'user' message ({len(msg.content)} chars)")
            
            if not search_message:
                logger.warning("No suitable user message found for memory search")
        
        # For memory search, extract the actual QUESTION from the message
        # Long messages with preamble can overwhelm the search
        context_for_search = _extract_question(search_message) if search_message else ""
        
        if context_for_search:
            logger.debug(f"Search context: '{context_for_search[:100]}...'")
        
        # Skip search if we have no context at all
        if not context_for_search.strip():
            logger.warning("No search context available, skipping memory retrieval")
        else:
            # Check for temporal queries (e.g., "co robiliśmy wczoraj?")
            temporal_result = _parse_temporal_query(context_for_search)
            
            if temporal_result:
                # Use date-based search
                start_date, end_date, cleaned_query = temporal_result
                logger.info(f"Using temporal search: {start_date.date()} to {end_date.date()}")
                
                # If there's a semantic component, combine with date filter
                if cleaned_query and len(cleaned_query) > 5:
                    memory_results = memory_engine.search_memories_by_date(
                        start_date=start_date,
                        end_date=end_date,
                        limit=settings.retrieval_top_k * 2,
                        also_search_query=cleaned_query
                    )
                else:
                    # Pure date search
                    memory_results = memory_engine.search_memories_by_date(
                        start_date=start_date,
                        end_date=end_date,
                        limit=settings.retrieval_top_k * 2
                    )
                
                retrieved_memories = memory_results[:settings.retrieval_top_k]
            else:
                # Standard semantic search
                # Apply query expansion if enabled
                if settings.use_query_expansion:
                    context_for_search = await query_expander.expand_query(context_for_search)
                
                # First pass: get initial candidates
                memory_results, _, _ = memory_engine.search_memories(
                    query=context_for_search,
                    top_k=settings.retrieval_top_k * 2,
                    use_hybrid=settings.use_hybrid_search
                )
                
                # Second pass: use memory context to find related memories
                # Use the extracted question for refinement
                question_for_refinement = _extract_question(search_message) if search_message else ""
                if settings.use_query_expansion and memory_results and question_for_refinement:
                    memory_summaries = [r.memory.content[:150] for r in memory_results[:5]]
                    refined_query = await query_expander.expand_with_memory_context(
                        question_for_refinement,
                        memory_summaries
                    )
                    
                    if refined_query != search_message:
                        refined_results, _, _ = memory_engine.search_memories(
                            query=refined_query,
                            top_k=settings.retrieval_top_k,
                            use_hybrid=settings.use_hybrid_search
                        )
                        
                        seen_ids = {r.memory.id for r in refined_results}
                        for r in memory_results:
                            if r.memory.id not in seen_ids and len(refined_results) < settings.retrieval_top_k:
                                refined_results.append(r)
                                seen_ids.add(r.memory.id)
                        
                        memory_results = refined_results[:settings.retrieval_top_k]
                
                retrieved_memories = memory_results[:settings.retrieval_top_k]
    
    history = [{"role": m.role, "content": m.content} for m in conversation[-settings.conversation_window:]]
    
    memory_dicts = [
        {
            "content": mr.memory.content,
            "emotional_tone": mr.memory.emotional_tone,
            "similarity": mr.similarity,
            "created_at": mr.memory.created_at.isoformat() if mr.memory.created_at else None
        }
        for mr in retrieved_memories
    ]
    
    messages = build_prompt_with_memories(
        user_message=request.message,
        conversation_history=history,
        memories=memory_dicts
    )
    
    async def generate():
        full_response = ""
        
        # First, send metadata with full memory details
        memories_data = [
            {
                "id": mr.memory.id,
                "content": mr.memory.content,
                "emotional_tone": mr.memory.emotional_tone,
                "similarity": mr.similarity,
                "created_at": mr.memory.created_at.isoformat() if mr.memory.created_at else None
            }
            for mr in retrieved_memories
        ]
        yield f"data: {json.dumps({'type': 'meta', 'conversation_id': conversation_id, 'memories': memories_data})}\n\n"
        
        try:
            stream = await llm_client.chat_completion(messages, stream=True)
            async for chunk in stream:
                full_response += chunk
                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            return
        
        # Store in SQLite (only save user message if not empty)
        if request.message:
            user_msg = Message(role=MessageRole.USER, content=request.message)
            conversation_store.add_message(conversation_id, user_msg)
        assistant_msg = Message(role=MessageRole.ASSISTANT, content=full_response)
        conversation_store.add_message(conversation_id, assistant_msg)
        
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# === Memory Management ===

@app.post("/memories", response_model=Memory)
async def save_memory(request: SaveMemoryRequest):
    """Manually save a memory."""
    memory = Memory(
        content=request.content,
        summary=request.summary,
        emotional_tone=request.emotional_tone,
        importance=request.importance,
        tags=request.tags
    )
    
    memory_engine.save_memory(memory)
    return memory


@app.post("/memories/search", response_model=SearchMemoriesResponse)
async def search_memories(request: SearchMemoriesRequest):
    """Search for relevant memories."""
    results, embed_time, search_time = memory_engine.search_memories(
        query=request.query,
        top_k=request.top_k,
        min_similarity=request.min_similarity
    )
    
    return SearchMemoriesResponse(
        results=results,
        query_embedding_time_ms=embed_time,
        search_time_ms=search_time
    )


@app.get("/memories", response_model=list[Memory])
async def list_memories(limit: int = 100):
    """List all stored memories."""
    return memory_engine.get_all_memories(limit=limit)


@app.get("/memories/{memory_id}", response_model=Memory)
async def get_memory(memory_id: str):
    """Get a specific memory by ID."""
    memory = memory_engine.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory


@app.put("/memories/{memory_id}", response_model=Memory)
async def update_memory(memory_id: str, request: UpdateMemoryRequest):
    """Update a specific memory."""
    updated = memory_engine.update_memory(
        memory_id=memory_id,
        content=request.content,
        summary=request.summary,
        emotional_tone=request.emotional_tone,
        importance=request.importance,
        tags=request.tags
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Memory not found")
    return updated


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a specific memory."""
    success = memory_engine.delete_memory(memory_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "deleted", "memory_id": memory_id}


# === Conversation Management ===

@app.get("/conversations")
async def list_conversations(limit: int = 50, offset: int = 0):
    """List all conversations, newest first."""
    conversations = conversation_store.list_conversations(limit=limit, offset=offset)
    return {
        "conversations": conversations,
        "total": conversation_store.get_total_conversation_count()
    }


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history."""
    conv = conversation_store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    messages = conversation_store.get_messages(conversation_id)
    
    return {
        "conversation_id": conversation_id,
        "title": conv.get("title"),
        "created_at": conv.get("created_at"),
        "updated_at": conv.get("updated_at"),
        "messages": [m.model_dump() for m in messages]
    }


@app.put("/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, title: Optional[str] = None):
    """Update conversation metadata (e.g., title)."""
    if not conversation_store.conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation_store.update_conversation(conversation_id, title=title)
    return {"status": "updated", "conversation_id": conversation_id}


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and all its messages."""
    if not conversation_store.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"status": "deleted", "conversation_id": conversation_id}


# === Message Management ===

@app.put("/messages/{message_id}")
async def update_message(message_id: str, content: str):
    """Update the content of a specific message."""
    updated = conversation_store.update_message(message_id, content)
    if not updated:
        raise HTTPException(status_code=404, detail="Message not found")
    
    return {
        "status": "updated",
        "message": updated.model_dump()
    }


@app.delete("/messages/{message_id}")
async def delete_message(message_id: str):
    """Delete a specific message."""
    if not conversation_store.delete_message(message_id):
        raise HTTPException(status_code=404, detail="Message not found")
    
    return {"status": "deleted", "message_id": message_id}


@app.get("/conversations/{conversation_id}/context-window")
async def get_context_window_stats(conversation_id: str):
    """
    Get statistics about the context window for a conversation.
    
    Shows how many messages are "active" (within the time window)
    vs "expired" (should be archived to memories).
    """
    if not conversation_store.conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversation_store.get_context_window_stats(conversation_id)


@app.get("/conversations/expired-messages")
async def get_expired_messages(limit: int = 100):
    """
    Get all expired messages across all conversations.
    
    These are messages outside the context window that should be
    archived to ChromaDB as memories.
    """
    expired = conversation_store.get_unarchived_expired_messages(limit=limit)
    
    return {
        "count": len(expired),
        "window_hours": settings.context_window_hours,
        "cutoff_time": conversation_store.get_context_window_cutoff().isoformat(),
        "messages": [
            {
                "message": msg["message"].model_dump(),
                "conversation_id": msg["conversation_id"],
                "conversation_title": msg["conversation_title"]
            }
            for msg in expired
        ]
    }


@app.post("/conversations/{conversation_id}/extract-memories")
async def extract_memories_from_conversation(conversation_id: str):
    """
    Extract and save memories from a conversation.
    This uses the LLM to identify important moments.
    """
    if not conversation_store.conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation = conversation_store.get_messages(conversation_id)
    if len(conversation) < 4:
        return {"status": "skipped", "reason": "Conversation too short"}
    
    # Build context for memory extraction
    conv_text = "\n".join([
        f"{m.role.upper()}: {m.content}" for m in conversation[-20:]
    ])
    
    extraction_prompt = [
        {
            "role": "system",
            "content": """You are a memory curator. Your task is to identify important moments from a conversation that should be remembered for future reference.

Focus on:
- Emotional revelations or significant feelings expressed
- Personal information shared (preferences, experiences, relationships)
- Important decisions or turning points
- Meaningful exchanges that define the relationship

For each memory, provide:
1. A concise summary (1-2 sentences)
2. The emotional tone (e.g., "hopeful", "sad", "excited", "vulnerable")
3. Importance score (0.0-1.0)

Respond in JSON format:
{"memories": [{"content": "...", "emotional_tone": "...", "importance": 0.X}, ...]}

If there are no significant memories to extract, respond with: {"memories": []}"""
        },
        {
            "role": "user",
            "content": f"Extract important memories from this conversation:\n\n{conv_text}"
        }
    ]
    
    try:
        response = await llm_client.chat_completion(extraction_prompt)
        
        # Parse response
        import json
        # Try to extract JSON from response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(response[start:end])
            
            saved_memories = []
            for mem_data in data.get("memories", []):
                memory = Memory(
                    content=mem_data["content"],
                    emotional_tone=mem_data.get("emotional_tone"),
                    importance=mem_data.get("importance", 0.5),
                    source_messages=[m.id for m in conversation[-20:]]
                )
                memory_engine.save_memory(memory)
                saved_memories.append(memory)
            
            return {
                "status": "success",
                "memories_extracted": len(saved_memories),
                "memories": saved_memories
            }
    except Exception as e:
        logger.error(f"Memory extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Memory extraction failed: {str(e)}")
    
    return {"status": "no_memories", "reason": "No significant memories found"}


# === Archival ===

@app.get("/archival/stats")
async def get_archival_stats():
    """
    Get current archival statistics.
    
    Shows how many messages are pending archival and system status.
    """
    return archiver.get_archival_stats()


@app.post("/archival/run")
async def run_archival(
    mode: str = "llm",
    batch_size: int = 50,
    conversation_id: Optional[str] = None
):
    """
    Run archival process to convert expired messages to memories.
    
    Args:
        mode: "llm" (LLM-extracted memories, default) or "simple" (direct copy)
        batch_size: Max messages to process (default 50)
        conversation_id: Optional - only archive from this conversation
    
    Returns:
        Archival results with statistics
    """
    if mode not in ("simple", "llm"):
        raise HTTPException(status_code=400, detail="Mode must be 'simple' or 'llm'")
    
    if archiver.is_running:
        raise HTTPException(status_code=409, detail="Archival already in progress")
    
    result = await archiver.archive_expired_messages(
        mode=mode,
        batch_size=batch_size,
        conversation_id=conversation_id
    )
    
    return {
        "status": "completed",
        "messages_processed": result.messages_processed,
        "memories_created": result.memories_created,
        "messages_archived": len(result.messages_archived),
        "errors": result.errors,
        "memories": [m.model_dump() for m in result.memories]
    }


@app.post("/archival/run-all")
async def run_full_archival(mode: str = "llm"):
    """
    Run archival for ALL expired messages (may take a while).
    
    Processes in batches until no more expired messages remain.
    Uses LLM extraction by default to create meaningful summaries.
    """
    if mode not in ("simple", "llm"):
        raise HTTPException(status_code=400, detail="Mode must be 'simple' or 'llm'")
    
    if archiver.is_running:
        raise HTTPException(status_code=409, detail="Archival already in progress")
    
    total_processed = 0
    total_memories = 0
    all_errors = []
    batch_count = 0
    max_batches = 100  # Safety limit
    
    while batch_count < max_batches:
        result = await archiver.archive_expired_messages(
            mode=mode,
            batch_size=100
        )
        
        if result.messages_processed == 0:
            break
        
        total_processed += result.messages_processed
        total_memories += result.memories_created
        all_errors.extend(result.errors)
        batch_count += 1
        
        logger.info(f"Archival batch {batch_count}: {result.messages_processed} messages → {result.memories_created} memories")
    
    return {
        "status": "completed",
        "batches_run": batch_count,
        "total_messages_processed": total_processed,
        "total_memories_created": total_memories,
        "errors": all_errors
    }


# === Auto-Archival Control ===

@app.get("/archival/auto/status")
async def get_auto_archival_status():
    """
    Get current auto-archival status.
    
    Returns whether auto-archival is enabled, running, and pending message count.
    """
    return background_archiver.get_status()


@app.post("/archival/auto/enable")
async def enable_auto_archival():
    """
    Enable automatic background archival.
    
    When enabled, the system will check every minute for expired messages
    and archive them using LLM extraction if the threshold is exceeded.
    """
    background_archiver.enable()
    
    # Start the task if it's not running
    if not background_archiver.is_running:
        await background_archiver.start()
    
    return {
        "status": "enabled",
        **background_archiver.get_status()
    }


@app.post("/archival/auto/disable")
async def disable_auto_archival():
    """
    Disable automatic background archival.
    
    The background task will continue running but skip archival checks.
    This allows re-enabling without restarting the server.
    """
    background_archiver.disable()
    return {
        "status": "disabled",
        **background_archiver.get_status()
    }


@app.post("/archival/rebuild")
async def rebuild_all_memories(mode: str = "llm"):
    """
    Delete ALL memories and re-archive all messages from scratch.
    
    This is useful when the memory extraction prompts have been updated
    and you want to regenerate all memories with the new format.
    
    WARNING: This deletes all existing memories and can take a long time!
    
    Args:
        mode: "simple" (direct conversion) or "llm" (LLM-extracted memories, recommended)
    
    Returns:
        Statistics about the rebuild process
    """
    if mode not in ("simple", "llm"):
        raise HTTPException(status_code=400, detail="Mode must be 'simple' or 'llm'")
    
    if archiver.is_running:
        raise HTTPException(status_code=409, detail="Archival already in progress")
    
    logger.warning("=== STARTING FULL MEMORY REBUILD ===")
    
    # Step 1: Delete all existing memories
    deleted_count = memory_engine.delete_all_memories()
    logger.info(f"Step 1/3: Deleted {deleted_count} existing memories")
    
    # Step 2: Reset archived flags on all messages
    reset_count = conversation_store.reset_archived_flags()
    logger.info(f"Step 2/3: Reset archived flag on {reset_count} messages")
    
    # Step 3: Re-archive all expired messages
    total_processed = 0
    total_memories = 0
    all_errors = []
    batch_count = 0
    max_batches = 200  # Higher limit for full rebuild
    
    while batch_count < max_batches:
        result = await archiver.archive_expired_messages(
            mode=mode,
            batch_size=50  # Smaller batches for LLM mode
        )
        
        if result.messages_processed == 0:
            break
        
        total_processed += result.messages_processed
        total_memories += result.memories_created
        all_errors.extend(result.errors)
        batch_count += 1
        
        logger.info(f"Step 3/3: Batch {batch_count} - {result.messages_processed} messages → {result.memories_created} memories")
    
    logger.warning(f"=== MEMORY REBUILD COMPLETE: {total_memories} memories created from {total_processed} messages ===")
    
    return {
        "status": "completed",
        "memories_deleted": deleted_count,
        "messages_reset": reset_count,
        "batches_run": batch_count,
        "total_messages_processed": total_processed,
        "total_memories_created": total_memories,
        "errors": all_errors
    }


# === Run Server ===

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
