"""
Katherine Orchestrator - Message Archiver
Automatically archives expired messages to ChromaDB as memories.

Phase 3: Conversation messages outside the context window are converted
into memories, allowing Katherine to recall them through RAG while
keeping the active context window lean.
"""
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

from loguru import logger

from config import settings
from models import Memory, Message
from memory_engine import memory_engine
from conversation_store import conversation_store
from llm_client import llm_client


@dataclass
class ArchiveResult:
    """Result of an archival operation."""
    messages_processed: int
    memories_created: int
    messages_archived: list[str]  # Message IDs
    memories: list[Memory]
    errors: list[str]


class MessageArchiver:
    """
    Archives expired conversation messages to ChromaDB.
    
    Two archival modes:
    1. Simple: Each message pair (user + assistant) becomes one memory
    2. LLM-extracted: Use LLM to identify important moments (more expensive but smarter)
    """
    
    def __init__(self):
        self._is_running = False
    
    async def archive_expired_messages(
        self,
        mode: str = "simple",
        batch_size: int = 50,
        conversation_id: Optional[str] = None
    ) -> ArchiveResult:
        """
        Archive expired messages to ChromaDB.
        
        Args:
            mode: "simple" (direct conversion) or "llm" (LLM-extracted memories)
            batch_size: Max messages to process in one run
            conversation_id: If provided, only archive from this conversation
        
        Returns:
            ArchiveResult with statistics
        """
        if self._is_running:
            return ArchiveResult(
                messages_processed=0,
                memories_created=0,
                messages_archived=[],
                memories=[],
                errors=["Archival already in progress"]
            )
        
        self._is_running = True
        result = ArchiveResult(
            messages_processed=0,
            memories_created=0,
            messages_archived=[],
            memories=[],
            errors=[]
        )
        
        try:
            # Get expired messages
            if conversation_id:
                expired = self._get_expired_for_conversation(conversation_id, batch_size)
            else:
                expired = conversation_store.get_unarchived_expired_messages(limit=batch_size)
            
            if not expired:
                logger.info("No expired messages to archive")
                return result
            
            logger.info(f"Archiving {len(expired)} expired messages (mode: {mode})")
            
            if mode == "llm":
                result = await self._archive_with_llm(expired)
            else:
                result = await self._archive_simple(expired)
            
            # Mark messages as archived
            if result.messages_archived:
                logger.info(f"Attempting to mark {len(result.messages_archived)} messages as archived")
                logger.debug(f"Message IDs to archive: {result.messages_archived}")
                archived_count = conversation_store.mark_messages_as_archived(result.messages_archived)
                logger.info(f"Marked {archived_count} messages as archived in database")
            else:
                logger.warning("No messages in result.messages_archived list!")
            
        except Exception as e:
            logger.error(f"Archival failed: {e}")
            result.errors.append(str(e))
        finally:
            self._is_running = False
        
        return result
    
    def _get_expired_for_conversation(
        self, 
        conversation_id: str, 
        limit: int
    ) -> list[dict]:
        """Get expired messages for a specific conversation."""
        all_expired = conversation_store.get_unarchived_expired_messages(limit=limit * 2)
        return [e for e in all_expired if e["conversation_id"] == conversation_id][:limit]
    
    async def _archive_simple(self, expired: list[dict]) -> ArchiveResult:
        """
        Simple archival: group messages into pairs and create memories.
        
        Groups consecutive user+assistant messages into single memories.
        """
        result = ArchiveResult(
            messages_processed=0,
            memories_created=0,
            messages_archived=[],
            memories=[],
            errors=[]
        )
        
        # Group messages by conversation
        by_conversation: dict[str, list[dict]] = {}
        for item in expired:
            conv_id = item["conversation_id"]
            if conv_id not in by_conversation:
                by_conversation[conv_id] = []
            by_conversation[conv_id].append(item)
        
        # Process each conversation
        for conv_id, messages in by_conversation.items():
            # Sort by timestamp
            messages.sort(key=lambda x: x["message"].timestamp)
            
            # Group into pairs (user + assistant)
            i = 0
            while i < len(messages):
                msg = messages[i]["message"]
                conv_title = messages[i].get("conversation_title", "")
                
                # Try to pair user message with following assistant message
                if msg.role == "user" and i + 1 < len(messages):
                    next_msg = messages[i + 1]["message"]
                    if next_msg.role == "assistant":
                        # Create memory from pair
                        memory = self._create_memory_from_pair(
                            user_msg=msg,
                            assistant_msg=next_msg,
                            conversation_title=conv_title
                        )
                        
                        saved_id = memory_engine.save_memory(memory)
                        if saved_id:
                            result.memories_created += 1
                            result.memories.append(memory)
                        
                        result.messages_processed += 2
                        result.messages_archived.extend([msg.id, next_msg.id])
                        i += 2
                        continue
                
                # Single message (no pair) - still archive it
                memory = self._create_memory_from_single(
                    msg=msg,
                    conversation_title=conv_title
                )
                
                saved_id = memory_engine.save_memory(memory)
                if saved_id:
                    result.memories_created += 1
                    result.memories.append(memory)
                
                result.messages_processed += 1
                result.messages_archived.append(msg.id)
                i += 1
        
        return result
    
    def _create_memory_from_pair(
        self,
        user_msg: Message,
        assistant_msg: Message,
        conversation_title: str
    ) -> Memory:
        """Create a memory from a user+assistant message pair."""
        # Format the exchange
        content = f"[Conversation: {conversation_title or 'untitled'}]\n"
        content += f"{settings.user_name}: {user_msg.content}\n"
        content += f"{settings.ai_name}: {assistant_msg.content}"
        
        # Truncate if too long
        if len(content) > 2000:
            content = content[:1997] + "..."
        
        return Memory(
            content=content,
            summary=f"Exchange from {user_msg.timestamp.strftime('%Y-%m-%d %H:%M')}",
            emotional_tone=None,  # Could be detected by LLM
            importance=0.5,
            source_messages=[user_msg.id, assistant_msg.id],
            created_at=user_msg.timestamp,
            tags=["archived", "conversation"]
        )
    
    def _create_memory_from_single(
        self,
        msg: Message,
        conversation_title: str
    ) -> Memory:
        """Create a memory from a single message."""
        role_name = settings.user_name if msg.role == "user" else settings.ai_name
        
        content = f"[Conversation: {conversation_title or 'untitled'}]\n"
        content += f"{role_name}: {msg.content}"
        
        if len(content) > 2000:
            content = content[:1997] + "..."
        
        return Memory(
            content=content,
            summary=f"Message from {role_name} at {msg.timestamp.strftime('%Y-%m-%d %H:%M')}",
            emotional_tone=None,
            importance=0.4,
            source_messages=[msg.id],
            created_at=msg.timestamp,
            tags=["archived", "single_message"]
        )
    
    async def _archive_with_llm(self, expired: list[dict]) -> ArchiveResult:
        """
        LLM-based archival: use LLM to extract meaningful memories.
        
        More expensive but produces higher quality memories.
        """
        result = ArchiveResult(
            messages_processed=0,
            memories_created=0,
            messages_archived=[],
            memories=[],
            errors=[]
        )
        
        # Group messages by conversation
        by_conversation: dict[str, list[dict]] = {}
        for item in expired:
            conv_id = item["conversation_id"]
            if conv_id not in by_conversation:
                by_conversation[conv_id] = []
            by_conversation[conv_id].append(item)
        
        # Process each conversation
        for conv_id, messages in by_conversation.items():
            messages.sort(key=lambda x: x["message"].timestamp)
            
            # Build conversation text
            conv_text = "\n".join([
                f"{(settings.user_name if m['message'].role == 'user' else settings.ai_name)}: {m['message'].content}"
                for m in messages
            ])
            
            conv_title = messages[0].get("conversation_title", "untitled")
            
            # Ask LLM to extract memories
            extraction_prompt = [
                {
                    "role": "system",
                    "content": f"""You are a memory curator. Your task is to identify important moments from the conversation that should be remembered.

Focus on:
- Emotional confessions or significant feelings
- Personal information (preferences, experiences, relationships)
- Important decisions or turning points
- Significant exchanges that define the relationship

CRITICAL: Each memory MUST include key quotes from the conversation. The AI hallucinates exact words when memories don't contain original quotes.

For each memory provide:
1. Summary: Brief context (1 sentence describing the situation)
2. Key quotes: 1-3 verbatim quotes from the conversation (MANDATORY - copy exact words from {settings.user_name} and/or {settings.ai_name})
3. Emotional tone (e.g. "hopeful", "sad", "excited", "vulnerable", "intimate", "playful")
4. Importance (0.0-1.0)

Respond in JSON format:
{{"memories": [{{"summary": "...", "key_quotes": [{{"speaker": "{settings.user_name}/{settings.ai_name}", "quote": "exact words"}}], "emotional_tone": "...", "importance": 0.X}}, ...]}}

Example:
{{"memories": [{{"summary": "{settings.user_name} opened up about his childhood fears.", "key_quotes": [{{"speaker": "{settings.user_name}", "quote": "I was always afraid of being alone. Even now, sometimes I wake up and check if someone's there."}}, {{"speaker": "{settings.ai_name}", "quote": "That vulnerability takes courage to share. I'm here with you."}}], "emotional_tone": "vulnerable", "importance": 0.8}}]}}

If there are no significant memories to extract, respond with: {{"memories": []}}"""
                },
                {
                    "role": "user",
                    "content": f"Extract important memories from this conversation (title: {conv_title}):\n\n{conv_text}"
                }
            ]
            
            try:
                import json
                response = await llm_client.chat_completion(extraction_prompt)
                
                message_ids = [m["message"].id for m in messages]
                
                # Parse JSON from response
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    data = json.loads(response[start:end])
                    
                    for mem_data in data.get("memories", []):
                        # Build content with summary and key quotes
                        summary = mem_data.get("summary", mem_data.get("content", ""))
                        key_quotes = mem_data.get("key_quotes", [])
                        
                        content_parts = [summary]
                        if key_quotes:
                            content_parts.append("\nKey quotes:")
                            for quote in key_quotes:
                                speaker = quote.get("speaker", "Unknown")
                                text = quote.get("quote", "")
                                if text:
                                    content_parts.append(f'  {speaker}: "{text}"')
                        
                        content = "\n".join(content_parts)
                        
                        memory = Memory(
                            content=content,
                            summary=f"Extracted from conversation: {conv_title}",
                            emotional_tone=mem_data.get("emotional_tone"),
                            importance=mem_data.get("importance", 0.5),
                            source_messages=message_ids,
                            created_at=messages[0]["message"].timestamp,
                            tags=["archived", "llm_extracted"]
                        )
                        
                        saved_id = memory_engine.save_memory(memory)
                        if saved_id:
                            result.memories_created += 1
                            result.memories.append(memory)
                else:
                    # LLM didn't return JSON - no memories to extract but still mark as processed
                    logger.warning(f"LLM returned non-JSON response for conversation {conv_id}: {response[:100]}...")
                
                # Always mark messages as archived after LLM processing (even if no memories extracted)
                result.messages_processed += len(messages)
                result.messages_archived.extend(message_ids)
                    
            except Exception as e:
                logger.error(f"LLM extraction failed for conversation {conv_id}: {e}")
                result.errors.append(f"Conversation {conv_id}: {str(e)}")
                
                # Fall back to simple archival for this conversation
                for item in messages:
                    msg = item["message"]
                    memory = self._create_memory_from_single(msg, conv_title)
                    saved_id = memory_engine.save_memory(memory)
                    if saved_id:
                        result.memories_created += 1
                        result.memories.append(memory)
                    result.messages_processed += 1
                    result.messages_archived.append(msg.id)
        
        return result
    
    @property
    def is_running(self) -> bool:
        """Check if archival is currently in progress."""
        return self._is_running
    
    def get_archival_stats(self) -> dict:
        """Get current archival statistics."""
        expired = conversation_store.get_unarchived_expired_messages(limit=1000)
        cutoff = conversation_store.get_context_window_cutoff()
        
        return {
            "pending_messages": len(expired),
            "window_hours": settings.context_window_hours,
            "cutoff_time": cutoff.isoformat(),
            "is_running": self._is_running,
            "memory_count": memory_engine.memory_count
        }


# Singleton instance
archiver = MessageArchiver()
