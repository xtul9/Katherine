"""
Katherine Orchestrator - Data Models
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class MessageRole(str, Enum):
    """Role of the message sender."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """A single message in the conversation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Internal monologue (only for assistant messages)
    # This is Katherine's private reflection on why she said what she said
    internal_monologue: Optional[str] = None
    
    # IDs of memories that were retrieved and influenced this response
    retrieved_memory_ids: list[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True


class Memory(BaseModel):
    """A stored memory extracted from conversation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    summary: Optional[str] = None
    emotional_tone: Optional[str] = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    source_messages: list[str] = Field(default_factory=list)  # Message IDs
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[str] = Field(default_factory=list)
    
    # Internal monologue from the assistant's response
    # Preserved from the original message for memory context
    internal_monologue: Optional[str] = None
    
    # IDs of memories that influenced this exchange
    # Provides causal chain: "I said X because I remembered Y and Z"
    influencing_memory_ids: list[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemorySearchResult(BaseModel):
    """Result from memory search."""
    memory: Memory
    similarity: float
    relevance_explanation: Optional[str] = None


# === API Request/Response Models ===

class ChatRequest(BaseModel):
    """Request to send a chat message."""
    message: str
    conversation_id: Optional[str] = None
    include_memories: bool = True


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    response: str
    conversation_id: str
    retrieved_memories: list[MemorySearchResult] = Field(default_factory=list)
    tokens_used: Optional[int] = None


class SaveMemoryRequest(BaseModel):
    """Request to manually save a memory."""
    content: str
    summary: Optional[str] = None
    emotional_tone: Optional[str] = None
    importance: float = 0.5
    tags: list[str] = Field(default_factory=list)


class SearchMemoriesRequest(BaseModel):
    """Request to search memories."""
    query: str
    top_k: int = 5
    min_similarity: float = 0.3


class UpdateMemoryRequest(BaseModel):
    """Request to update an existing memory."""
    content: Optional[str] = None
    summary: Optional[str] = None
    emotional_tone: Optional[str] = None
    importance: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    tags: Optional[list[str]] = None


class SearchMemoriesResponse(BaseModel):
    """Response from memory search."""
    results: list[MemorySearchResult]
    query_embedding_time_ms: float
    search_time_ms: float


class ConversationHistoryRequest(BaseModel):
    """Request to process conversation history."""
    messages: list[Message]
    extract_memories: bool = True


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    embedding_model_loaded: bool
    chroma_connected: bool
    vllm_reachable: bool
    memory_count: int
    conversation_count: int
    message_count: int


# === User Tags Models ===

class UserTag(BaseModel):
    """A tag describing the user, as perceived by the AI."""
    tag: str
    display_order: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TagMove(BaseModel):
    """Instruction to move a tag to a new position."""
    tag: str
    position: str  # "BEFORE tag_name", "AFTER tag_name", "TO_TOP", "TO_BOTTOM"


class TagChanges(BaseModel):
    """Changes to user tags extracted from inner monologue."""
    add: list[str] = Field(default_factory=list)  # Tags to add
    remove: list[str] = Field(default_factory=list)  # Tags to remove
    move: list[TagMove] = Field(default_factory=list)  # Tags to move