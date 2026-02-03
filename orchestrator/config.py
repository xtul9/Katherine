"""
Katherine Orchestrator - Configuration
"""
import os
import base64
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path


def _generate_or_load_encryption_key() -> str:
    """
    Generate or load encryption key for internal monologue.
    
    The key is stored in .env file. If not present, generates a new one.
    This ensures the same key is used across restarts.
    """
    env_path = Path(__file__).parent / ".env"
    key_name = "KATHERINE_MONOLOGUE_ENCRYPTION_KEY"
    
    # Check if key exists in environment
    existing_key = os.environ.get(key_name)
    if existing_key:
        return existing_key
    
    # Try to read from .env file
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith(f"{key_name}="):
                    return line.split('=', 1)[1].strip().strip('"\'')
    
    # Generate new key (32 bytes for AES-256, base64 encoded)
    new_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
    
    # Append to .env file
    with open(env_path, 'a') as f:
        f.write(f"\n# Auto-generated encryption key for internal monologue\n")
        f.write(f"{key_name}={new_key}\n")
    
    return new_key


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Pydantic v2 configuration for loading from .env
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="KATHERINE_",
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore extra fields in .env
    )
    
    # Server
    host: str = "0.0.0.0"
    port: int = 10000
    debug: bool = False
    
    # OpenRouter API
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str = ""  # Required - set in .env
    openrouter_model: str = "deepseek/deepseek-chat-v3-0324"
    
    # Site info for OpenRouter (optional but recommended)
    openrouter_site_url: str = "http://localhost:10001"
    openrouter_site_name: str = "Katherine"
    
    # Embedding configuration
    # Mode: "api" (uses OpenAI API) or "local" (sentence-transformers, needs Python 3.10-3.12)
    embedding_mode: str = "api"
    embedding_model: str = "all-MiniLM-L6-v2"  # For local mode
    
    # OpenAI embedding settings (for API mode)
    # Note: OpenRouter does NOT support embeddings, so we use OpenAI directly
    openai_embedding_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""  # Set KATHERINE_OPENAI_API_KEY in .env (required for API embeddings)
    openai_embedding_model: str = "text-embedding-3-small"
    # Timeout for embedding API calls (seconds)
    embedding_timeout: float = 120.0
    # Number of retries for embedding API calls on timeout
    embedding_retries: int = 3
    
    # ChromaDB
    chroma_persist_directory: str = "./data/chroma"
    chroma_collection_name: str = "katherine_memories"
    
    # RAG settings
    retrieval_top_k: int = 10
    min_similarity_threshold: float = 0.3
    
    # Hybrid search settings
    # Enable hybrid search (semantic + BM25 keyword matching)
    use_hybrid_search: bool = True
    
    # Query expansion settings
    # Enable LLM-based query expansion for better retrieval
    use_query_expansion: bool = True
    # Model to use for query expansion (smaller/cheaper model recommended)
    query_expansion_model: str = "openai/gpt-4o-mini"
    
    # Deduplication settings
    # Write-time: block true duplicates (very high similarity)
    dedup_write_threshold: float = 0.95
    # Read-time: cluster similar memories, return only representatives
    dedup_read_threshold: float = 0.85
    
    # Memory processing
    conversation_window: int = 10  # Messages to consider for context
    auto_save_interval: int = 5   # Auto-save every N messages
    
    # Time-based context window (in hours)
    # Messages older than this are "expired" and should be archived to memories
    context_window_hours: int = 24
    # Minimum messages to always include in context (even if older than window)
    # Ensures continuity when user returns after a long break
    min_context_messages: int = 10
    
    # Auto-archival settings
    # Enable automatic background archival of expired messages
    auto_archival_enabled: bool = True
    # Check interval in seconds (how often to check for expired messages)
    auto_archival_interval_seconds: int = 60
    # Minimum unarchived messages before triggering archival
    auto_archival_threshold: int = 10
    # Batch size for auto-archival (to avoid overwhelming the LLM)
    auto_archival_batch_size: int = 20
    
    # Token limits (approximate)
    max_context_tokens: int = 4096
    max_memory_injection_tokens: int = 1024
    
    # Persona configuration (for privacy - keep personal details out of code)
    user_name: str = "User"           # Name for the human user
    ai_name: str = "Katherine"        # Name for the AI character
    persona_file: str = "persona.txt" # Path to custom system prompt file
    
    # Internal monologue settings
    # Encryption key for internal monologue (auto-generated if not set)
    monologue_encryption_key: str = Field(default_factory=_generate_or_load_encryption_key)
    # Separator between public response and internal monologue
    # Using XML-style tags as LLMs are better trained to recognize these
    monologue_separator: str = "<internal_monologue>"
    monologue_separator_closing_tag: str = "</internal_monologue>"
    # Max tokens for LLM output (increased to accommodate internal monologue)
    llm_max_tokens: int = 6500


settings = Settings()

# Ensure data directories exist
Path(settings.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
