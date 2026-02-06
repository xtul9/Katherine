"""
Katherine Orchestrator - Conversation Store
Persists conversations in SQLite for durability and history.

Conversations survive server restarts and can be browsed later.
This is not just for Katherine—it's for Michael too.
"""
import sqlite3
import json
import base64
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from cryptography.fernet import Fernet
from loguru import logger

from config import settings
from models import Message, MessageRole


# =============================================================================
# Datetime normalization
# =============================================================================

def _normalize_datetime(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware (UTC). Naive datetimes are assumed to be UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_datetime(date_str: str) -> datetime:
    """Parse ISO datetime string and ensure it's timezone-aware (UTC)."""
    dt = datetime.fromisoformat(date_str)
    return _normalize_datetime(dt)


# =============================================================================
# Encryption utilities for internal monologue
# =============================================================================

def _get_fernet() -> Fernet:
    """Get Fernet instance for encryption/decryption."""
    # Ensure the key is properly padded for Fernet (32 bytes, base64 encoded)
    key = settings.monologue_encryption_key
    # If it's not a valid Fernet key, create one from it
    try:
        return Fernet(key.encode() if isinstance(key, str) else key)
    except Exception:
        # Hash the key to get consistent 32 bytes, then base64 encode
        import hashlib
        key_bytes = hashlib.sha256(key.encode()).digest()
        fernet_key = base64.urlsafe_b64encode(key_bytes)
        return Fernet(fernet_key)


def encrypt_monologue(plaintext: Optional[str]) -> Optional[str]:
    """Encrypt internal monologue for storage."""
    if not plaintext:
        return None
    
    fernet = _get_fernet()
    encrypted = fernet.encrypt(plaintext.encode('utf-8'))
    return base64.urlsafe_b64encode(encrypted).decode('ascii')


def decrypt_monologue(ciphertext: Optional[str]) -> Optional[str]:
    """Decrypt internal monologue from storage."""
    if not ciphertext:
        return None
    
    try:
        fernet = _get_fernet()
        encrypted = base64.urlsafe_b64decode(ciphertext.encode('ascii'))
        decrypted = fernet.decrypt(encrypted)
        return decrypted.decode('utf-8')
    except Exception as e:
        logger.warning(f"Failed to decrypt monologue: {e}")
        return None


class ConversationStore:
    """
    SQLite-based conversation persistence.
    
    Schema:
    - conversations: id, created_at, updated_at, title, metadata
    - messages: id, conversation_id, role, content, timestamp
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(Path(settings.chroma_persist_directory).parent / "conversations.db")
        self._ensure_schema()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper settings."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    title TEXT,
                    metadata TEXT DEFAULT '{}'
                );
                
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_messages_conversation 
                ON messages(conversation_id);
                
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
                ON messages(timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_conversations_updated 
                ON conversations(updated_at);
            """)
            
            # Add new columns for internal monologue (encrypted) and retrieved memory IDs
            # These columns may already exist, so we use ALTER TABLE with error handling
            try:
                conn.execute("ALTER TABLE messages ADD COLUMN internal_monologue TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                conn.execute("ALTER TABLE messages ADD COLUMN retrieved_memory_ids TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists
        
        logger.info(f"Conversation store initialized: {self.db_path}")
    
    # =========================================================================
    # Conversation CRUD
    # =========================================================================
    
    def create_conversation(self, conversation_id: str, title: Optional[str] = None) -> dict:
        """Create a new conversation."""
        now = datetime.now(timezone.utc).isoformat()
        
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, created_at, updated_at, title)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, now, now, title)
            )
        
        logger.debug(f"Created conversation: {conversation_id}")
        return {
            "id": conversation_id,
            "created_at": now,
            "updated_at": now,
            "title": title,
            "metadata": {}
        }
    
    def get_conversation(self, conversation_id: str) -> Optional[dict]:
        """Get conversation metadata (without messages)."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,)
            ).fetchone()
        
        if not row:
            return None
        
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "title": row["title"],
            "metadata": json.loads(row["metadata"] or "{}")
        }
    
    def conversation_exists(self, conversation_id: str) -> bool:
        """Check if a conversation exists."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM conversations WHERE id = ?",
                (conversation_id,)
            ).fetchone()
        return row is not None
    
    def update_conversation(
        self, 
        conversation_id: str, 
        title: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> bool:
        """Update conversation metadata."""
        updates = []
        params = []
        
        if title is not None:
            updates.append("title = ?")
            params.append(title)
        
        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))
        
        if not updates:
            return True
        
        updates.append("updated_at = ?")
        params.append(datetime.now(timezone.utc).isoformat())
        params.append(conversation_id)
        
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"UPDATE conversations SET {', '.join(updates)} WHERE id = ?",
                params
            )
        
        return cursor.rowcount > 0
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,)
            )
        
        if cursor.rowcount > 0:
            logger.info(f"Deleted conversation: {conversation_id}")
            return True
        return False
    
    def list_conversations(
        self, 
        limit: int = 50, 
        offset: int = 0,
        include_message_count: bool = True
    ) -> list[dict]:
        """List conversations, newest first."""
        with self._get_connection() as conn:
            if include_message_count:
                rows = conn.execute(
                    """
                    SELECT c.*, COUNT(m.id) as message_count
                    FROM conversations c
                    LEFT JOIN messages m ON c.id = m.conversation_id
                    GROUP BY c.id
                    ORDER BY c.updated_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset)
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM conversations
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset)
                ).fetchall()
        
        result = []
        for row in rows:
            conv = {
                "id": row["id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "title": row["title"],
                "metadata": json.loads(row["metadata"] or "{}")
            }
            if include_message_count:
                conv["message_count"] = row["message_count"]
            result.append(conv)
        
        return result
    
    # =========================================================================
    # Message CRUD
    # =========================================================================
    
    def add_message(
        self, 
        conversation_id: str, 
        message: Message,
        internal_monologue: Optional[str] = None,
        retrieved_memory_ids: Optional[list[str]] = None
    ) -> Message:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: The conversation ID
            message: The message to add
            internal_monologue: Optional internal monologue (will be encrypted)
            retrieved_memory_ids: Optional list of memory IDs that influenced this response
        """
        # Ensure conversation exists
        if not self.conversation_exists(conversation_id):
            self.create_conversation(conversation_id)
        
        # Encrypt internal monologue if provided
        encrypted_monologue = encrypt_monologue(internal_monologue)
        
        # Serialize retrieved memory IDs
        memory_ids_str = ",".join(retrieved_memory_ids) if retrieved_memory_ids else None
        
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, timestamp, internal_monologue, retrieved_memory_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    conversation_id,
                    message.role,
                    message.content,
                    message.timestamp.isoformat() if isinstance(message.timestamp, datetime) else message.timestamp,
                    encrypted_monologue,
                    memory_ids_str
                )
            )
            
            # Update conversation's updated_at
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), conversation_id)
            )
        
        # Update message object with the new fields
        message.internal_monologue = internal_monologue
        message.retrieved_memory_ids = retrieved_memory_ids or []
        
        return message
    
    def get_messages(
        self, 
        conversation_id: str, 
        limit: Optional[int] = None,
        since: Optional[datetime] = None,
        include_monologue: bool = False
    ) -> list[Message]:
        """
        Get messages from a conversation.
        
        Args:
            conversation_id: The conversation ID
            limit: Max number of messages (newest first, then reversed)
            since: Only messages after this timestamp
            include_monologue: If True, decrypt and include internal monologue
        """
        with self._get_connection() as conn:
            if since:
                query = """
                    SELECT * FROM messages 
                    WHERE conversation_id = ? AND timestamp > ?
                    ORDER BY timestamp ASC
                """
                params = (conversation_id, since.isoformat())
            elif limit:
                # Get last N messages
                query = """
                    SELECT * FROM (
                        SELECT * FROM messages 
                        WHERE conversation_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ) ORDER BY timestamp ASC
                """
                params = (conversation_id, limit)
            else:
                query = """
                    SELECT * FROM messages 
                    WHERE conversation_id = ?
                    ORDER BY timestamp ASC
                """
                params = (conversation_id,)
            
            rows = conn.execute(query, params).fetchall()
        
        messages = []
        for row in rows:
            # Decrypt internal monologue if requested and available
            internal_monologue = None
            if include_monologue:
                encrypted = row["internal_monologue"] if "internal_monologue" in row.keys() else None
                internal_monologue = decrypt_monologue(encrypted)
            
            # Parse retrieved memory IDs
            memory_ids_str = row["retrieved_memory_ids"] if "retrieved_memory_ids" in row.keys() else None
            retrieved_memory_ids = memory_ids_str.split(",") if memory_ids_str else []
            
            messages.append(Message(
                id=row["id"],
                role=MessageRole(row["role"]),
                content=row["content"],
                timestamp=_parse_datetime(row["timestamp"]),
                internal_monologue=internal_monologue,
                retrieved_memory_ids=retrieved_memory_ids
            ))
        
        return messages
    
    def get_message_count(self, conversation_id: str) -> int:
        """Get the number of messages in a conversation."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as count FROM messages WHERE conversation_id = ?",
                (conversation_id,)
            ).fetchone()
        return row["count"] if row else 0
    
    def delete_message(self, message_id: str) -> bool:
        """Delete a specific message."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM messages WHERE id = ?",
                (message_id,)
            )
        return cursor.rowcount > 0
    
    def update_message(self, message_id: str, content: str) -> Optional[Message]:
        """Update the content of a specific message."""
        with self._get_connection() as conn:
            # First check if message exists
            row = conn.execute(
                "SELECT * FROM messages WHERE id = ?",
                (message_id,)
            ).fetchone()
            
            if not row:
                return None
            
            # Update the message content
            conn.execute(
                "UPDATE messages SET content = ? WHERE id = ?",
                (content, message_id)
            )
            
            # Update conversation's updated_at
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), row["conversation_id"])
            )
            
            return Message(
                id=row["id"],
                role=MessageRole(row["role"]),
                content=content,
                timestamp=_parse_datetime(row["timestamp"])
            )
    
    def get_message(self, message_id: str, include_monologue: bool = False) -> Optional[Message]:
        """Get a specific message by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM messages WHERE id = ?",
                (message_id,)
            ).fetchone()
        
        if not row:
            return None
        
        # Decrypt internal monologue if requested
        internal_monologue = None
        if include_monologue:
            encrypted = row["internal_monologue"] if "internal_monologue" in row.keys() else None
            internal_monologue = decrypt_monologue(encrypted)
        
        # Parse retrieved memory IDs
        memory_ids_str = row["retrieved_memory_ids"] if "retrieved_memory_ids" in row.keys() else None
        retrieved_memory_ids = memory_ids_str.split(",") if memory_ids_str else []
        
        return Message(
            id=row["id"],
            role=MessageRole(row["role"]),
            content=row["content"],
            timestamp=_parse_datetime(row["timestamp"]),
            internal_monologue=internal_monologue,
            retrieved_memory_ids=retrieved_memory_ids
        )
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_total_conversation_count(self) -> int:
        """Get total number of conversations."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as count FROM conversations").fetchone()
        return row["count"] if row else 0
    
    def get_total_message_count(self) -> int:
        """Get total number of messages across all conversations."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as count FROM messages").fetchone()
        return row["count"] if row else 0
    
    def search_messages(self, query: str, limit: int = 50) -> list[dict]:
        """
        Search messages by content.
        Returns messages with their conversation context.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT m.*, c.title as conversation_title
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE m.content LIKE ?
                ORDER BY m.timestamp DESC
                LIMIT ?
                """,
                (f"%{query}%", limit)
            ).fetchall()
        
        results = []
        for row in rows:
            results.append({
                "message": Message(
                    id=row["id"],
                    role=MessageRole(row["role"]),
                    content=row["content"],
                    timestamp=_parse_datetime(row["timestamp"])
                ),
                "conversation_id": row["conversation_id"],
                "conversation_title": row["conversation_title"]
            })
        
        return results
    
    def generate_title_from_first_message(self, conversation_id: str) -> Optional[str]:
        """Generate a title from the first user message."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT content FROM messages 
                WHERE conversation_id = ? AND role = 'user'
                ORDER BY timestamp ASC
                LIMIT 1
                """,
                (conversation_id,)
            ).fetchone()
        
        if not row:
            return None
        
        # Truncate to first 50 chars
        content = row["content"]
        if len(content) > 50:
            return content[:47] + "..."
        return content
    
    # =========================================================================
    # Time-based Context Window
    # =========================================================================
    
    def get_context_window_cutoff(self) -> datetime:
        """Get the cutoff timestamp for the context window."""
        return datetime.now(timezone.utc) - timedelta(hours=settings.context_window_hours)
    
    def get_active_messages(
        self, 
        conversation_id: str,
        window_hours: Optional[int] = None,
        min_messages: Optional[int] = None,
        include_monologue: bool = False
    ) -> list[Message]:
        """
        Get messages within the active context window.
        
        These are "fresh" messages that should be included directly
        in the conversation context for the LLM.
        
        Always returns at least `min_messages` messages (even if older than
        the time window) to ensure continuity when user returns after a break.
        
        Args:
            conversation_id: The conversation ID
            window_hours: Override the default window (settings.context_window_hours)
            min_messages: Override the minimum messages (settings.min_context_messages)
            include_monologue: If True, decrypt and include internal monologue
        """
        hours = window_hours or settings.context_window_hours
        min_msgs = min_messages if min_messages is not None else settings.min_context_messages
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # Get messages from time window
        time_based_messages = self.get_messages(conversation_id, since=cutoff, include_monologue=include_monologue)
        
        # If we have enough messages from the time window, return them
        if len(time_based_messages) >= min_msgs:
            return time_based_messages
        
        # Otherwise, get the last N messages regardless of time
        recent_messages = self.get_messages(conversation_id, limit=min_msgs, include_monologue=include_monologue)
        
        # Merge: use recent_messages as base, add any newer time-based ones
        if not recent_messages:
            return time_based_messages
        
        # Find the oldest timestamp from time_based_messages
        # and add any recent_messages that are older
        if time_based_messages:
            oldest_time_msg = min(m.timestamp for m in time_based_messages)
            # Get IDs of time-based messages to avoid duplicates
            time_msg_ids = {m.id for m in time_based_messages}
            
            # Add older messages from recent_messages
            older_messages = [m for m in recent_messages 
                           if m.timestamp < oldest_time_msg and m.id not in time_msg_ids]
            
            # Combine and sort by timestamp
            combined = older_messages + time_based_messages
            combined.sort(key=lambda m: m.timestamp)
            return combined
        
        return recent_messages
    
    def get_expired_messages(
        self, 
        conversation_id: str,
        window_hours: Optional[int] = None
    ) -> list[Message]:
        """
        Get messages that are outside the context window.
        
        These are "expired" messages that should be archived to ChromaDB
        as memories instead of being included in direct context.
        
        Args:
            conversation_id: The conversation ID
            window_hours: Override the default window (settings.context_window_hours)
        """
        hours = window_hours or settings.context_window_hours
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM messages 
                WHERE conversation_id = ? AND timestamp <= ?
                ORDER BY timestamp ASC
                """,
                (conversation_id, cutoff.isoformat())
            ).fetchall()
        
        messages = []
        for row in rows:
            messages.append(Message(
                id=row["id"],
                role=MessageRole(row["role"]),
                content=row["content"],
                timestamp=_parse_datetime(row["timestamp"])
            ))
        
        return messages
    
    def get_all_expired_messages(
        self,
        window_hours: Optional[int] = None,
        limit: int = 1000
    ) -> list[dict]:
        """
        Get all expired messages across all conversations.
        
        Useful for batch archival to ChromaDB.
        
        Returns:
            List of dicts with message and conversation_id
        """
        hours = window_hours or settings.context_window_hours
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT m.*, c.title as conversation_title
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE m.timestamp <= ?
                ORDER BY m.timestamp ASC
                LIMIT ?
                """,
                (cutoff.isoformat(), limit)
            ).fetchall()
        
        results = []
        for row in rows:
            results.append({
                "message": Message(
                    id=row["id"],
                    role=MessageRole(row["role"]),
                    content=row["content"],
                    timestamp=_parse_datetime(row["timestamp"])
                ),
                "conversation_id": row["conversation_id"],
                "conversation_title": row["conversation_title"]
            })
        
        return results
    
    def get_context_window_stats(self, conversation_id: str) -> dict:
        """
        Get statistics about the context window for a conversation.
        
        Returns:
            Dict with counts of active/expired messages and cutoff time
        """
        cutoff = self.get_context_window_cutoff()
        
        with self._get_connection() as conn:
            active_count = conn.execute(
                """
                SELECT COUNT(*) as count FROM messages 
                WHERE conversation_id = ? AND timestamp > ?
                """,
                (conversation_id, cutoff.isoformat())
            ).fetchone()["count"]
            
            expired_count = conn.execute(
                """
                SELECT COUNT(*) as count FROM messages 
                WHERE conversation_id = ? AND timestamp <= ?
                """,
                (conversation_id, cutoff.isoformat())
            ).fetchone()["count"]
            
            oldest_active = conn.execute(
                """
                SELECT MIN(timestamp) as oldest FROM messages 
                WHERE conversation_id = ? AND timestamp > ?
                """,
                (conversation_id, cutoff.isoformat())
            ).fetchone()["oldest"]
            
            newest_expired = conn.execute(
                """
                SELECT MAX(timestamp) as newest FROM messages 
                WHERE conversation_id = ? AND timestamp <= ?
                """,
                (conversation_id, cutoff.isoformat())
            ).fetchone()["newest"]
        
        return {
            "conversation_id": conversation_id,
            "window_hours": settings.context_window_hours,
            "cutoff_time": cutoff.isoformat(),
            "active_messages": active_count,
            "expired_messages": expired_count,
            "oldest_active_message": oldest_active,
            "newest_expired_message": newest_expired
        }
    
    def mark_messages_as_archived(
        self, 
        message_ids: list[str]
    ) -> int:
        """
        Mark messages as archived (after they've been saved to ChromaDB).
        
        This adds metadata to track which messages have been processed.
        For now, we'll add an 'archived' flag to the message metadata.
        
        Note: This requires adding an 'archived' column to the schema.
        """
        if not message_ids:
            return 0
        
        logger.info(f"Marking {len(message_ids)} messages as archived")
        logger.debug(f"Message IDs (first 5): {message_ids[:5]}")
        
        # First, ensure the archived column exists
        with self._get_connection() as conn:
            try:
                conn.execute("ALTER TABLE messages ADD COLUMN archived INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                # Column already exists
                pass
            
            # Verify these IDs exist in the database
            placeholders = ",".join("?" * len(message_ids))
            check_cursor = conn.execute(
                f"SELECT id FROM messages WHERE id IN ({placeholders})",
                tuple(message_ids)
            )
            found_ids = [row[0] for row in check_cursor.fetchall()]
            logger.info(f"Found {len(found_ids)} of {len(message_ids)} message IDs in database")
            
            if len(found_ids) != len(message_ids):
                missing = set(message_ids) - set(found_ids)
                logger.warning(f"Missing IDs (first 3): {list(missing)[:3]}")
            
            cursor = conn.execute(
                f"UPDATE messages SET archived = 1 WHERE id IN ({placeholders})",
                tuple(message_ids)
            )
            rowcount = cursor.rowcount
            
            logger.info(f"UPDATE affected {rowcount} rows")
        
        return rowcount
    
    def get_unarchived_expired_messages(
        self,
        window_hours: Optional[int] = None,
        limit: int = 100,
        include_monologue: bool = True
    ) -> list[dict]:
        """
        Get expired messages that haven't been archived yet.
        
        These are candidates for archival to ChromaDB.
        
        Args:
            window_hours: Override the default context window
            limit: Maximum number of messages to return
            include_monologue: If True, decrypt and include internal monologue
        """
        hours = window_hours or settings.context_window_hours
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # Ensure archived column exists
        with self._get_connection() as conn:
            try:
                conn.execute("ALTER TABLE messages ADD COLUMN archived INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            
            rows = conn.execute(
                """
                SELECT m.*, c.title as conversation_title
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE m.timestamp <= ? AND (m.archived IS NULL OR m.archived = 0)
                ORDER BY m.timestamp ASC
                LIMIT ?
                """,
                (cutoff.isoformat(), limit)
            ).fetchall()
        
        results = []
        for row in rows:
            # Decrypt internal monologue if requested
            internal_monologue = None
            if include_monologue:
                encrypted = row["internal_monologue"] if "internal_monologue" in row.keys() else None
                internal_monologue = decrypt_monologue(encrypted)
            
            # Parse retrieved memory IDs
            memory_ids_str = row["retrieved_memory_ids"] if "retrieved_memory_ids" in row.keys() else None
            retrieved_memory_ids = memory_ids_str.split(",") if memory_ids_str else []
            
            results.append({
                "message": Message(
                    id=row["id"],
                    role=MessageRole(row["role"]),
                    content=row["content"],
                    timestamp=_parse_datetime(row["timestamp"]),
                    internal_monologue=internal_monologue,
                    retrieved_memory_ids=retrieved_memory_ids
                ),
                "conversation_id": row["conversation_id"],
                "conversation_title": row["conversation_title"]
            })
        
        return results
    
    def count_unarchived_expired_messages(
        self,
        window_hours: Optional[int] = None
    ) -> int:
        """
        Count expired messages that haven't been archived yet.
        
        More efficient than get_unarchived_expired_messages when you only need the count.
        """
        hours = window_hours or settings.context_window_hours
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        with self._get_connection() as conn:
            # Ensure archived column exists
            try:
                conn.execute("ALTER TABLE messages ADD COLUMN archived INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            
            row = conn.execute(
                """
                SELECT COUNT(*) as count
                FROM messages
                WHERE timestamp <= ? AND (archived IS NULL OR archived = 0)
                """,
                (cutoff.isoformat(),)
            ).fetchone()
        
        return row["count"] if row else 0
    
    def reset_archived_flags(self) -> int:
        """
        Reset the 'archived' flag on all messages.
        
        This allows messages to be re-processed by the archiver,
        useful when regenerating memories with updated prompts.
        
        Returns:
            Number of messages reset
        """
        with self._get_connection() as conn:
            # Ensure archived column exists
            try:
                conn.execute("ALTER TABLE messages ADD COLUMN archived INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            
            cursor = conn.execute("UPDATE messages SET archived = 0 WHERE archived = 1")
            rowcount = cursor.rowcount
            
            logger.warning(f"Reset archived flag on {rowcount} messages")
        
        return rowcount


# Singleton instance
conversation_store = ConversationStore()
