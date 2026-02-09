"""
Katherine Orchestrator - Thought Threading System
Tracks and links related thoughts across monologues to maintain continuity.

A "thought thread" is a conceptual topic or theme that spans multiple monologues.
Examples: "user's job search", "emotional vulnerability", "boundary setting", "self-development growth"
"""
import re
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, List
from pathlib import Path
from contextlib import contextmanager
from collections import Counter
import json

from loguru import logger
from pydantic import BaseModel, Field

from config import settings


class ThoughtThread(BaseModel):
    """Represents a thread of related thoughts across multiple monologues."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str  # Main topic/theme (e.g., "user's job search", "boundary setting")
    description: Optional[str] = None  # Optional description of the thread
    first_mentioned: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_mentioned: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message_ids: List[str] = Field(default_factory=list)  # Message IDs that belong to this thread
    evolution_summary: Optional[str] = None  # How the thought evolved over time
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ThoughtThreadManager:
    """
    Manages thought threads - tracks topics across monologues and links related thoughts.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(Path(settings.chroma_persist_directory).parent / "thought_threads.db")
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
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS thought_threads (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    description TEXT,
                    first_mentioned TEXT NOT NULL,
                    last_mentioned TEXT NOT NULL,
                    evolution_summary TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                
                CREATE TABLE IF NOT EXISTS message_thread_links (
                    message_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    relevance_score REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (message_id, thread_id),
                    FOREIGN KEY (thread_id) REFERENCES thought_threads(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_threads_topic ON thought_threads(topic);
                CREATE INDEX IF NOT EXISTS idx_threads_last_mentioned ON thought_threads(last_mentioned);
                CREATE INDEX IF NOT EXISTS idx_links_message ON message_thread_links(message_id);
                CREATE INDEX IF NOT EXISTS idx_links_thread ON message_thread_links(thread_id);
            """)
        
        logger.info(f"Thought threading database initialized: {self.db_path}")
    
    def extract_topics_from_monologue(self, monologue: str) -> List[str]:
        """
        Extract key topics/themes from a monologue.
        
        Uses simple keyword-based extraction. Can be enhanced with LLM-based extraction later.
        
        Returns:
            List of topic strings (e.g., ["user's job search", "boundary setting"])
        """
        if not monologue:
            return []
        
        # Common topic patterns based on monologue structure
        topics = []
        
        # Look for explicit topic mentions in "Influences" section
        influences_match = re.search(r'Influences:\s*(.*?)(?:\n\n|\n[A-Z]|$)', monologue, re.DOTALL | re.IGNORECASE)
        if influences_match:
            influences_text = influences_match.group(1)
            # Extract key phrases (simple heuristic)
            # Look for patterns like "Memory #X about Y", "previous thought about X", etc.
            memory_pattern = r'Memory\s+#\d+\s+about\s+([^,\n]+)'
            topics.extend(re.findall(memory_pattern, influences_text, re.IGNORECASE))
            
            thought_pattern = r'(?:previous|past|earlier)\s+thought\s+about\s+([^,\n]+)'
            topics.extend(re.findall(thought_pattern, influences_text, re.IGNORECASE))
        
        # Look for topic mentions in "Side notes"
        side_notes_match = re.search(r'Side notes:\s*(.*?)(?:\n\n|\n[A-Z]|$)', monologue, re.DOTALL | re.IGNORECASE)
        if side_notes_match:
            side_notes_text = side_notes_match.group(1)
            # Extract questions or observations that might indicate topics
            question_pattern = r'(?:wonder|question|note|observe).*?about\s+([^?\n]+)'
            topics.extend(re.findall(question_pattern, side_notes_text, re.IGNORECASE))
        
        # Look for emotional states that might indicate ongoing threads
        emotional_match = re.search(r'Emotional state:\s*(.*?)(?:\n|$)', monologue, re.IGNORECASE)
        if emotional_match:
            # High-intensity emotions might indicate important threads
            emotional_text = emotional_match.group(1)
            if any(word in emotional_text.lower() for word in ['intense', 'strong', 'deep', 'significant']):
                # Try to find what the emotion is about
                context_match = re.search(r'about\s+([^,\n]+)', monologue, re.IGNORECASE)
                if context_match:
                    topics.append(context_match.group(1).strip())
        
        # Clean and normalize topics
        cleaned_topics = []
        for topic in topics:
            topic = topic.strip().strip('.,;:!?')
            if len(topic) > 3 and len(topic) < 100:  # Reasonable length
                cleaned_topics.append(topic)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_topics = []
        for topic in cleaned_topics:
            topic_lower = topic.lower()
            if topic_lower not in seen:
                seen.add(topic_lower)
                unique_topics.append(topic)
        
        return unique_topics[:5]  # Limit to top 5 topics
    
    def find_matching_threads(self, topics: List[str], conversation_id: Optional[str] = None) -> List[ThoughtThread]:
        """
        Find existing threads that match the given topics.
        
        Args:
            topics: List of topic strings extracted from monologue
            conversation_id: Optional conversation ID to limit search
        
        Returns:
            List of matching ThoughtThread objects
        """
        if not topics:
            return []
        
        matching_threads = []
        
        with self._get_connection() as conn:
            # Search for threads with similar topics
            # Simple keyword matching - can be enhanced with semantic similarity later
            # Build OR conditions for each topic
            conditions = []
            params = []
            for topic in topics:
                conditions.append("LOWER(t.topic) LIKE ?")
                params.append(f'%{topic.lower()}%')
            
            query = f"""
                SELECT DISTINCT t.*
                FROM thought_threads t
                WHERE {' OR '.join(conditions)}
                ORDER BY t.last_mentioned DESC
                LIMIT 10
            """
            
            rows = conn.execute(query, params).fetchall()
            
            for row in rows:
                # Get message IDs for this thread
                msg_rows = conn.execute(
                    "SELECT message_id FROM message_thread_links WHERE thread_id = ?",
                    (row["id"],)
                ).fetchall()
                
                thread = ThoughtThread(
                    id=row["id"],
                    topic=row["topic"],
                    description=row["description"],
                    first_mentioned=datetime.fromisoformat(row["first_mentioned"]),
                    last_mentioned=datetime.fromisoformat(row["last_mentioned"]),
                    message_ids=[msg_row["message_id"] for msg_row in msg_rows],
                    evolution_summary=row["evolution_summary"]
                )
                matching_threads.append(thread)
        
        return matching_threads
    
    def create_or_update_thread(
        self,
        topic: str,
        message_id: str,
        description: Optional[str] = None
    ) -> ThoughtThread:
        """
        Create a new thread or update an existing one with a new message.
        
        Args:
            topic: The topic/theme of the thread
            message_id: Message ID to link to this thread
            description: Optional description
        
        Returns:
            ThoughtThread object
        """
        now = datetime.now(timezone.utc)
        
        with self._get_connection() as conn:
            # Check if thread with similar topic exists
            existing = conn.execute(
                "SELECT * FROM thought_threads WHERE LOWER(topic) = LOWER(?)",
                (topic,)
            ).fetchone()
            
            if existing:
                # Update existing thread
                thread_id = existing["id"]
                conn.execute(
                    """
                    UPDATE thought_threads 
                    SET last_mentioned = ?, description = COALESCE(?, description)
                    WHERE id = ?
                    """,
                    (now.isoformat(), description, thread_id)
                )
                
                # Link message to thread if not already linked
                conn.execute(
                    """
                    INSERT OR IGNORE INTO message_thread_links (message_id, thread_id, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (message_id, thread_id, now.isoformat())
                )
                
                # Get updated thread
                row = conn.execute("SELECT * FROM thought_threads WHERE id = ?", (thread_id,)).fetchone()
                msg_rows = conn.execute(
                    "SELECT message_id FROM message_thread_links WHERE thread_id = ?",
                    (thread_id,)
                ).fetchall()
                
                return ThoughtThread(
                    id=row["id"],
                    topic=row["topic"],
                    description=row["description"],
                    first_mentioned=datetime.fromisoformat(row["first_mentioned"]),
                    last_mentioned=datetime.fromisoformat(row["last_mentioned"]),
                    message_ids=[msg_row["message_id"] for msg_row in msg_rows],
                    evolution_summary=row["evolution_summary"]
                )
            else:
                # Create new thread
                thread_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO thought_threads (id, topic, description, first_mentioned, last_mentioned, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (thread_id, topic, description, now.isoformat(), now.isoformat(), now.isoformat())
                )
                
                # Link message to thread
                conn.execute(
                    """
                    INSERT INTO message_thread_links (message_id, thread_id, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (message_id, thread_id, now.isoformat())
                )
                
                return ThoughtThread(
                    id=thread_id,
                    topic=topic,
                    description=description,
                    first_mentioned=now,
                    last_mentioned=now,
                    message_ids=[message_id],
                    evolution_summary=None
                )
    
    def link_message_to_threads(
        self,
        message_id: str,
        monologue: str,
        conversation_id: Optional[str] = None
    ) -> List[ThoughtThread]:
        """
        Extract topics from monologue and link message to relevant threads.
        
        Args:
            message_id: ID of the message
            monologue: Internal monologue text
            conversation_id: Optional conversation ID
        
        Returns:
            List of ThoughtThread objects this message was linked to
        """
        # Extract topics
        topics = self.extract_topics_from_monologue(monologue)
        
        if not topics:
            return []
        
        # Find matching existing threads
        matching_threads = self.find_matching_threads(topics, conversation_id)
        
        linked_threads = []
        
        # Link to existing threads or create new ones
        for topic in topics:
            # Check if topic matches an existing thread
            matched = False
            for thread in matching_threads:
                if topic.lower() in thread.topic.lower() or thread.topic.lower() in topic.lower():
                    # Update existing thread
                    updated = self.create_or_update_thread(thread.topic, message_id)
                    linked_threads.append(updated)
                    matched = True
                    break
            
            if not matched:
                # Create new thread
                new_thread = self.create_or_update_thread(topic, message_id)
                linked_threads.append(new_thread)
        
        return linked_threads
    
    def get_threads_for_message(self, message_id: str) -> List[ThoughtThread]:
        """Get all threads linked to a specific message."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT t.* FROM thought_threads t
                JOIN message_thread_links l ON t.id = l.thread_id
                WHERE l.message_id = ?
                ORDER BY t.last_mentioned DESC
                """,
                (message_id,)
            ).fetchall()
            
            threads = []
            for row in rows:
                msg_rows = conn.execute(
                    "SELECT message_id FROM message_thread_links WHERE thread_id = ?",
                    (row["id"],)
                ).fetchall()
                
                threads.append(ThoughtThread(
                    id=row["id"],
                    topic=row["topic"],
                    description=row["description"],
                    first_mentioned=datetime.fromisoformat(row["first_mentioned"]),
                    last_mentioned=datetime.fromisoformat(row["last_mentioned"]),
                    message_ids=[msg_row["message_id"] for msg_row in msg_rows],
                    evolution_summary=row["evolution_summary"]
                ))
            
            return threads
    
    def get_recent_threads(
        self,
        limit: int = 10,
        since_days: Optional[int] = None
    ) -> List[ThoughtThread]:
        """
        Get recently active threads.
        
        Args:
            limit: Maximum number of threads to return
            since_days: Only return threads active within last N days
        
        Returns:
            List of ThoughtThread objects
        """
        with self._get_connection() as conn:
            if since_days:
                cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
                query = """
                    SELECT * FROM thought_threads
                    WHERE last_mentioned >= ?
                    ORDER BY last_mentioned DESC
                    LIMIT ?
                """
                params = (cutoff.isoformat(), limit)
            else:
                query = """
                    SELECT * FROM thought_threads
                    ORDER BY last_mentioned DESC
                    LIMIT ?
                """
                params = (limit,)
            
            rows = conn.execute(query, params).fetchall()
            
            threads = []
            for row in rows:
                msg_rows = conn.execute(
                    "SELECT message_id FROM message_thread_links WHERE thread_id = ?",
                    (row["id"],)
                ).fetchall()
                
                threads.append(ThoughtThread(
                    id=row["id"],
                    topic=row["topic"],
                    description=row["description"],
                    first_mentioned=datetime.fromisoformat(row["first_mentioned"]),
                    last_mentioned=datetime.fromisoformat(row["last_mentioned"]),
                    message_ids=[msg_row["message_id"] for msg_row in msg_rows],
                    evolution_summary=row["evolution_summary"]
                ))
            
            return threads


# Singleton instance
thought_thread_manager = ThoughtThreadManager()
