"""
Katherine Orchestrator - User Tag Manager
Manages tags describing the user, as perceived by the AI.

Tags are curated by the AI itself through inner monologue.
The AI can add, remove, and reorder tags based on its evolving understanding.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from loguru import logger

from config import settings
from models import UserTag, TagChanges, TagMove


def _normalize_datetime(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware (UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_datetime(date_str: str) -> datetime:
    """Parse ISO datetime string and ensure it's timezone-aware (UTC)."""
    dt = datetime.fromisoformat(date_str)
    return _normalize_datetime(dt)


class UserTagManager:
    """
    Manages user tags stored in SQLite.
    
    Tags represent the AI's understanding of the user's characteristics.
    They are curated by the AI itself through inner monologue analysis.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the user tag manager.
        
        Args:
            db_path: Path to SQLite database (defaults to conversations.db location)
        """
        if db_path is None:
            self.db_path = str(Path(settings.chroma_persist_directory).parent / "conversations.db")
        else:
            self.db_path = db_path
        
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
        """Create user_tags table if it doesn't exist."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS user_tags (
                    tag TEXT PRIMARY KEY,
                    display_order INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_user_tags_order 
                ON user_tags(display_order);
            """)
        
        logger.debug(f"User tags schema ensured in: {self.db_path}")
    
    def get_all_tags(self) -> list[UserTag]:
        """
        Get all user tags, sorted by display_order (ascending).
        
        Returns:
            List of UserTag objects, ordered by importance/priority
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM user_tags ORDER BY display_order ASC"
            ).fetchall()
        
        tags = []
        for row in rows:
            tags.append(UserTag(
                tag=row["tag"],
                display_order=row["display_order"],
                created_at=_parse_datetime(row["created_at"]),
                updated_at=_parse_datetime(row["updated_at"])
            ))
        
        return tags
    
    def add_tag(self, tag: str) -> bool:
        """
        Add a new tag.
        
        Args:
            tag: The tag name to add
            
        Returns:
            True if tag was added, False if it already exists
        """
        # Check if tag already exists
        existing = self.get_tag(tag)
        if existing:
            logger.debug(f"Tag '{tag}' already exists, skipping add")
            return False
        
        # Get the next display_order (highest + 1)
        all_tags = self.get_all_tags()
        next_order = max([t.display_order for t in all_tags] + [-1]) + 1
        
        now = datetime.now(timezone.utc).isoformat()
        
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_tags (tag, display_order, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (tag, next_order, now, now)
            )
        
        logger.info(f"Added user tag: {tag} (order: {next_order})")
        return True
    
    def remove_tag(self, tag: str) -> bool:
        """
        Remove a tag.
        
        Args:
            tag: The tag name to remove
            
        Returns:
            True if tag was removed, False if it didn't exist
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM user_tags WHERE tag = ?",
                (tag,)
            )
        
        if cursor.rowcount > 0:
            logger.info(f"Removed user tag: {tag}")
            # Reorder remaining tags to fill gaps
            self._renumber_orders()
            return True
        
        logger.debug(f"Tag '{tag}' not found, skipping remove")
        return False
    
    def get_tag(self, tag: str) -> Optional[UserTag]:
        """Get a specific tag by name."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM user_tags WHERE tag = ?",
                (tag,)
            ).fetchone()
        
        if not row:
            return None
        
        return UserTag(
            tag=row["tag"],
            display_order=row["display_order"],
            created_at=_parse_datetime(row["created_at"]),
            updated_at=_parse_datetime(row["updated_at"])
        )
    
    def _renumber_orders(self) -> None:
        """Renumber display_order to be sequential (0, 1, 2, ...) after removals."""
        all_tags = self.get_all_tags()
        with self._get_connection() as conn:
            for i, tag in enumerate(all_tags):
                conn.execute(
                    "UPDATE user_tags SET display_order = ? WHERE tag = ?",
                    (i, tag.tag)
                )
    
    def _apply_move(self, move: TagMove) -> bool:
        """
        Apply a single MOVE instruction.
        
        Args:
            move: TagMove instruction
            
        Returns:
            True if move was applied, False if tag doesn't exist
        """
        tag_to_move = self.get_tag(move.tag)
        if not tag_to_move:
            logger.warning(f"Tag '{move.tag}' not found, cannot move")
            return False
        
        all_tags = self.get_all_tags()
        
        # Remove the tag to move from the list
        tags_without_moved = [t for t in all_tags if t.tag != move.tag]
        
        if not tags_without_moved:
            # Only one tag, no move needed
            return True
        
        # Determine new position
        if move.position == "TO_TOP":
            new_order = 0
            # Shift all other tags down
            for tag in tags_without_moved:
                self._update_order(tag.tag, tag.display_order + 1)
        elif move.position == "TO_BOTTOM":
            new_order = len(tags_without_moved)
            # No need to shift others, it goes to the end
        elif move.position.startswith("BEFORE "):
            target_tag_name = move.position.replace("BEFORE ", "").strip()
            target_tag = self.get_tag(target_tag_name)
            if not target_tag:
                logger.warning(f"Target tag '{target_tag_name}' not found for MOVE")
                return False
            new_order = target_tag.display_order
            # Shift target and all tags after it down
            for tag in tags_without_moved:
                if tag.display_order >= new_order:
                    self._update_order(tag.tag, tag.display_order + 1)
        elif move.position.startswith("AFTER "):
            target_tag_name = move.position.replace("AFTER ", "").strip()
            target_tag = self.get_tag(target_tag_name)
            if not target_tag:
                logger.warning(f"Target tag '{target_tag_name}' not found for MOVE")
                return False
            new_order = target_tag.display_order + 1
            # Shift all tags after target down
            for tag in tags_without_moved:
                if tag.display_order >= new_order:
                    self._update_order(tag.tag, tag.display_order + 1)
        else:
            logger.warning(f"Unknown MOVE position format: {move.position}")
            return False
        
        # Update the moved tag's order
        self._update_order(move.tag, new_order)
        logger.info(f"Moved tag '{move.tag}' to position {new_order} ({move.position})")
        return True
    
    def _update_order(self, tag: str, new_order: int) -> None:
        """Update display_order for a tag."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE user_tags 
                SET display_order = ?, updated_at = ?
                WHERE tag = ?
                """,
                (new_order, now, tag)
            )
    
    def apply_tag_changes(self, changes: TagChanges) -> None:
        """
        Apply tag changes from inner monologue.
        
        Args:
            changes: TagChanges object with add/remove/move instructions
        """
        if not changes.add and not changes.remove and not changes.move:
            return
        
        logger.info(f"Applying tag changes: add={len(changes.add)}, remove={len(changes.remove)}, move={len(changes.move)}")
        
        # Apply removals first (to avoid moving tags that will be removed)
        for tag in changes.remove:
            self.remove_tag(tag)
        
        # Apply additions
        for tag in changes.add:
            self.add_tag(tag)
        
        # Apply moves (in order, as they may be relative to each other)
        for move in changes.move:
            self._apply_move(move)
        
        # Final cleanup: renumber to ensure sequential ordering
        self._renumber_orders()
        
        logger.info("Tag changes applied successfully")


# Singleton instance
user_tag_manager = UserTagManager()
