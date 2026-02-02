#!/usr/bin/env python3
"""
Katherine - SillyTavern Chat Importer (to SQLite)

Imports chat history from SillyTavern exports into Katherine's conversation store (SQLite).
This allows you to continue the conversation where you left off.

Unlike import_sillytavern.py (which imports to ChromaDB as memories), this script
imports the full conversation history to SQLite so you can:
- See the full chat history in the UI
- Continue the conversation naturally
- Let the archiver handle converting old messages to memories

Usage:
    python import_chat_to_sqlite.py /path/to/chat_export.jsonl
    python import_chat_to_sqlite.py /path/to/chat_export.jsonl --title "Rozmowa z Katherine"
    python import_chat_to_sqlite.py /path/to/chat_export.jsonl --conversation-id "existing-id"

After import, start the orchestrator and use the conversation ID to continue chatting.
"""

import argparse
import json
import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")


def parse_timestamp(timestamp_value) -> datetime:
    """Parse various timestamp formats from SillyTavern."""
    if not timestamp_value:
        return datetime.now(timezone.utc)
    
    try:
        if isinstance(timestamp_value, (int, float)):
            # Unix timestamp (might be milliseconds)
            ts = timestamp_value / 1000 if timestamp_value > 1e10 else timestamp_value
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        elif isinstance(timestamp_value, str):
            # ISO format string
            return datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
    except Exception:
        pass
    
    return datetime.now(timezone.utc)


def parse_sillytavern_jsonl(file_path: Path) -> list[dict]:
    """
    Parse a SillyTavern JSONL chat export.
    
    SillyTavern format (each line is a JSON object):
    {
        "name": "Katherine" or "You",
        "is_user": true/false,
        "mes": "message content",
        "send_date": timestamp,
        ...
    }
    """
    messages = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                msg = json.loads(line)
                
                # Skip system/narrator messages if they exist
                if msg.get('is_system', False) or msg.get('is_narrator', False):
                    continue
                
                content = msg.get('mes', msg.get('message', ''))
                if not content:
                    continue
                
                messages.append({
                    'role': 'user' if msg.get('is_user', False) else 'assistant',
                    'name': msg.get('name', 'Unknown'),
                    'content': content,
                    'timestamp': parse_timestamp(msg.get('send_date', msg.get('timestamp'))),
                })
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")
                continue
    
    logger.info(f"Parsed {len(messages)} messages from {file_path.name}")
    return messages


def parse_sillytavern_json(file_path: Path) -> list[dict]:
    """
    Parse a SillyTavern JSON chat export (array format).
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    messages = []
    
    # Handle different JSON structures
    if isinstance(data, list):
        chat_data = data
    elif isinstance(data, dict):
        chat_data = data.get('chat', data.get('messages', data.get('data', [])))
    else:
        chat_data = []
    
    for msg in chat_data:
        if not isinstance(msg, dict):
            continue
        
        # Skip system messages
        if msg.get('is_system', False) or msg.get('is_narrator', False):
            continue
        
        content = msg.get('mes', msg.get('message', msg.get('content', '')))
        if not content:
            continue
        
        messages.append({
            'role': 'user' if msg.get('is_user', False) else 'assistant',
            'name': msg.get('name', 'Unknown'),
            'content': content,
            'timestamp': parse_timestamp(msg.get('send_date', msg.get('timestamp'))),
        })
    
    logger.info(f"Parsed {len(messages)} messages from {file_path.name}")
    return messages


def import_to_sqlite(
    messages: list[dict],
    conversation_id: Optional[str] = None,
    title: Optional[str] = None,
    dry_run: bool = False
) -> str:
    """
    Import messages into Katherine's SQLite conversation store.
    
    Returns the conversation ID.
    """
    if dry_run:
        conv_id = conversation_id or f"preview-{uuid.uuid4().hex[:8]}"
        logger.info("DRY RUN - would import the following messages:")
        for i, msg in enumerate(messages[:10]):
            role_icon = "👤" if msg['role'] == 'user' else "👸"
            logger.info(f"  {role_icon} [{msg['timestamp'].strftime('%Y-%m-%d %H:%M')}] {msg['content'][:80]}...")
        if len(messages) > 10:
            logger.info(f"  ... and {len(messages) - 10} more messages")
        logger.info(f"Would create conversation with ID: {conv_id}")
        return conv_id
    
    # Import conversation store
    from conversation_store import conversation_store
    from models import Message, MessageRole
    
    # Generate or use provided conversation ID
    conv_id = conversation_id or str(uuid.uuid4())
    
    # Create the conversation
    if not conversation_store.conversation_exists(conv_id):
        # Generate title from first user message if not provided
        if not title:
            first_user_msg = next((m for m in messages if m['role'] == 'user'), None)
            if first_user_msg:
                title = first_user_msg['content'][:50]
                if len(first_user_msg['content']) > 50:
                    title += "..."
        
        conversation_store.create_conversation(conv_id, title=title)
        logger.info(f"Created conversation: {conv_id}")
    else:
        logger.info(f"Adding to existing conversation: {conv_id}")
    
    # Import messages
    imported = 0
    skipped = 0
    
    for msg in messages:
        try:
            message = Message(
                id=str(uuid.uuid4()),
                role=MessageRole(msg['role']),
                content=msg['content'],
                timestamp=msg['timestamp']
            )
            
            conversation_store.add_message(conv_id, message)
            imported += 1
            
            if imported % 100 == 0:
                logger.info(f"Progress: {imported}/{len(messages)} messages imported")
                
        except Exception as e:
            logger.warning(f"Failed to import message: {e}")
            skipped += 1
    
    # Update conversation title if we have one now
    if title:
        conversation_store.update_conversation(conv_id, title=title)
    
    logger.info(f"Import complete: {imported} messages imported, {skipped} skipped")
    
    # Show stats
    stats = conversation_store.get_context_window_stats(conv_id)
    logger.info(f"Conversation stats:")
    logger.info(f"  - Active messages (in {stats['window_hours']}h window): {stats['active_messages']}")
    logger.info(f"  - Expired messages (ready for archival): {stats['expired_messages']}")
    
    return conv_id


def main():
    parser = argparse.ArgumentParser(
        description="Import SillyTavern chat history into Katherine's SQLite conversation store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Import a chat file
    python import_chat_to_sqlite.py ~/SillyTavern/chats/Katherine/chat.jsonl
    
    # Import with custom title
    python import_chat_to_sqlite.py chat.jsonl --title "Nasza rozmowa"
    
    # Continue an existing conversation
    python import_chat_to_sqlite.py new_messages.jsonl --conversation-id "abc123"
    
    # Dry run to preview
    python import_chat_to_sqlite.py chat.jsonl --dry-run

After import:
    1. Note the conversation ID printed at the end
    2. Start the orchestrator: python main.py
    3. In the frontend, go to History and select the imported conversation
    4. Or use the API: POST /chat with conversation_id
    
To archive old messages to memories:
    POST /archival/run-all?mode=simple
        """
    )
    
    parser.add_argument(
        'path',
        type=Path,
        help='Path to JSONL or JSON chat export file'
    )
    
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='Title for the conversation (auto-generated if not provided)'
    )
    
    parser.add_argument(
        '--conversation-id',
        type=str,
        default=None,
        help='Existing conversation ID to append to (creates new if not provided)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be imported without actually importing'
    )
    
    args = parser.parse_args()
    
    # Validate path
    if not args.path.exists():
        logger.error(f"File not found: {args.path}")
        sys.exit(1)
    
    if not args.path.is_file():
        logger.error(f"Path must be a file: {args.path}")
        sys.exit(1)
    
    # Parse the chat file
    if args.path.suffix.lower() == '.jsonl':
        messages = parse_sillytavern_jsonl(args.path)
    else:
        messages = parse_sillytavern_json(args.path)
    
    if not messages:
        logger.error("No messages found in the chat file")
        sys.exit(1)
    
    # Sort by timestamp
    messages.sort(key=lambda m: m['timestamp'])
    
    # Show date range
    first_ts = messages[0]['timestamp']
    last_ts = messages[-1]['timestamp']
    logger.info(f"Date range: {first_ts.strftime('%Y-%m-%d %H:%M')} → {last_ts.strftime('%Y-%m-%d %H:%M')}")
    
    # Import to SQLite
    conv_id = import_to_sqlite(
        messages,
        conversation_id=args.conversation_id,
        title=args.title,
        dry_run=args.dry_run
    )
    
    if not args.dry_run:
        print()
        print("=" * 60)
        print("✓ Import complete!")
        print()
        print(f"  Conversation ID: {conv_id}")
        print()
        print("  To continue this conversation:")
        print("  1. Start the orchestrator: python main.py")
        print("  2. Open the frontend and go to 'Historia'")
        print("  3. Select the imported conversation")
        print()
        print("  To archive old messages to memories:")
        print("  POST /archival/run-all?mode=simple")
        print("=" * 60)


if __name__ == "__main__":
    main()
