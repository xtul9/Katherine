#!/usr/bin/env python3
"""
Katherine - SillyTavern Chat History Importer

Imports chat history from SillyTavern JSONL exports into Katherine's memory system.
This gives Katherine continuity from day one - she remembers everything.

Usage:
    python import_sillytavern.py /path/to/chat_export.jsonl [--extract-memories]
    python import_sillytavern.py /path/to/chats_folder/ [--extract-memories]

The --extract-memories flag uses the LLM to identify and extract important moments.
Without it, raw conversation chunks are stored as memories.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import hashlib

from loguru import logger

from config import settings

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")


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
                messages.append({
                    'role': 'user' if msg.get('is_user', False) else 'assistant',
                    'name': msg.get('name', 'Unknown'),
                    'content': msg.get('mes', ''),
                    'timestamp': msg.get('send_date', ''),
                    'line_num': line_num
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
    chat_data = data if isinstance(data, list) else data.get('chat', data.get('messages', []))
    
    for i, msg in enumerate(chat_data):
        if isinstance(msg, dict):
            messages.append({
                'role': 'user' if msg.get('is_user', False) else 'assistant',
                'name': msg.get('name', 'Unknown'),
                'content': msg.get('mes', msg.get('message', msg.get('content', ''))),
                'timestamp': msg.get('send_date', msg.get('timestamp', '')),
                'line_num': i + 1
            })
    
    logger.info(f"Parsed {len(messages)} messages from {file_path.name}")
    return messages


def chunk_conversation(messages: list[dict], chunk_size: int = 10, overlap: int = 2) -> list[dict]:
    """
    Split conversation into overlapping chunks for memory storage.
    
    Each chunk becomes a memory unit that can be retrieved.
    Overlap ensures context continuity between chunks.
    """
    chunks = []
    
    for i in range(0, len(messages), chunk_size - overlap):
        chunk_messages = messages[i:i + chunk_size]
        if len(chunk_messages) < 3:  # Skip very small chunks
            continue
        
        # Build chunk content
        content_parts = []
        for msg in chunk_messages:
            role_name = settings.user_name if msg['role'] == 'user' else settings.ai_name
            content_parts.append(f"{role_name}: {msg['content']}")
        
        content = "\n".join(content_parts)
        
        # Determine emotional tone from content (simple heuristic)
        emotional_tone = detect_emotional_tone(content)
        
        # Generate a unique ID based on content hash
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        
        # Get timestamp from first message in chunk
        timestamp = chunk_messages[0].get('timestamp', '')
        if timestamp:
            try:
                # Try to parse various timestamp formats
                if isinstance(timestamp, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp / 1000 if timestamp > 1e10 else timestamp)
                else:
                    timestamp = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
            except:
                timestamp = datetime.now(timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)
        
        chunks.append({
            'id': f"st_import_{content_hash}",
            'content': content,
            'emotional_tone': emotional_tone,
            'importance': 0.6,  # Imported memories start at moderate importance
            'timestamp': timestamp,
            'tags': ['sillytavern_import', 'conversation_chunk'],
            'source': 'sillytavern_import'
        })
    
    logger.info(f"Created {len(chunks)} conversation chunks")
    return chunks


def detect_emotional_tone(content: str) -> str:
    """
    Simple heuristic to detect emotional tone from content.
    Could be enhanced with sentiment analysis.
    """
    content_lower = content.lower()
    
    # Keywords for different tones
    tones = {
        'vulnerable': ['afraid', 'scared', 'hurt', 'pain', 'trauma', 'abuse', 'lonely', 'crying', 'tears'],
        'hopeful': ['hope', 'better', 'improve', 'progress', 'forward', 'future', 'dream'],
        'intimate': ['love', 'close', 'together', 'us', 'connection', 'feel', 'heart'],
        'therapeutic': ['understand', 'process', 'heal', 'work on', 'realize', 'learn'],
        'playful': ['haha', 'lol', 'funny', 'joke', 'tease', 'smile', 'laugh'],
        'intense': ['need', 'must', 'always', 'never', 'everything', 'nothing'],
        'reflective': ['think', 'wonder', 'remember', 'past', 'used to', 'before'],
    }
    
    scores = {tone: 0 for tone in tones}
    for tone, keywords in tones.items():
        for keyword in keywords:
            if keyword in content_lower:
                scores[tone] += 1
    
    # Return the tone with highest score, or 'neutral' if none
    max_tone = max(scores, key=scores.get)
    return max_tone if scores[max_tone] > 0 else 'neutral'


def import_to_chromadb(chunks: list[dict], dry_run: bool = False):
    """
    Import conversation chunks into Katherine's ChromaDB memory store.
    """
    if dry_run:
        logger.info("DRY RUN - would import the following memories:")
        for chunk in chunks[:5]:
            logger.info(f"  [{chunk['emotional_tone']}] {chunk['content'][:100]}...")
        if len(chunks) > 5:
            logger.info(f"  ... and {len(chunks) - 5} more")
        return
    
    # Import the memory engine
    from memory_engine import memory_engine
    from models import Memory
    
    # Initialize the engine
    logger.info("Initializing memory engine...")
    memory_engine.initialize()
    
    # Import each chunk
    imported = 0
    skipped = 0
    
    for chunk in chunks:
        try:
            memory = Memory(
                id=chunk['id'],
                content=chunk['content'],
                emotional_tone=chunk['emotional_tone'],
                importance=chunk['importance'],
                created_at=chunk['timestamp'] if isinstance(chunk['timestamp'], datetime) else datetime.now(timezone.utc),
                tags=chunk['tags']
            )
            
            memory_engine.save_memory(memory)
            imported += 1
            
            if imported % 50 == 0:
                logger.info(f"Progress: {imported}/{len(chunks)} memories imported")
                
        except Exception as e:
            logger.warning(f"Failed to import chunk {chunk['id']}: {e}")
            skipped += 1
    
    logger.info(f"Import complete: {imported} memories imported, {skipped} skipped")
    logger.info(f"Total memories in database: {memory_engine.memory_count}")


async def extract_memories_with_llm(messages: list[dict], batch_size: int = 50) -> list[dict]:
    """
    Use the LLM to extract meaningful memories from conversation history.
    This creates higher-quality, summarized memories instead of raw chunks.
    """
    import asyncio
    from llm_client import llm_client
    
    await llm_client.initialize()
    
    memories = []
    
    # Process in batches
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        
        # Build conversation text
        conv_text = "\n".join([
            f"{settings.user_name if m['role'] == 'user' else settings.ai_name}: {m['content']}"
            for m in batch
        ])
        
        extraction_prompt = [
            {
                "role": "system",
                "content": f"""You are extracting important memories from {settings.ai_name}'s conversation history with {settings.user_name}.

Focus on moments that {settings.ai_name} should remember:
- Emotional revelations or breakthroughs
- Personal information {settings.user_name} shared
- Important decisions or commitments made
- Meaningful exchanges that define their relationship
- Therapeutic insights or progress
- Vulnerable moments that require sensitivity

CRITICAL: Each memory MUST include key quotes from the conversation. {settings.ai_name} hallucinates exact words when memories don't contain original quotes.

For each memory, provide:
1. Summary: Brief context (1 sentence describing the situation)
2. Key quotes: 1-3 verbatim quotes from the conversation (MANDATORY - copy exact words from {settings.user_name} and/or {settings.ai_name})
3. Emotional tone (vulnerable, hopeful, intimate, therapeutic, playful, intense, reflective)
4. Importance score (0.0-1.0, where 1.0 is critical to remember)

Respond in JSON format:
{{"memories": [{{"summary": "...", "key_quotes": [{{"speaker": "{settings.user_name}/{settings.ai_name}", "quote": "exact words"}}], "emotional_tone": "...", "importance": 0.X}}, ...]}}

Example:
{{"memories": [{{"summary": "{settings.user_name} revealed their struggle with anxiety.", "key_quotes": [{{"speaker": "{settings.user_name}", "quote": "Sometimes the thoughts just spiral and I can't stop them."}}, {{"speaker": "{settings.ai_name}", "quote": "When that happens, try to ground yourself. What can you see right now?"}}], "emotional_tone": "therapeutic", "importance": 0.7}}]}}

Extract only genuinely significant moments. Quality over quantity."""
            },
            {
                "role": "user",
                "content": f"Extract important memories from this conversation segment:\n\n{conv_text}"
            }
        ]
        
        try:
            response = await llm_client.chat_completion(extraction_prompt)
            
            # Parse response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                
                # Find first valid timestamp in batch (skip metadata lines without send_date)
                batch_timestamp = None
                for msg in batch:
                    ts = msg.get('timestamp')
                    if ts:
                        if isinstance(ts, str):
                            try:
                                batch_timestamp = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                                break
                            except:
                                continue
                        elif isinstance(ts, datetime):
                            batch_timestamp = ts
                            break
                
                if batch_timestamp is None:
                    batch_timestamp = datetime.now(timezone.utc)
                
                for mem in data.get("memories", []):
                    # Build content with summary and key quotes
                    summary = mem.get("summary", mem.get("content", ""))
                    key_quotes = mem.get("key_quotes", [])
                    
                    content_parts = [summary]
                    if key_quotes:
                        content_parts.append("\nKey quotes:")
                        for quote in key_quotes:
                            speaker = quote.get("speaker", "Unknown")
                            text = quote.get("quote", "")
                            if text:
                                content_parts.append(f'  {speaker}: "{text}"')
                    
                    content = "\n".join(content_parts)
                    content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
                    
                    memories.append({
                        'id': f"st_extracted_{content_hash}",
                        'content': content,
                        'emotional_tone': mem.get('emotional_tone', 'neutral'),
                        'importance': mem.get('importance', 0.5),
                        'timestamp': batch_timestamp,  # Use original conversation timestamp
                        'tags': ['sillytavern_import', 'llm_extracted'],
                        'source': 'sillytavern_extraction'
                    })
                    
        except Exception as e:
            logger.warning(f"Failed to extract memories from batch {i//batch_size + 1}: {e}")
        
        logger.info(f"Processed batch {i//batch_size + 1}/{(len(messages) + batch_size - 1)//batch_size}")
    
    await llm_client.close()
    
    logger.info(f"Extracted {len(memories)} memories using LLM")
    return memories


def find_chat_files(path: Path) -> list[Path]:
    """Find all SillyTavern chat files in a path."""
    files = []
    
    if path.is_file():
        files.append(path)
    elif path.is_dir():
        # Look for JSONL and JSON files
        files.extend(path.glob("*.jsonl"))
        files.extend(path.glob("*.json"))
        # Also check subdirectories (SillyTavern often organizes by character)
        files.extend(path.glob("**/*.jsonl"))
        files.extend(path.glob("**/*.json"))
    
    # Filter to likely chat files
    chat_files = []
    for f in files:
        name_lower = f.name.lower()
        # Skip obviously non-chat files
        if 'settings' in name_lower or 'config' in name_lower:
            continue
        chat_files.append(f)
    
    return list(set(chat_files))  # Remove duplicates


def main():
    parser = argparse.ArgumentParser(
        description="Import SillyTavern chat history into Katherine's memory system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Import a single chat file (raw chunks)
    python import_sillytavern.py ~/SillyTavern/chats/Katherine/chat_2024.jsonl
    
    # Import with LLM-extracted memories (higher quality, uses API credits)
    python import_sillytavern.py ~/SillyTavern/chats/Katherine/ --extract-memories
    
    # Dry run to see what would be imported
    python import_sillytavern.py ~/SillyTavern/chats/Katherine/ --dry-run

Note: Make sure to set KATHERINE_OPENROUTER_API_KEY in .env before using --extract-memories
        """
    )
    
    parser.add_argument(
        'path',
        type=Path,
        help='Path to JSONL file or directory containing chat exports'
    )
    
    parser.add_argument(
        '--extract-memories',
        action='store_true',
        help='Use LLM to extract meaningful memories (uses API credits, higher quality)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10,
        help='Number of messages per chunk (default: 10)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be imported without actually importing'
    )
    
    args = parser.parse_args()
    
    # Find chat files
    chat_files = find_chat_files(args.path)
    
    if not chat_files:
        logger.error(f"No chat files found in {args.path}")
        sys.exit(1)
    
    logger.info(f"Found {len(chat_files)} chat file(s)")
    
    # Parse all messages
    all_messages = []
    for chat_file in chat_files:
        if chat_file.suffix == '.jsonl':
            messages = parse_sillytavern_jsonl(chat_file)
        else:
            messages = parse_sillytavern_json(chat_file)
        all_messages.extend(messages)
    
    logger.info(f"Total messages: {len(all_messages)}")
    
    if not all_messages:
        logger.error("No messages found in chat files")
        sys.exit(1)
    
    # Process messages into memories
    if args.extract_memories:
        logger.info("Extracting memories using LLM (this may take a while and use API credits)...")
        import asyncio
        memories = asyncio.run(extract_memories_with_llm(all_messages))
    else:
        logger.info("Chunking conversation into memories...")
        memories = chunk_conversation(all_messages, chunk_size=args.chunk_size)
    
    # Import to ChromaDB
    import_to_chromadb(memories, dry_run=args.dry_run)
    
    if not args.dry_run:
        logger.info("✓ Import complete! Katherine now remembers your history together.")


if __name__ == "__main__":
    main()
