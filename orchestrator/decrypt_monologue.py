#!/usr/bin/env python3
"""
Decrypt the internal monologue of the last assistant message.
Run from the orchestrator directory.
"""
import sqlite3
import base64
import hashlib
from pathlib import Path

from cryptography.fernet import Fernet


def load_encryption_key(env_path: Path) -> str:
    """Load encryption key from .env file."""
    key_name = "KATHERINE_MONOLOGUE_ENCRYPTION_KEY"
    
    if not env_path.exists():
        raise FileNotFoundError(f".env file not found at {env_path}")
    
    with open(env_path, 'r') as f:
        for line in f:
            if line.startswith(f"{key_name}="):
                return line.split('=', 1)[1].strip().strip('"\'')
    
    raise ValueError(f"{key_name} not found in .env")


def get_fernet(key: str) -> Fernet:
    """Create Fernet instance from key."""
    try:
        return Fernet(key.encode() if isinstance(key, str) else key)
    except Exception:
        # Hash the key to get consistent 32 bytes, then base64 encode
        key_bytes = hashlib.sha256(key.encode()).digest()
        fernet_key = base64.urlsafe_b64encode(key_bytes)
        return Fernet(fernet_key)


def decrypt_monologue(ciphertext: str, fernet: Fernet) -> str:
    """Decrypt internal monologue."""
    encrypted = base64.urlsafe_b64decode(ciphertext.encode('ascii'))
    decrypted = fernet.decrypt(encrypted)
    return decrypted.decode('utf-8')


def main():
    # Paths
    orchestrator_dir = Path(__file__).parent
    env_path = orchestrator_dir / ".env"
    db_path = orchestrator_dir / "data" / "conversations.db"
    
    # Load key and create Fernet
    key = load_encryption_key(env_path)
    fernet = get_fernet(key)
    
    # Connect to database and get last assistant message with monologue
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    row = conn.execute("""
        SELECT id, content, timestamp, internal_monologue 
        FROM messages 
        WHERE role = 'assistant' AND internal_monologue IS NOT NULL
        ORDER BY timestamp DESC 
        LIMIT 1
    """).fetchone()
    
    conn.close()
    
    if not row:
        print("No assistant message with internal monologue found.")
        return
    
    print(f"=== Message ID: {row['id']} ===")
    print(f"=== Timestamp: {row['timestamp']} ===")
    print()
    print("=== PUBLIC RESPONSE ===")
    print(row['content'][:500] + "..." if len(row['content']) > 500 else row['content'])
    print()
    print("=== INTERNAL MONOLOGUE (DECRYPTED) ===")
    
    try:
        decrypted = decrypt_monologue(row['internal_monologue'], fernet)
        print(decrypted)
    except Exception as e:
        print(f"Failed to decrypt: {e}")


if __name__ == "__main__":
    main()