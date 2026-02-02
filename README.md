# Katherine

> Memory-augmented AI companion with persistent context through RAG

Katherine is an AI orchestrator that maintains long-term memory of conversations, enabling continuity across sessions without the cost of massive context windows. Built with RAG (Retrieval-Augmented Generation) at its core, Katherine retrieves relevant past memories to inform each response.

## The Problem

Large Language Models have a fundamental limitation: context windows. Once a conversation exceeds the available context, the model loses access to earlier information. This leads to:

- **Context rot** — degraded response quality as context fills up
- **Hallucinations** — when users reference past events outside the context, the model fabricates responses
- **Cost** — large context windows (128k+) are expensive and often unnecessary
- **Inconsistency** — without persistent memory, the model gives different answers to the same questions

## The Solution

Katherine maintains a lean context window (last 24 hours or minimum 10 messages) supplemented by semantically relevant memories retrieved from a vector database. This produces contexts under 12k tokens while maintaining full conversational continuity.

The memory system ensures that established facts remain consistent. Ask the AI its favorite song once, and it will remember the answer instead of generating a new one each time. References to past conversations are grounded in actual stored memories, not hallucinated reconstructions.

## How is it different from Lorebooks?

Traditional Lorebooks (as seen in SillyTavern and similar tools) rely on keyword or regex activation:

```
┌──────────────────────────────┬───────────────────────────────┐
│ Lorebooks                    │ Katherine                     │
├──────────────────────────────┼───────────────────────────────┤
│ Keyword/regex triggers       │ Semantic similarity search    │
│ Manual entry management      │ Automatic memory storage      │
│ Exact match required         │ Conceptual matching           │
│ No relevance scoring         │ Confidence-weighted retrieval │
│ No temporal awareness        │ Timestamp-based recall        │
└──────────────────────────────┴───────────────────────────────┘
```

**Example**: A Lorebook entry triggered by "dog" won't activate if the user says "my canine friend" or "Billy" (the dog's name). Katherine's vector search understands that these all relate to the same concept and retrieves the relevant memory regardless of phrasing.

Katherine uses ChromaDB to store mathematical representations (embeddings) of memories. When a user message arrives, it's converted to the same vector space and compared semantically — not lexically. This enables:

1. **Zero-configuration memories** — just store them, no keyword engineering required
2. **Relevance scoring** — memories are ranked by semantic similarity, high-confidence matches are prioritized
3. **Temporal queries** — the AI can recall what was discussed on specific dates ("What did I wear last Thursday?")

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                         Katherine                          │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Client     │◄──►│ Orchestrator │◄──►│  OpenRouter  │  │
│  │  (React/TS)  │    │  (FastAPI)   │    │   LLM API    │  │
│  │  :10001      │    │   :10000     │    │   (cloud)    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                              │                             │
│                              ▼                             │
│                      ┌──────────────┐                      │
│                      │  ChromaDB    │                      │
│                      │  (Vectors)   │                      │
│                      └──────────────┘                      │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### Components

1. **Orchestrator** (Python/FastAPI)
   - Memory embedding & storage via ChromaDB
   - RAG retrieval of relevant memories
   - Persona management & system prompt construction
   - Conversation history management
   - REST API for client communication

2. **Client** (TypeScript/React)
   - Chat interface with streaming responses
   - Memory visualization (shows which memories influenced each response)
   - Memory administration panel

3. **LLM Backend** (OpenRouter)
   - Cloud-hosted LLM via OpenRouter API
   - OpenAI-compatible interface
   - Model-agnostic (works with any OpenRouter model)

## Quick Start

### Prerequisites

- Python 3.10 - 3.12 (ChromaDB has compatibility issues with newer versions)
- Node.js 18+
- OpenRouter API key ([get one here](https://openrouter.ai/keys))

### 1. Configure API Access

1. Go to https://openrouter.ai/keys
2. Create a new API key
3. Add credits to your account

### 2. Start the Orchestrator

```bash
cd orchestrator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp env.example .env
# Edit .env with your API key and preferences

# Run the orchestrator
python main.py
```

The orchestrator will start on `http://localhost:10000`.

### 3. Start the Client

```bash
cd client

# Install dependencies
npm install

# Start development server
npm run dev
```

The client will be available at `http://localhost:10001`.

## How It Works

### The RAG Pipeline

```
User Message → Generate Embedding → Vector Search → Retrieve Memories → Inject into Prompt → LLM Response
```

1. **Embedding**: User messages are converted to high-dimensional vectors using sentence-transformers
2. **Retrieval**: The most semantically similar memories are found in ChromaDB
3. **Injection**: Retrieved memories are added to the LLM prompt as context
4. **Generation**: The LLM generates a response informed by relevant memories

### Memory Retrieval

Each response includes the top 5 most relevant memories, ranked by similarity score. The client displays these memories alongside responses, providing transparency into what information influenced the output.

## Importing SillyTavern History

Katherine can import existing chat history from SillyTavern.

### Quick Import

```bash
cd orchestrator
source venv/bin/activate

# Import a single chat file
python import_sillytavern.py /path/to/chat.jsonl

# Import entire chats folder
python import_sillytavern.py ~/SillyTavern/public/chats/YourCharacter/

# Dry run to preview
python import_sillytavern.py /path/to/chats/ --dry-run
```

### LLM-Extracted Memories

For higher quality memories, use the `--extract-memories` flag. This uses the configured LLM to identify and summarize meaningful moments:

```bash
python import_sillytavern.py /path/to/chats/ --extract-memories
```

This consumes API credits but produces better results — the system stores distilled insights rather than raw conversation chunks.

## Configuration

### Persona

Edit `orchestrator/persona.txt` to define the AI's identity, personality traits, and behavioral guidelines. This file is injected into every system prompt. It's just my personal opinion, but the chat bot functions significantly better when you define how it looks, how it speaks, what flaws or quirks it has.

### Environment Variables

See `orchestrator/env.example` for available configuration options including:
- API keys
- Model selection
- Memory retrieval parameters
- Context window limits

Just copy the file and rename it to `.env`.

## Limitations

- Does not completely eliminate hallucinations (though they decrease significantly with a rich memory set)
- Requires manual memory curation for best results
- Vector search quality depends on embedding model choice

## License

MIT
