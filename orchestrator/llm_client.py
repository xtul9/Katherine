"""
Katherine Orchestrator - LLM Client
Handles communication with OpenRouter (DeepSeek V3.2).

LLM parameters are hardcoded to match SillyTavern configuration.
Do not modify these values unless you know what you're doing.
"""
from typing import Optional, AsyncGenerator
import httpx
from loguru import logger

from config import settings


# =============================================================================
# Hardcoded LLM Parameters (matching SillyTavern configuration)
# =============================================================================

LLM_PARAMS = {
    # Token limits
    "max_tokens": 4512,
    
    # Sampling parameters
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 0,
    "min_p": 0.01,
    "typical_p": 1.0,
    "top_a": 0,
    "tfs": 1.0,
    
    # Penalty parameters
    "repetition_penalty": 1.1,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    
    # Special options
    "skip_special_tokens": True,
}

# Banned tokens/strings
BANNED_STRINGS = [
    ", if you will,",
", once a ",
" a reminder",
"adam's apple bobbing",
"admit it",
" air was filled with",
" air filled with",
" an ethereal beauty",
"another day in your life",
"arched spine",
"arousal pooling in",
"as an ai",
"as you turn to leave",
"audible pop",
"barely above a whisper",
"bites your ear",
"body and soul",
"bosomy breasts",
"breathless and eager",
"bruising kiss",
"but he can't help it",
"but she can't help it",
"but they can't help it",
"cacophony",
"catch my drift",
"choice is yours",
"chuckles darkly",
"cold and calculating",
"could not help but",
"couldn't help but",
"crinkle at the corner",
"crinkling at the corner",
"dance as old as time",
"dance of",
"deep shade",
"deeper shade",
"despite herself",
"despite himself",
"despite themselves",
"dimly lit",
"don't stop, don't ever stop",
"embark on this",
"ethereal beauty",
"evident in her eyes",
"evident in his eyes",
"evident in their eyes",
"exhausted and spent",
"her eye alight",
"her eye full",
"her eye gleam",
"her eye glint",
"her eye glow",
"her eye shin",
"her eye sparkl",
"her eye twinkl",
"her eyes alight",
"her eyes full",
"her eyes gleam",
"her eyes glint",
"her eyes glow",
"her eyes shin",
"her eyes sparkl",
"her eyes twinkl",
"his eye alight",
"his eye full",
"his eye gleam",
"his eye glint",
"his eye glow",
"his eye shin",
"his eye sparkl",
"his eye twinkl",
"his eyes alight",
"his eyes full",
"his eyes gleam",
"his eyes glint",
"his eyes glow",
"his eyes shin",
"his eyes sparkl",
"his eyes twinkl",
"their eye alight",
"their eye full",
"their eye gleam",
"their eye glint",
"their eye glow",
"their eye shin",
"their eye sparkl",
"their eye twinkl",
"their eyes alight",
"their eyes full",
"their eyes gleam",
"their eyes glint",
"their eyes glow",
"their eyes shin",
"their eyes sparkl",
"their eyes twinkl",
", eye alight",
", eye full",
", eye gleam",
", eye glint",
", eye glow",
", eye shin",
", eye sparkl",
", eye twinkl",
", eyes alight",
", eyes full",
", eyes gleam",
", eyes glint",
", eyes glow",
", eyes shin",
", eyes sparkl",
", eyes twinkl",
"'s eye alight",
"'s eye full",
"'s eye gleam",
"'s eye glint",
"'s eye glow",
"'s eye shin",
"'s eye sparkl",
"'s eye twinkl",
"'s eyes alight",
"'s eyes full",
"'s eyes gleam",
"'s eyes glint",
"'s eyes glow",
"'s eyes shin",
"'s eyes sparkl",
"'s eyes twinkl",
"felt a mix of",
"finds solace in",
"for now, that",
"for the sake of keeping things",
"for what comes next",
"for what felt like",
"for what seemed like",
"game changer",
"grins wickedly",
"grips like a vice",
"half-lidded eyes",
"haze of pleasure",
"heart, body and soul belong to you",
"heart, body, and soul belong to you",
"heaving with desire",
"her bare mound",
"her cheeks flaming",
"her heart aching",
"her wet heat",
"his wet heat",
"however, it is important",
"however, it's important",
"hum with delight",
"humble abode",
"husky voice",
"husky whispers",
"i don't bite... unless you want me to",
"important to remember that",
"in a rhythm",
"iridescent",
"journey of mutual understanding",
"kaleidoscope",
"kiss-bruised lips",
"knowing smile",
"knuckles turning white",
"knuckles whitening",
"lay ahead",
"leaves little to the imagination",
"life would never be the same",
"like a predator stalking its prey",
"like an electric shock",
"little did he kn",
"little did she kn",
"little did they kn",
"little mouse",
"long lashes",
"looking like the cat that got the cream",
"make me yours, claim me",
"mask of",
"maybe that was enough",
"maybe, just maybe",
"maybe, that was enough",
"maybe... just maybe",
"maybe you'll find you",
"mind, body, and soul",
"ministration",
"minx",
"mischie",
"moth to a flame",
"nails rake angry red lines",
"nestled deep within",
"only just getting started",
"overwhelmed by the sheer",
"palpable",
"pebbled",
"perhaps, just perhaps",
"playfully smirking",
"pleasure and pain",
"pooled around her",
"practiced ease",
"propriety be damned",
"puckered hole",
"pupils blown wide with pleasure",
"pushing aside a strand of hair",
"racing with anticipation",
"reveling in the satisfaction",
"revulsion warred with reluctance",
"revulsion warred with reluctant",
"rich tapestry",
"rivulets of",
"runs a nail",
"seductive purrs",
"sending a shive",
"sending shive",
"sends a shive",
"sends shive",
"sensitive flesh",
"sent a shive",
"sent shive",
"sent shockwaves",
"shiver down",
"shiver run",
"shiver up",
"shivers down",
"shivers run",
"shivers running up",
"shivers up",
"siren call",
"siren's call",
"slick folds",
"smirk playing on her lips",
"softly but firmly",
"soothing balm",
"stars burst behind her eyes",
"steels h",
"steeling h",
"swallowed hard",
"swallowing hard",
"swallows hard",
"sway hypnotically",
"sways hypnotically",
"symphony of",
"take your pleasure",
"tantalizing promise",
"tapestry of",
"testament to",
"that was enough",
"the air is thick",
"the ball is in your court",
"the din of the crowd",
"the game is on",
"the night is still young",
"the pressure in her loins",
"the pressure in his loins",
"the scene shifts",
"the shift in",
"the task at hand",
"the weight of",
"the world narrow",
"their wet heat",
"they would face it together",
"threatens to consume",
"torn between",
"towers over",
"tracing a finger",
"tracing a nail",
"unbeknownst to them",
"waggles her eyebrows",
"warring with",
"was only just beginning",
"was taken aback",
"waves of arousal",
"wet flesh",
"wet pop",
"what do you say",
"what felt like an eternity",
"what seemed like an eternity",
"whether you like it or not",
"whimpers, biting her lip",
"whimpers, biting his lip",
"whimpers, biting their lip",
"whispering words of passion",
"wild abandon",
"with a mix",
"with each breath",
"with each slow, deliberate movement",
"with emotion",
"with genuine emotion",
"with reckless abandon",
"without waiting for a response",
"words turn into a purr",
"you really know how to treat a lady",
"you're a bold one",
"yours to take",
"I want to know. I want to understand.",
"I want to understand.",
"barely a whisper",
"single tear slips",
"takes a deep breath",
"took a deep breath",
"with tension",
"ran a nail",
"hugs every curve",
"felt a twinge",
"curves in all the right places",
" purred",
"fidget with the hem of",
"fidgeting with the hem of",
"fidget nervously with the hem of",
"fidgeting nervously with the hem of",
"but whatever happens, we",
"are you ready to navigate the ",
"the atmosphere was charg",
"an unspoken tension in",
"filled with an unspoken",
"her body language ",
"his body language ",
"their body language ",
"I want this. I need this.",
"a smile that did not reach",
"Unless you want me to.",
"voice a low purr",
"voice a soft purr",
"couldn't help but feel",
"help but feel a sense",
"casting long shadows",
"long shadows across",
"voice barely audible",
"couldn't shake the feeling",
"couldn't help but wonder",
"sun dipped below the horizon",
"felt a chill run",
"voice steady despite",
"ready to face whatever",
", his voice barely",
", her voice barely",
", their voice barely",
", my voice barely",
" felt a strange sense",
" words hung in the air",
" hung heavy in the air",
" brow furrowed in concentration",
" air hung thick",
" air was thick with",
" renewed sense of purpose",
" something else, something",
" face whatever challenges",
" felt a sense of",
" newfound sense of",
"picking out dust mote",
"picks out dust mote",
"catching dust mote",
"catches the dust mote",
"illuminating dust mote",
"illuminates the dust mote",
"dust motes danc",
"dancing dust mote",
"dust motes swirl",
"swirling dust mote",
" her voice thick with",
" his voice thick with",
" their voice thick with",
" there's no real heat",
", though her voice lacks any real bite",
", though his voice lacks any real bite",
", though their voice lacks any real bite",
". Or something else",
"but they're together",
"And that's all that matters.",
"ozone",
"anchor"
]


# =============================================================================
# Persona Loading - System Prompt from External File
# =============================================================================

from pathlib import Path

def load_persona() -> str:
    """
    Load system prompt from external persona file.
    
    This keeps personal details out of the codebase, allowing the project
    to be published on GitHub without exposing private information.
    
    The persona file can use {user_name} and {ai_name} placeholders
    which will be replaced with values from settings.
    """
    persona_path = Path(settings.persona_file)
    
    # Try relative to orchestrator directory first
    if not persona_path.is_absolute():
        orchestrator_dir = Path(__file__).parent
        persona_path = orchestrator_dir / settings.persona_file
    
    if persona_path.exists():
        content = persona_path.read_text(encoding='utf-8')
        logger.info(f"Loaded persona from: {persona_path}")
    else:
        # Fallback to basic prompt if persona file doesn't exist
        logger.warning(
            f"Persona file not found: {persona_path}. "
            f"Copy persona.example.txt to persona.txt and customize it."
        )
        content = (
            "You are {ai_name}, a helpful AI assistant.\n\n"
            "Reference past conversations naturally when memories are provided.\n"
            "Communicate in the language {user_name} uses."
        )
    
    # Substitute placeholders with actual names from config
    return content.format(
        user_name=settings.user_name,
        ai_name=settings.ai_name
    )


# Load the system prompt at module initialization
SYSTEM_PROMPT = load_persona()


class LLMClient:
    """
    Client for communicating with OpenRouter's API (OpenAI-compatible).
    Connects to DeepSeek V3.2.
    """
    
    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self) -> None:
        """Initialize the HTTP client."""
        if not settings.openrouter_api_key:
            logger.warning("OpenRouter API key not set! Set KATHERINE_OPENROUTER_API_KEY in .env")
        
        self._client = httpx.AsyncClient(
            base_url=settings.openrouter_base_url,
            timeout=httpx.Timeout(120.0, connect=10.0),  # Longer timeout for large model
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "HTTP-Referer": settings.openrouter_site_url,
                "X-Title": settings.openrouter_site_name,
            }
        )
        logger.info(f"LLM client initialized for OpenRouter: {settings.openrouter_model}")
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def health_check(self) -> bool:
        """Check if OpenRouter is reachable."""
        if not self._client:
            return False
        
        if not settings.openrouter_api_key:
            return False
        
        try:
            response = await self._client.get("/models")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"OpenRouter health check failed: {e}")
            return False
    
    async def chat_completion(
        self,
        messages: list[dict],
        stream: bool = False
    ) -> str | AsyncGenerator[str, None]:
        """
        Send a chat completion request to OpenRouter/DeepSeek.
        
        All sampling parameters are hardcoded in LLM_PARAMS to match
        SillyTavern configuration. They cannot be overridden.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream the response
        
        Returns:
            Complete response text or async generator for streaming
        """
        if not self._client:
            raise RuntimeError("LLM client not initialized")
        
        if not settings.openrouter_api_key:
            raise RuntimeError("OpenRouter API key not configured")
        
        # Build payload with hardcoded parameters (no overrides allowed)
        payload = {
            "model": settings.openrouter_model,
            "messages": messages,
            "stream": stream,
            
            # Hardcoded sampling parameters
            "max_tokens": LLM_PARAMS["max_tokens"],
            "temperature": LLM_PARAMS["temperature"],
            "top_p": LLM_PARAMS["top_p"],
            "top_k": LLM_PARAMS["top_k"],
            "min_p": LLM_PARAMS["min_p"],
            "repetition_penalty": LLM_PARAMS["repetition_penalty"],
            "frequency_penalty": LLM_PARAMS["frequency_penalty"],
            "presence_penalty": LLM_PARAMS["presence_penalty"],
            
            # Banned strings as stop sequences
            "stop": BANNED_STRINGS,
        }
        
        if stream:
            return self._stream_completion(payload)
        else:
            return await self._sync_completion(payload)
    
    async def _sync_completion(self, payload: dict) -> str:
        """Non-streaming completion."""
        response = await self._client.post(
            "/chat/completions",
            json=payload
        )
        response.raise_for_status()
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        logger.debug(f"LLM response: {content[:100]}...")
        return content
    
    async def _stream_completion(self, payload: dict) -> AsyncGenerator[str, None]:
        """Streaming completion using SSE."""
        async with self._client.stream(
            "POST",
            "/chat/completions",
            json=payload
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        import json
                        data = json.loads(data_str)
                        
                        if "choices" in data and data["choices"]:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except Exception as e:
                        logger.warning(f"Failed to parse SSE data: {e}")
                        continue


def build_prompt_with_memories(
    user_message: str,
    conversation_history: list[dict],
    memories: list[dict],
    system_prompt: Optional[str] = None
) -> list[dict]:
    """
    Build the full prompt with Katherine's personality and injected memories.
    This is where RAG meets the soul.
    
    Args:
        user_message: Current user message
        conversation_history: Recent conversation messages
        memories: Retrieved relevant memories to inject
        system_prompt: Optional custom system prompt (defaults to Katherine's full prompt)
    
    Returns:
        List of messages ready for the LLM
    """
    # Use Katherine's full system prompt
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Inject memories as continuity context
    if memories:
        memory_context = "\n\n[RETRIEVED MEMORIES - Your actual memories from the database]\n"
        memory_context += "CRITICAL: These are the ONLY memories you have. Do NOT invent or hallucinate memories that are not listed here.\n"
        memory_context += "If asked about something not in these memories, say you don't remember or the memory wasn't retrieved.\n\n"
        memory_context += "Memory Type Guide:\n"
        memory_context += "- 'Narrative Anchor': Roleplay scenario where participants played fictional characters/roles within an imagined narrative. These are creative fiction, not real exchanges.\n"
        memory_context += "- 'Lived Moment': Direct, authentic conversation where you were aware of being an AI and the user was acknowledged as human. These are real exchanges about actual thoughts and experiences.\n\n"
        
        for i, mem in enumerate(memories, 1):
            memory_context += f"[Memory {i}]: {mem['content']}"
            
            # Add metadata in a structured way
            metadata_parts = []
            
            if mem.get('created_at'):
                metadata_parts.append(f"date: {mem['created_at']}")
            
            if mem.get('emotional_tone'):
                metadata_parts.append(f"emotional resonance: {mem['emotional_tone']}")
            
            # Show importance/confidence
            importance = mem.get('importance', 0.5)
            metadata_parts.append(f"importance: {importance:.1f}/1.0")
            
            # Show relevance to current query
            metadata_parts.append(f"relevance: {mem.get('similarity', 0):.0%}")
            
            # Show memory type (Narrative Anchor vs Lived Moment)
            tags = mem.get('tags', [])
            memory_type = None
            if 'Narrative Anchor' in tags:
                memory_type = 'Narrative Anchor'
            elif 'Lived Moment' in tags:
                memory_type = 'Lived Moment'
            
            if memory_type:
                metadata_parts.append(f"type: {memory_type}")
            
            if metadata_parts:
                memory_context += f" [{', '.join(metadata_parts)}]"
            
            memory_context += "\n"
        
        memory_context += "\n[END OF MEMORIES]\n"
        memory_context += "IMPORTANT: Use ONLY these memories. If you said something different in previous messages that contradicts these memories, the memories are CORRECT and your previous response was a hallucination. Correct yourself.\n"
        
        # Log what memories are being injected
        logger.info(f"Injecting {len(memories)} memories into prompt:")
        for i, mem in enumerate(memories, 1):
            logger.info(f"  Memory {i} ({mem.get('similarity', 0):.0%}): {mem['content'][:80]}...")
        
        # Add memories as additional system context
        messages.append({
            "role": "system",
            "content": memory_context
        })
    else:
        logger.warning("No memories to inject into prompt!")
    
    # Add conversation history
    for msg in conversation_history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # Add current user message (only if not empty)
    if user_message:
        messages.append({
            "role": "user",
            "content": user_message
        })
    
    return messages


# Singleton instance
llm_client = LLMClient()
