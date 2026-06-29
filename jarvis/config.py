# Groq API key (or set GROQ_API_KEY env variable before launching)
import os
GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    "API KEY NOT SET — please set GROQ_API_KEY in your environment or config.py",
)

# Model — llama-3.1-8b-instant is fastest; llama-3.1-70b-versatile is smarter
LLM_MODEL    = "llama-3.1-8b-instant"

# Edge-TTS voice  (run `edge-tts --list-voices | grep en` to browse)
VOICE        = "en-GB-SoniaNeural"   # British female — classic JARVIS
SPEECH_RATE  = "+18%"                # Slightly faster = more natural AI feel
AUDIO_RATE   = 24_000                # Hz — must match edge-tts output

# Paths
INDEX_PATH   = "jarvis_vector_index"
EMBED_MODEL  = "all-MiniLM-L6-v2"

# How many past conversation turns to include as memory  (0 = disable)
HISTORY_TURNS = 100

# Minimum characters before a sentence boundary is considered real
# (prevents splitting "Dr. Smith" into two sentences)
MIN_SENTENCE_LEN = 12

EXIT_PHRASES = {
    "goodbye jarvis", "shut down", "shutdown", "power down",
    "power off", "exit", "quit", "terminate",
}