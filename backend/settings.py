import os

from dotenv import load_dotenv

load_dotenv()


def env(name: str) -> str:
    """Read a required setting from the environment / .env, with a clear error if missing."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to your .env file (see .env.example).")
    return value


# ── Models (all configured in .env) ──────────────────────────────────────────
CHAT_MODEL = env("CHAT_MODEL")                   # Groq: RAG, btw side channel, session names
FALLBACK_CHAT_MODEL = env("FALLBACK_CHAT_MODEL")  # Gemini: used when Groq fails
JUDGE_MODEL = env("JUDGE_MODEL")                 # Groq: deepeval judge + golden generation
EMBEDDING_MODEL = env("EMBEDDING_MODEL")         # Cohere embeddings
EMBEDDING_DIM = int(env("EMBEDDING_DIM"))        # must match EMBEDDING_MODEL's vector size
