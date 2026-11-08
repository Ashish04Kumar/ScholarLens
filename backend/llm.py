import sys

import groq
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from backend.settings import CHAT_MODEL, FALLBACK_CHAT_MODEL

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────

GROQ_MODEL = CHAT_MODEL
FALLBACK_MODEL = FALLBACK_CHAT_MODEL

# gpt-oss on Groq sometimes emits a malformed tool call (e.g. a tool name with a
# leaked "<|channel|>commentary" token), which Groq rejects with a 400. It is
# random, so retrying usually succeeds. Same for JSON with a wrong/missing key.
GROQ_TOOL_RETRY = dict(
    retry_if_exception_type=(groq.BadRequestError, OutputParserException), stop_after_attempt=3
)


class _LogGroqErrors(BaseCallbackHandler):
    """Print why Groq failed; the fallback otherwise hides the original error."""

    def on_llm_error(self, error, **kwargs):
        print(f"[groq failed] {type(error).__name__}: {str(error)[:300]}", file=sys.stderr, flush=True)


class FallbackLLM(Runnable):
    """Groq as primary, Gemini as fallback.

    `with_fallbacks()` returns a plain Runnable that has no `bind_tools` /
    `with_structured_output`, so those are applied to each model separately
    and then wrapped in fallbacks.
    """

    def __init__(self, groq_model: str = GROQ_MODEL):
        # Free tier is ~8k tokens/min; retry (honours retry-after) before falling back to Gemini.
        self.primary = ChatGroq(model=groq_model, max_retries=6, callbacks=[_LogGroqErrors()])
        self.fallback = ChatGoogleGenerativeAI(model=FALLBACK_MODEL)
        self._chat = self.primary.with_fallbacks([self.fallback])

    def invoke(self, input, config=None, **kwargs):
        return self._chat.invoke(input, config, **kwargs)

    async def ainvoke(self, input, config=None, **kwargs):
        return await self._chat.ainvoke(input, config, **kwargs)

    def stream(self, input, config=None, **kwargs):
        return self._chat.stream(input, config, **kwargs)

    def astream(self, input, config=None, **kwargs):
        return self._chat.astream(input, config, **kwargs)

    def with_structured_output(self, schema, **kwargs) -> Runnable:
        # Groq's gpt-oss models often skip the forced tool call in function-calling mode; JSON schema is reliable.
        primary = self.primary.with_structured_output(schema, method="json_schema", **kwargs)
        return primary.with_retry(**GROQ_TOOL_RETRY).with_fallbacks(
            [self.fallback.with_structured_output(schema, **kwargs)]
        )

    def bind_tools(self, tools, parallel_tool_calls: bool | None = None, **kwargs) -> Runnable:
        groq_kwargs = dict(kwargs)
        if parallel_tool_calls is not None:
            groq_kwargs["parallel_tool_calls"] = parallel_tool_calls
        # Gemini doesn't accept `parallel_tool_calls`; the system prompt already asks for one tool per turn.
        primary = self.primary.bind_tools(tools, **groq_kwargs)
        return primary.with_retry(**GROQ_TOOL_RETRY).with_fallbacks(
            [self.fallback.bind_tools(tools, **kwargs)]
        )
