import os
import sqlite3
import warnings
from typing import Annotated

warnings.filterwarnings("ignore", message="The default value of `allowed_objects`")

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langgraph.types import Command
from pydantic import BaseModel, Field
from tavily import TavilyClient

from backend.llm import FallbackLLM
from backend.models import ClaimVerificationResult, RelevancyDecision, RouterDecision
from backend.vector_store import search as vs_search

load_dotenv()

llm = FallbackLLM()
# Low reasoning effort: folding old turns into the chat summary is a small job.
summary_llm = FallbackLLM(fast=True)


# ── State ─────────────────────────────────────────────────────────────────────

class RAGState(MessagesState):
    session_id: str
    query: str
    route: str | None
    retrieved_docs: list[Document]
    retrieval_attempts: int
    claim_verdict: str | None
    claim_source: str | None
    superseding_papers: list[dict] | None
    answer: str | None
    is_relevant: bool | None
    rewrite_count: int
    # Rolling memory: older turns folded into a short summary, recent turns kept as they are.
    summary: str
    summarized_count: int


# ── Router ────────────────────────────────────────────────────────────────────

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a routing assistant for a research paper Q&A system. "
        "Classify the user query into exactly one of three categories:\n\n"
        "  retrieve — Use this for TWO types of questions:\n"
        "    (a) Questions about the content of uploaded research papers "
        "(e.g. methods, results, conclusions, authors).\n"
        "    (b) Questions that require live or current information that cannot be "
        "answered from general knowledge alone — such as current events, today's weather, "
        "live prices, recent news, or anything where the answer changes over time "
        "(e.g. 'Who is the current president?', 'What is the price of gold today?', "
        "'What is the weather in Delhi?').\n"
        "  verify_claim — The user wants to check whether a specific claim or finding "
        "from a paper is still accurate or has been superseded.\n"
        "  direct_answer — A stable general knowledge question answerable from training data "
        "with no retrieval needed (e.g. 'What is softmax?', 'Who invented the transformer?', "
        "'Explain backpropagation.').\n\n"
        "When in doubt between retrieve and direct_answer, prefer retrieve.\n\n"
        "Return only the route field.",
    ),
    ("human", "{query}"),
])

router_chain = ROUTER_PROMPT | llm.with_structured_output(RouterDecision)


def router_node(state: RAGState) -> dict:
    query = state["messages"][-1].content
    decision: RouterDecision = router_chain.invoke({"query": query})
    return {"route": decision.route}


# ── Tool schemas ──────────────────────────────────────────────────────────────

class RetrieverInput(BaseModel):
    query: str = Field(description="Semantic query to search research paper chunks")
    k: int = Field(default=4, ge=1, le=10, description="Number of chunks to retrieve")


class WebSearchInput(BaseModel):
    optimized_query: str = Field(description="Query rewritten and optimized for web search")
    max_results: int = Field(default=3, ge=1, le=10, description="Number of web results to return")


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool(args_schema=RetrieverInput)
def retrieve_from_vectorstore(
    query: str,
    k: int,
    session_id: Annotated[str, InjectedState("session_id")],
    current_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """Search the uploaded research paper vector store for relevant passages."""
    docs = vs_search(query=query, session_id=session_id, k=k)
    if not docs:
        return [ToolMessage(content="No relevant documents found in the vector store.", tool_call_id=tool_call_id)]
    summary = f"Retrieved {len(docs)} chunk(s) from the vector store."
    return [
        ToolMessage(content=summary, tool_call_id=tool_call_id),
        Command(update={"retrieved_docs": (current_docs or []) + docs}),
    ]


@tool(args_schema=WebSearchInput)
def web_search(
    optimized_query: str,
    max_results    : int,
    current_docs   : Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id   : Annotated[str, InjectedToolCallId],
) -> list:
    """Search the web for current or supplementary information using Tavily."""
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    results = client.search(optimized_query, max_results=max_results)
    if not results.get("results"):
        return [ToolMessage(content="No web results found.", tool_call_id=tool_call_id)]
    web_docs = [
        Document(
            page_content=r["content"],
            metadata={"url": r["url"], "title": r.get("title", "Web Result")},
        ) for r in results["results"]
    ]

    summary = f"Found {len(web_docs)} web result(s) for: {optimized_query}"
    return [ 
        ToolMessage(content=summary, tool_call_id=tool_call_id),
        Command(update={"retrieved_docs": (current_docs or []) + web_docs}),
    ]


# ── Retrieval agent singletons ────────────────────────────────────────────────

RETRIEVAL_TOOLS = [retrieve_from_vectorstore, web_search]
retrieval_llm = llm.bind_tools(RETRIEVAL_TOOLS, parallel_tool_calls=False)
base_tool_node = ToolNode(RETRIEVAL_TOOLS)

RETRIEVE_SYSTEM = (
    "You are a research assistant gathering context to answer a user's question about research papers.\n\n"
    "You have two tools available and full control over how you use them:\n\n"
    "1. retrieve_from_vectorstore — searches the uploaded paper collection.\n"
    "   You decide:\n"
    "   - query: the semantic search query (phrase it to best match relevant paper chunks)\n"
    "   - k: how many chunks to retrieve (1–10; use more for broad questions, fewer for specific ones)\n\n"
    "2. web_search — searches the live web via Tavily.\n"
    "   You decide:\n"
    "   - optimized_query: rewrite the user's question as a concise, keyword-rich web search query\n"
    "   - max_results: how many results to fetch (1–10)\n\n"
    "Choose the right source based on the question:\n"
    "- Questions about the uploaded papers → use retrieve_from_vectorstore\n"
    "- Questions about current events, recent developments, or supplementary information → use web_search\n"
    "- Call only one tool per turn.\n\n"
    "Do NOT produce a final answer. Only call tools to collect context."
)


# ── Relevancy check ───────────────────────────────────────────────────────────

RELEVANCY_CHECK_SYSTEM = (
    "You are evaluating whether retrieved document chunks are relevant enough "
    "to answer a user's question about research papers.\n\n"
    "Return is_relevant=true if the chunks contain information that meaningfully "
    "addresses the question — even partially. "
    "Return is_relevant=false only if the chunks are clearly off-topic or contain "
    "no useful information.\n\nBe lenient: if there is any substantive overlap, return true.\n"
    "Broad questions (summarise, key contributions, main results, what is this paper about) are "
    "answered from the paper as a whole: if the chunks come from a paper or document the question "
    "is about, return true even though no single chunk answers it fully.\n"
    "Chunks may start mid-sentence; judge the topic, not the formatting."
)

relevancy_llm = llm.with_structured_output(RelevancyDecision)

QUERY_REWRITE_SYSTEM = (
    "You are a query rewriting assistant for a research paper retrieval system. "
    "The previous query failed to retrieve relevant document chunks. "
    "Rewrite the query using more specific or alternative terminology, "
    "domain-specific keywords, or a narrower sub-question.\n\n"
    "Return ONLY the rewritten query as plain text. No explanation, no preamble."
)


# ── Nodes ─────────────────────────────────────────────────────────────────────

def _drop_orphan_tool_calls(messages: list) -> list:
    """Remove tool calls that never got a ToolMessage reply (and their partial replies).

    A run that fails or is interrupted mid-tool-call leaves such a message in the
    checkpoint, and chat APIs reject any later history that contains it.
    """
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    orphan_ids: set[str] = set()
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls and any(tc["id"] not in answered for tc in m.tool_calls):
            orphan_ids.update(tc["id"] for tc in m.tool_calls)
    if not orphan_ids:
        return messages
    return [
        m for m in messages
        if not (isinstance(m, AIMessage) and any(tc["id"] in orphan_ids for tc in m.tool_calls))
        and not (isinstance(m, ToolMessage) and m.tool_call_id in orphan_ids)
    ]


# ── Chat memory (rolling summary + sliding window) ────────────────────────────
# The latest RECENT_MESSAGES questions/answers are sent as they are (shortened); anything older
# is folded into one running summary. Folding happens in batches of SUMMARY_BATCH so it costs
# one small LLM call every couple of turns, not every turn.
RECENT_MESSAGES = 8          # 4 question + answer pairs
RECENT_CHARS = 800           # per message, so one long answer can't blow up the prompt
SUMMARY_BATCH = 4            # fold 2 pairs at a time

SUMMARY_SYSTEM = (
    "You maintain a short running summary of a conversation between a user and a research "
    "assistant about research papers. Merge the new exchanges into the existing summary. "
    "Keep the topics, papers, key facts, numbers and conclusions the user may refer back to; "
    "drop greetings and filler. Plain text, at most 180 words. Return only the summary."
)


def _conversation(messages: list) -> list:
    """Only the user's real questions and the final answers (no tool calls, results or rewrites)."""
    return [
        m for m in messages
        if (isinstance(m, HumanMessage) and m.name != "query_rewrite")
        or (isinstance(m, AIMessage) and m.name == "answer")
    ]


def _memory_block(state: RAGState) -> str:
    """Summary of older turns + the recent turns as text, for prompts. Excludes the current question."""
    msgs = state["messages"]
    last_h = max((i for i, m in enumerate(msgs) if isinstance(m, HumanMessage) and m.name != "query_rewrite"),
                 default=len(msgs))
    # Everything not yet folded into the summary (the window plus up to one pending batch),
    # so no turn falls in the gap between the summary and the window.
    recent = _conversation(msgs[:last_h])[state.get("summarized_count", 0) or 0:]
    parts = []
    if state.get("summary"):
        parts.append(f"Summary of earlier conversation:\n{state['summary']}")
    if recent:
        lines = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.text[:RECENT_CHARS]}" for m in recent
        )
        parts.append(f"Recent conversation:\n{lines}")
    return "\n\n".join(parts)


def _agent_context(messages: list) -> list:
    """Only the current turn (question, any rewrite, its tool calls/results).

    The checkpointer keeps the whole chat, including every earlier tool result (paper chunks,
    web pages). Sending all of it made each question bigger than the last, so long chats hit
    Groq's tokens-per-minute limit and slowed down. Earlier turns reach the agent through
    _memory_block instead.
    """
    last_h = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
    return _drop_orphan_tool_calls(messages[last_h:])


def summarize_node(state: RAGState) -> dict:
    """After the answer: fold turns that slid out of the recent window into the summary."""
    convo = _conversation(state["messages"])
    done = state.get("summarized_count", 0) or 0
    upto = len(convo) - RECENT_MESSAGES
    if upto - done < SUMMARY_BATCH:
        return {}
    batch = convo[done:upto]
    # An old chat that predates the summary folds many turns at once; cap the total (~6k chars)
    # so that one call stays inside the free tier's tokens-per-minute limit.
    per_msg = max(300, min(1500, 6000 // len(batch)))
    new = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.text[:per_msg]}" for m in batch
    )
    try:
        summary = summary_llm.invoke([
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": f"Existing summary:\n{state.get('summary') or '(none)'}\n\nNew exchanges:\n{new}"},
        ]).text.strip()
    except Exception:
        # The answer is already shown; a failed summary just retries on a later turn.
        return {}
    return {"summary": summary, "summarized_count": upto}


def agent_node(state: RAGState) -> dict:
    current_attempts = state.get("retrieval_attempts", 0)
    # Once at the cap, use plain LLM so the agent cannot emit more tool calls.
    # This prevents orphaned tool_call IDs from entering the persisted message history.
    # retrieval llm --> tool call --> tool result
    # llm --> no tools are bounded --> tool call
    # Groq's gpt-oss still emits tool calls without tools bound (400 tool_use_failed),
    # and this message's text is never used, so skip the LLM call at the cap.
    if current_attempts >= MAX_RETRIEVAL_ATTEMPTS:
        return {"messages": [AIMessage(content="Retrieval limit reached.")]}
    system = RETRIEVE_SYSTEM
    memory = _memory_block(state)
    if memory:
        system += "\n\nUse this conversation history to resolve follow-ups (\"it\", \"that paper\"):\n\n" + memory
    messages = [{"role": "system", "content": system}] + _agent_context(state["messages"])
    response = retrieval_llm.invoke(messages)
    updates: dict = {"messages": [response]}
    if getattr(response, "tool_calls", None):
        updates["retrieval_attempts"] = current_attempts + 1
    return updates


def _user_question(state: RAGState) -> str:
    """The user's own question; state["query"] becomes the rewritten search query after a rewrite."""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage) and m.name != "query_rewrite":
            return m.text
    return state["query"]


RELEVANCY_CHUNKS = 6
RELEVANCY_CHARS = 600


def relevancy_check_node(state: RAGState) -> dict:
    # Judge against the user's question, not a rewritten search query: a rewrite like
    # "...in each of the uploaded articles" made one paper's chunks look off-topic.
    query = _user_question(state)
    docs = state.get("retrieved_docs") or []
    # Every chunk (up to 6) at 600 chars, with its source title; 3 x 300 chars was too little
    # to judge and often only showed a chunk's ragged start.
    doc_snippets = "\n\n---\n\n".join(
        f"[{(doc.metadata or {}).get('title') or 'source'}] {doc.page_content[:RELEVANCY_CHARS]}"
        for doc in docs[:RELEVANCY_CHUNKS]
    )
    if not doc_snippets:
        return {"is_relevant": False}
    prompt = (
        f"Question: {query}\n\nRetrieved chunks:\n{doc_snippets}\n\n"
        "Are these chunks relevant to answering the question?"
    )
    decision: RelevancyDecision = relevancy_llm.invoke([
        {"role": "system", "content": RELEVANCY_CHECK_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    return {"is_relevant": decision.is_relevant}


def query_rewrite_node(state: RAGState) -> dict:
    original_query = state["query"]
    rewrite_count = state.get("rewrite_count", 0)
    response = llm.invoke([
        {"role": "system", "content": QUERY_REWRITE_SYSTEM},
        {"role": "user", "content": f"Original query: {original_query}\n\nWrite an improved search query."},
    ])
    rewritten = response.text.strip()
    return {
        "messages": [HumanMessage(content=rewritten, name="query_rewrite")],
        "query": rewritten,
        "retrieved_docs": [],
        "retrieval_attempts": 0,
        "rewrite_count": rewrite_count + 1,
        "is_relevant": None,
    }


CLAIM_ANALYSIS_PROMPT = (
    "You are a research fact-checker. Given a claim from a research paper and "
    "a set of recent web and arXiv search results, determine:\n"
    "1. Has this claim been superseded, significantly challenged, or updated by more recent work?\n"
    "2. Identify up to 3 papers from the provided results that supersede or update the claim.\n\n"
    "Rules:\n"
    "- Use ONLY titles and URLs that appear verbatim in the provided search results.\n"
    "- Prefer arXiv paper links (arxiv.org) over general web links when available.\n"
    "- For each superseding paper, write one sentence explaining how it supersedes the claim.\n"
    "- If the claim still holds, set is_superseded=false and return an empty superseding_papers list.\n"
    "- verdict_summary should be 1-2 sentences suitable for display to the user."
)

verification_llm = llm.with_structured_output(ClaimVerificationResult)


def verify_claim_node(state: RAGState) -> dict:
    claim = state["messages"][-1].content
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    # General web search for recent work superseding the claim
    general_results = tavily_client.search(
        f"recent research superseding: {claim[:200]}",
        max_results=5,
    ).get("results", [])

    # arXiv-targeted search via web to get paper titles and links
    arxiv_results = tavily_client.search(
        f"site:arxiv.org {claim[:200]}",
        max_results=5,
    ).get("results", [])

    # Build context block
    lines = ["=== General Web Search Results ==="]
    for r in general_results:
        lines.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r.get('content', '')[:300]}\n"
        )

    lines.append("=== arXiv Paper Search Results ===")
    for r in arxiv_results:
        lines.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r.get('content', '')[:300]}\n"
        )

    context = "\n".join(lines)

    prompt = (
        f"{CLAIM_ANALYSIS_PROMPT}\n\n"
        f"Claim to verify:\n{claim}\n\n"
        f"Search Results:\n{context}"
    )
    result: ClaimVerificationResult = verification_llm.invoke([
        {"role": "user", "content": prompt}
    ])

    papers_dicts = [p.model_dump() for p in result.superseding_papers[:3]]
    return {
        "claim_verdict": result.verdict_summary,
        "claim_source": papers_dicts[0]["url"] if papers_dicts else None,
        "superseding_papers": papers_dicts,
    }


def _with_memory(state: RAGState) -> str:
    memory = _memory_block(state)
    return f"{memory}\n\n---\n\n" if memory else ""


def generate_answer_node(state: RAGState) -> dict:
    route = state.get("route")
    query = state["query"]

    if route == "retrieve":
        docs = state.get("retrieved_docs") or []
        question = _user_question(state)
        if not docs:
            answer = (
                "I wasn't able to find relevant information in the uploaded papers "
                "to answer your question. You may want to rephrase your question "
                "or upload additional papers."
            )
        else:
            context = "\n\n---\n\n".join(doc.page_content for doc in docs)
            # Still judged off-topic after a rewrite: answer from what was found, but say so,
            # instead of a canned "couldn't find" that throws the retrieved chunks away.
            weak = state.get("is_relevant") is False and state.get("rewrite_count", 0) >= 1
            caution = (
                "The retrieved context may only partly match the question. Answer only what it "
                "supports, and say briefly what it does not cover.\n\n" if weak else ""
            )
            prompt = (
                f"{_with_memory(state)}{caution}Answer the question using this context:\n\n{context}"
                f"\n\nQuestion: {question}"
            )
            answer = llm.invoke([{"role": "user", "content": prompt}]).text

    elif route == "verify_claim":
        verdict = state.get("claim_verdict", "")
        papers = state.get("superseding_papers") or []
        claim_text = state["query"]
        if papers:
            papers_block = "\n\n".join(
                f"{i + 1}. **{p['title']}**\n   {p['summary']}\n   Link: {p['url']}"
                for i, p in enumerate(papers)
            )
            answer = (
                f"**Claim Verification Result**\n\n"
                f"> {claim_text}\n\n"
                f"**Verdict:** {verdict}\n\n"
                f"**Superseding Papers:**\n\n{papers_block}\n\n"
                f"---\n"
                f"*You can load any of these papers into your knowledge base "
                f"to continue your research with the latest findings.*"
            )
        else:
            answer = (
                f"**Claim Verification Result**\n\n"
                f"> {claim_text}\n\n"
                f"**Verdict:** {verdict}\n\n"
                f"*No papers directly superseding this claim were found in recent literature.*"
            )

    else:  # direct_answer
        prompt = f"{_with_memory(state)}Answer from your knowledge.\n\nQuestion: {query}"
        answer = llm.invoke([{"role": "user", "content": prompt}]).text

    return {"answer": answer, "messages": [AIMessage(content=answer, name="answer")]}


# ── Graph ─────────────────────────────────────────────────────────────────────

# One search per question: each extra round costs another LLM call with the whole
# context, which is what made answers slow on the free Groq tier.
MAX_RETRIEVAL_ATTEMPTS = 1


def route_query(state: RAGState) -> str:
    return state["route"]


def agent_routing(state: RAGState) -> str:
    # Always execute pending tool calls first — shortcutting here would leave
    # an AIMessage with tool_calls unmatched by ToolMessages in the checkpointer,
    # corrupting history for all future turns in the same session.
    tc = tools_condition(state)
    if tc == "tools":
        return "retrieval"
    # At the cap the agent made no LLM call; still check what the one search found,
    # so an off-topic result can trigger a single query rewrite.
    return "relevancy_check"


def after_relevancy_routing(state: RAGState) -> str:
    if state.get("is_relevant", False):
        return "generate_answer"
    if state.get("rewrite_count", 0) < 1:
        return "query_rewrite"
    return "generate_answer"


def build_graph(db_path: str = "checkpoints.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = StateGraph(RAGState)
    graph.add_node("router", router_node)
    graph.add_node("agent_node", agent_node)
    graph.add_node("retrieval", base_tool_node)
    graph.add_node("relevancy_check", relevancy_check_node)
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("verify_claim", verify_claim_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_query,
        {
            "retrieve": "agent_node",
            "verify_claim": "verify_claim",
            "direct_answer": "generate_answer",
        },
    )

    graph.add_conditional_edges(
        "agent_node",
        agent_routing,
        {
            "retrieval": "retrieval",
            "relevancy_check": "relevancy_check",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("retrieval", "agent_node")

    graph.add_conditional_edges(
        "relevancy_check",
        after_relevancy_routing,
        {"query_rewrite": "query_rewrite", "generate_answer": "generate_answer"},
    )
    graph.add_edge("query_rewrite", "agent_node")

    graph.add_edge("verify_claim", "generate_answer")
    graph.add_edge("generate_answer", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile(checkpointer=checkpointer)

