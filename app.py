import html
import json
import random
import shutil
import urllib.request
import tempfile
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import HumanMessage

from backend.btw_handler import handle_btw
from backend.llm import FallbackLLM
from backend.paper_loader import load_arxiv, load_document, load_webpage
from backend.rag_graph import build_graph
from backend.vector_store import add_paper, delete_paper, delete_session_collections, list_papers

st.set_page_config(
    page_title="ScholarLens",
    page_icon=str(Path(__file__).parent / "assets" / "favicon.svg"),
    layout="centered",
)


@st.cache_resource
def get_graph():
    return build_graph()


SESSIONS_FILE = Path("sessions.json")

# Loader text shown while each graph node is running, before the answer streams.
NODE_STATUS = {
    "router": "Understanding your question…",
    "agent_node": "Planning the search…",
    "retrieval": "Searching papers and the web…",
    "relevancy_check": "Checking relevance of results…",
    "query_rewrite": "Refining the search query…",
    "verify_claim": "Verifying the claim against recent papers…",
    "generate_answer": "Writing the answer…",
}
_rename_llm = FallbackLLM()

ASSETS = Path(__file__).parent / "assets"
USER_AVATAR = str(ASSETS / "user_avatar.svg")
ASSISTANT_AVATAR = str(ASSETS / "bot_avatar.svg")


def _svg(paths: str, size: int = 14) -> str:
    """Inline line icon (Lucide-style) that inherits the surrounding text colour."""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        f'style="vertical-align:-2px;flex-shrink:0">{paths}</svg>'
    )


ICONS = {
    "book": '<path d="M2 4h6a4 4 0 0 1 4 4v13a3 3 0 0 0-3-3H2z"/><path d="M22 4h-6a4 4 0 0 0-4 4v13a3 3 0 0 1 3-3h7z"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/>',
    "bulb": '<path d="M9 18h6"/><path d="M10 22h4"/><path d="M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.3 1 2.3h6c0-1 .4-1.8 1-2.3A7 7 0 0 0 12 2z"/>',
    "link": '<path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7"/><path d="M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>',
    "globe": '<circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15 15 0 0 1 0 20a15 15 0 0 1 0-20z"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/>',
    "alert": '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "sparkle": '<path d="M12 3l1.9 5.8L20 10.7l-6.1 1.9L12 18.5l-1.9-5.9L4 10.7l6.1-1.9z"/>',
}


def icon(name: str, size: int = 14) -> str:
    return _svg(ICONS[name], size)


ROUTE_BADGES = {
    "retrieve": ("file", "From your papers"),
    "verify_claim": ("shield", "Claim check"),
    "direct_answer": ("bulb", "General knowledge"),
}

EXAMPLE_PROMPTS = [
    (":material/summarize:", "Summarize", "Key contributions of your papers", "Summarize the key contributions of the uploaded papers."),
    (":material/fact_check:", "Verify a claim", "Is it still state of the art?", "Transformers are still the state of the art for long-context modelling."),
    (":material/travel_explore:", "Latest research", "What's new on the web", "What are the latest developments in retrieval-augmented generation?"),
    (":material/forum:", "Side question", "Quick answer, not saved", "/btw What is the difference between precision and recall?"),
]

# ── Theme ──────────────────────────────────────────────────────────────────────
# One palette per mode. The same values drive Streamlit's own theme (widgets)
# and the custom CSS below, so both always match.
PALETTES = {
    "light": {
        "bg": "#F6F8F7",
        "surface": "rgba(255,255,255,0.82)",
        "surface-solid": "#FFFFFF",
        "surface-2": "rgba(15,23,42,0.045)",
        "sidebar": "rgba(255,255,255,0.55)",
        "border": "rgba(15,23,42,0.08)",
        "border-strong": "rgba(15,23,42,0.14)",
        "glass-edge": "rgba(255,255,255,0.75)",
        "text": "#0F1F1C",
        "muted": "#5E6B68",
        "accent": "#0D9488",
        "accent-hover": "#0F766E",
        "accent-soft": "rgba(13,148,136,0.09)",
        "accent-text": "#0F766E",
        "on-accent": "#FFFFFF",
        "grad": "#0D9488",
        "grad-vivid": "linear-gradient(120deg, #0F766E, #14B8A6, #65A30D, #14B8A6, #0F766E)",
        "blob-1": "rgba(20,184,166,0.30)",
        "blob-2": "rgba(132,204,22,0.22)",
        "blob-3": "rgba(6,182,212,0.18)",
        "ring": "rgba(13,148,136,0.18)",
        "shadow": "0 1px 2px rgba(15,31,28,0.05)",
        "shadow-lg": "0 1px 3px rgba(15,31,28,0.05)",
        "shadow-accent": "0 1px 3px rgba(13,148,136,0.18)",
    },
    "dark": {
        "bg": "#070B0A",
        "surface": "rgba(255,255,255,0.045)",
        "surface-solid": "#111816",
        "surface-2": "rgba(255,255,255,0.06)",
        "sidebar": "rgba(10,16,15,0.6)",
        "border": "rgba(255,255,255,0.08)",
        "border-strong": "rgba(255,255,255,0.15)",
        "glass-edge": "rgba(255,255,255,0.10)",
        "text": "#EAF4F1",
        "muted": "#93A6A1",
        "accent": "#2DD4BF",
        "accent-hover": "#5EEAD4",
        "accent-soft": "rgba(45,212,191,0.12)",
        "accent-text": "#5EEAD4",
        "on-accent": "#04201D",
        "grad": "#2DD4BF",
        "grad-vivid": "linear-gradient(120deg, #2DD4BF, #5EEAD4, #A3E635, #5EEAD4, #2DD4BF)",
        "blob-1": "rgba(20,184,166,0.22)",
        "blob-2": "rgba(132,204,22,0.12)",
        "blob-3": "rgba(6,182,212,0.14)",
        "ring": "rgba(45,212,191,0.22)",
        "shadow": "0 1px 2px rgba(0,0,0,0.5)",
        "shadow-lg": "0 1px 3px rgba(0,0,0,0.35)",
        "shadow-accent": "0 1px 3px rgba(0,0,0,0.35)",
    },
}


def current_theme() -> str:
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    return st.session_state.theme


def apply_streamlit_theme(mode: str) -> bool:
    """Point Streamlit's theme config at `mode`. Returns True if anything changed.

    Streamlit sends the theme from config at the start of every run, so a change
    here shows up after the next rerun. The config is process-wide; each session
    re-applies its own mode on its runs.
    """
    p = PALETTES[mode]
    wanted = {
        "theme.base": mode,
        "theme.primaryColor": p["accent"],
        "theme.backgroundColor": p["bg"],
        "theme.secondaryBackgroundColor": "#EEF2F1" if mode == "light" else "#141C1A",
        "theme.textColor": p["text"],
        "theme.borderColor": "#E1E7E5" if mode == "light" else "#22302D",
        "theme.sidebar.backgroundColor": "#F1F5F4" if mode == "light" else "#0A100F",
    }
    changed = False
    for key, value in wanted.items():
        if st.get_option(key) != value:
            st._config.set_option(key, value)
            changed = True
    return changed


CSS_TEMPLATE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root { __VARS__ }

html, body, .stApp, button, input, textarea, [data-testid="stMarkdownContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    -webkit-font-smoothing: antialiased;
}
.sl-welcome h1, .sl-brand-name, .sl-chat-title {
    font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
}

/* ── Animated mesh background ──────────────────────────────────────────── */
.stApp { background: var(--sl-bg); color: var(--sl-text); }
[data-testid="stAppViewContainer"], [data-testid="stMain"], .main { background: transparent !important; }
.stApp::before, .stApp::after {
    content: ""; position: fixed; pointer-events: none; z-index: 0;
    width: 60vmax; height: 60vmax; border-radius: 50%; filter: blur(70px);
}
.stApp::before {
    top: -22vmax; left: -12vmax;
    background: radial-gradient(circle, var(--sl-blob-1), transparent 65%);
    animation: sl-float-a 22s ease-in-out infinite alternate;
}
.stApp::after {
    bottom: -26vmax; right: -14vmax;
    background: radial-gradient(circle, var(--sl-blob-2), transparent 65%);
    animation: sl-float-b 26s ease-in-out infinite alternate;
}
[data-testid="stAppViewContainer"]::before {
    content: ""; position: fixed; pointer-events: none; z-index: 0;
    top: 20vh; right: 10vw; width: 38vmax; height: 38vmax; border-radius: 50%; filter: blur(80px);
    background: radial-gradient(circle, var(--sl-blob-3), transparent 65%);
    animation: sl-float-c 30s ease-in-out infinite alternate;
}
[data-testid="stAppViewContainer"] > *, [data-testid="stSidebar"] { position: relative; z-index: 1; }
@keyframes sl-float-a { to { transform: translate(14vmax, 10vmax) scale(1.12); } }
@keyframes sl-float-b { to { transform: translate(-12vmax, -8vmax) scale(1.08); } }
@keyframes sl-float-c { to { transform: translate(-10vmax, 12vmax) scale(0.9); } }

/* ── Streamlit chrome ──────────────────────────────────────────────────── */
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stToolbarActions"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; pointer-events: none; }
header[data-testid="stHeader"] button, [data-testid="stToolbar"] { pointer-events: auto; }
.block-container { max-width: 840px; padding-top: 2.6rem; padding-bottom: 6rem; }

[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"] {
    display: flex !important; visibility: visible !important; opacity: 1 !important;
}
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"] button {
    border-radius: 12px !important;
    background: var(--sl-surface) !important; color: var(--sl-text) !important;
    backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
    border: 1px solid var(--sl-border) !important; box-shadow: var(--sl-shadow);
}

/* Glass helper */
[data-testid="stChatInput"], [class*="st-key-example_"] button,
[data-testid="stChatMessage"]:not(:has(.sl-user)) [data-testid="stChatMessageContent"],
.st-key-theme_toggle button, .sl-tip, .sl-doc {
    backdrop-filter: blur(18px) saturate(160%);
    -webkit-backdrop-filter: blur(18px) saturate(160%);
}

/* ── Theme toggle: fixed top-right ─────────────────────────────────────── */
.st-key-theme_toggle {
    position: fixed !important; top: .65rem; right: 1rem; z-index: 1000000; width: auto !important;
}
.st-key-theme_toggle button {
    border-radius: 999px !important; padding: .3rem .9rem !important; min-height: 0 !important;
    background: var(--sl-surface) !important; color: var(--sl-text) !important;
    border: 1px solid var(--sl-glass-edge) !important; box-shadow: var(--sl-shadow-lg);
    transition: transform .2s ease, box-shadow .2s ease;
}
.st-key-theme_toggle button:hover { transform: translateY(-1px); box-shadow: var(--sl-shadow-accent); }
.st-key-theme_toggle button [data-testid="stIconMaterial"] { color: var(--sl-accent); transition: transform .5s ease; }
.st-key-theme_toggle button:hover [data-testid="stIconMaterial"] { transform: rotate(-25deg); }
.st-key-theme_toggle button p { font-size: .8rem; font-weight: 600; }

/* ── Welcome ───────────────────────────────────────────────────────────── */
.sl-welcome { text-align: center; margin: 2.8rem 0 2.2rem; animation: sl-rise .6s ease both; }
.sl-mark {
    width: 58px; height: 58px; border-radius: 18px; margin: 0 auto 1.3rem;
    display: grid; place-items: center; color: var(--sl-on-accent);
    background: var(--sl-grad);
}
@keyframes sl-bob { 50% { transform: translateY(-6px) rotate(-3deg); } }
.sl-pill {
    display: inline-flex; align-items: center; gap: .4rem; margin-bottom: 1rem;
    font-size: .75rem; font-weight: 600; color: var(--sl-accent-text);
    padding: .3rem .8rem; border-radius: 999px;
    background: var(--sl-surface); border: 1px solid var(--sl-glass-edge);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    box-shadow: var(--sl-shadow);
}
.sl-welcome h1 {
    font-size: 2.6rem; font-weight: 800; letter-spacing: -0.04em; line-height: 1.1;
    color: var(--sl-text); margin: 0 0 .8rem; padding: 0;
}
.sl-grad-text {
    background: var(--sl-grad-vivid); background-size: 250% 100%;
    -webkit-background-clip: text; background-clip: text; color: transparent;
    animation: sl-shimmer 6s linear infinite;
}
@keyframes sl-shimmer { to { background-position: 250% 0; } }
.sl-welcome p { color: var(--sl-muted); font-size: 1rem; margin: 0 auto; max-width: 540px; line-height: 1.6; }

[class*="st-key-example_"] { animation: sl-rise .6s ease both; }
.st-key-example_1 { animation-delay: .06s; } .st-key-example_2 { animation-delay: .12s; } .st-key-example_3 { animation-delay: .18s; }
[class*="st-key-example_"] button {
    position: relative; overflow: hidden;
    width: 100%; min-height: 70px;
    justify-content: flex-start !important; text-align: left; gap: .7rem;
    padding: .85rem 1rem !important; border-radius: 18px !important;
    background: var(--sl-surface) !important; color: var(--sl-text) !important;
    border: 1px solid var(--sl-glass-edge) !important; box-shadow: var(--sl-shadow-lg);
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
[class*="st-key-example_"] button:hover {
    transform: translateY(-1px);
    border-color: var(--sl-accent) !important;
}
[class*="st-key-example_"] button p { font-size: .87rem; color: var(--sl-muted); text-align: left; }
[class*="st-key-example_"] button strong { color: var(--sl-text); font-weight: 600; }
[class*="st-key-example_"] button > div {
    display: flex !important; align-items: center !important; justify-content: flex-start !important;
    gap: .8rem !important; width: 100%;
}
[class*="st-key-example_"] button [data-testid="stIconMaterial"] {
    color: var(--sl-accent-text); font-size: 1.15rem;
    width: 34px; height: 34px; min-width: 34px; display: grid; place-items: center;
    border-radius: 10px; background: var(--sl-accent-soft); margin: 0 !important;
}

/* Shine sweep on hover (cards + primary buttons) */
[class*="st-key-example_"] button::after,
.st-key-new_chat button::after, .st-key-btn_add_files button::after,
.st-key-btn_load_urls button::after, .st-key-btn_load_arxiv button::after {
    content: ""; position: absolute; top: 0; left: -60%; width: 40%; height: 100%;
    background: linear-gradient(100deg, transparent, rgba(255,255,255,0.35), transparent);
    transform: skewX(-20deg); pointer-events: none;
}
[class*="st-key-example_"] button:hover::after,
.st-key-new_chat button:hover::after, .st-key-btn_add_files button:hover::after,
.st-key-btn_load_urls button:hover::after, .st-key-btn_load_arxiv button:hover::after {
    animation: sl-shine .8s ease;
}
@keyframes sl-shine { to { left: 130%; } }

/* ── Messages ──────────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent; border: none; box-shadow: none;
    padding: 0; margin-bottom: 1.3rem; gap: .75rem; align-items: flex-start;
    animation: sl-rise .4s ease both;
}
@keyframes sl-rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { line-height: 1.72; font-size: .965rem; }
[data-testid="stChatMessage"] img,
[data-testid="stChatMessage"] [data-testid*="Avatar"] {
    width: 34px !important; height: 34px !important; border-radius: 11px; flex-shrink: 0; margin-top: 2px;
}
[data-testid="stChatMessageContent"] {
    min-width: 0 !important; flex: 1 1 auto; overflow-wrap: anywhere; word-break: break-word;
}
[data-testid="stChatMessage"]:not(:has(.sl-user)) { width: 100%; }
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessageContent"] [data-testid="stElementContainer"] { max-width: 100%; min-width: 0; }
[data-testid="stChatMessageContent"] table { display: block; max-width: 100%; overflow-x: auto; border-collapse: collapse; }
[data-testid="stChatMessageContent"] pre { max-width: 100%; overflow-x: auto; }
[data-testid="stChatMessage"] code {
    border-radius: 6px; padding: .08rem .32rem; font-size: .86em;
    background: var(--sl-surface-2); color: var(--sl-text);
}

/* Assistant: LEFT, glass card */
[data-testid="stChatMessage"]:not(:has(.sl-user)) { margin-right: auto; max-width: 94%; }

[data-testid="stChatMessage"]:not(:has(.sl-user)) [data-testid="stChatMessageContent"] {
    background: var(--sl-surface);
    border: 1px solid var(--sl-glass-edge);
    border-radius: 6px 20px 20px 20px;
    padding: 1rem 1.2rem;
    box-shadow: var(--sl-shadow-lg);
}
[data-testid="stChatMessage"]:not(:has(.sl-user)) p,
[data-testid="stChatMessage"]:not(:has(.sl-user)) li { color: var(--sl-text); }

/* User: RIGHT, gradient bubble */
[data-testid="stChatMessage"]:has(.sl-user) {
    flex-direction: row-reverse; margin-left: auto; width: fit-content; max-width: 75%;
}
[data-testid="stChatMessage"]:has(.sl-user) img { border-radius: 50%; }
[data-testid="stChatMessage"]:has(.sl-user) [data-testid="stChatMessageContent"] {
    flex: 0 1 auto;
    background: var(--sl-grad);
    border-radius: 20px 6px 20px 20px;
    padding: .75rem 1.1rem;
    box-shadow: var(--sl-shadow-accent);
}
[data-testid="stChatMessage"]:has(.sl-user) p,
[data-testid="stChatMessage"]:has(.sl-user) li,
[data-testid="stChatMessage"]:has(.sl-user) a,
[data-testid="stChatMessage"]:has(.sl-user) code { color: var(--sl-on-accent) !important; }
[data-testid="stChatMessage"]:has(.sl-user) code { background: rgba(255,255,255,0.2); }
.element-container:has(.sl-user), [data-testid="stElementContainer"]:has(.sl-user) { display: none; }

/* Thinking loader: shimmering label */
[data-testid="stChatMessage"] [data-testid="stExpander"] details {
    border-radius: 14px; border: 1px solid var(--sl-glass-edge); background: var(--sl-surface);
}
[data-testid="stChatMessage"] [data-testid="stExpander"] summary p {
    font-size: .84rem; font-weight: 500;
    background: linear-gradient(90deg, var(--sl-muted) 0%, var(--sl-accent) 50%, var(--sl-muted) 100%);
    background-size: 200% 100%; -webkit-background-clip: text; background-clip: text; color: transparent;
    animation: sl-shimmer 2.4s linear infinite;
}

/* Answer meta */
.sl-meta {
    display: flex; flex-wrap: wrap; gap: .45rem;
    margin-top: .9rem; padding-top: .8rem; border-top: 1px dashed var(--sl-border-strong);
}
.sl-badge {
    display: inline-flex; align-items: center; gap: .3rem;
    font-size: .74rem; font-weight: 500; color: var(--sl-muted);
    padding: .22rem .65rem; border-radius: 999px;
    background: var(--sl-surface-2); border: 1px solid var(--sl-border);
    transition: border-color .2s ease, color .2s ease;
}
.sl-badge:hover { border-color: var(--sl-accent); color: var(--sl-text); }
.sl-badge.sl-route { color: var(--sl-accent-text); background: var(--sl-accent-soft); border-color: transparent; font-weight: 600; }

/* ── Chat input: glass with animated gradient ring on focus ─────────────── */
[data-testid="stBottom"] > div {
    background: var(--sl-surface) !important;
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border-top: 1px solid var(--sl-border);
}
[data-testid="stBottomBlockContainer"] {
    background: transparent !important;
    padding: .65rem 1rem .9rem !important;
}
[data-testid="stChatInput"] {
    border-radius: 20px !important;
    background: var(--sl-surface-solid) !important;
    border: 1px solid var(--sl-border-strong) !important;
    box-shadow: var(--sl-shadow-lg);
    transition: box-shadow .25s ease, border-color .25s ease;
}
[data-testid="stChatInput"]:focus-within {
    border: 1px solid transparent !important;
    background:
        linear-gradient(var(--sl-surface-solid), var(--sl-surface-solid)) padding-box,
        var(--sl-grad-vivid) border-box !important;
    background-size: 100% 100%, 250% 100% !important;
    animation: sl-border 4s linear infinite;
    box-shadow: 0 0 0 5px var(--sl-ring), var(--sl-shadow-lg);
}
@keyframes sl-border { to { background-position: 0 0, 250% 0; } }
[data-testid="stChatInput"] textarea { font-size: .97rem; color: var(--sl-text); }
[data-testid="stChatInputSubmitButton"] {
    border-radius: 12px; background: var(--sl-grad) !important; color: var(--sl-on-accent) !important;
    box-shadow: var(--sl-shadow-accent); transition: transform .2s ease;
}
[data-testid="stChatInputSubmitButton"]:hover { transform: scale(1.06); }

/* ── Sidebar (glass) ───────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--sl-sidebar) !important;
    backdrop-filter: blur(24px) saturate(160%); -webkit-backdrop-filter: blur(24px) saturate(160%);
    border-right: 1px solid var(--sl-glass-edge);
}
[data-testid="stSidebarContent"] { padding-top: 0; background: transparent; overflow-x: hidden !important; }
/* The sidebar header only holds the collapse button; pin it to the corner so the brand sits at the top */
[data-testid="stSidebarHeader"] {
    position: absolute; top: .9rem; right: .75rem; z-index: 5;
    height: auto !important; min-height: 0 !important; padding: 0 !important; margin: 0 !important;
}
[data-testid="stSidebarUserContent"] { padding-top: 1rem !important; }
.sl-brand { display: flex; align-items: center; gap: .7rem; margin: 0 2.6rem 1.1rem 0; }
.sl-logo {
    width: 38px; height: 38px; border-radius: 12px; display: grid; place-items: center;
    color: var(--sl-on-accent); background: var(--sl-grad); box-shadow: var(--sl-shadow-accent);
    transition: transform .4s ease;
}
.sl-brand:hover .sl-logo { transform: rotate(-8deg) scale(1.05); }
.sl-brand-name { font-size: 1.2rem; font-weight: 800; letter-spacing: -0.035em; color: var(--sl-text); }
.sl-brand-tag { font-size: .72rem; color: var(--sl-muted); margin-top: -1px; }
.sl-side-title {
    display: flex; align-items: center; gap: .4rem;
    font-size: .7rem; font-weight: 600; letter-spacing: .07em; text-transform: uppercase;
    color: var(--sl-muted); margin: 1.3rem 0 .45rem;
}
[data-testid="stSidebar"] .stButton button {
    border-radius: 12px; justify-content: flex-start; text-align: left;
    background: transparent; border: 1px solid transparent; color: var(--sl-text);
    min-height: 36px; padding: .3rem .65rem;
    transition: background .2s ease;
}
.stButton button > div { gap: .55rem !important; align-items: center !important; }
[data-testid="stSidebar"] .stButton button > div,
[data-testid="stSidebar"] .stButton button [data-testid="stMarkdownContainer"] { justify-content: flex-start; text-align: left; }
.st-key-new_chat button > div, .st-key-btn_add_files button > div,
.st-key-btn_load_urls button > div, .st-key-btn_load_arxiv button > div { justify-content: center !important; }
.st-key-chat_list { overflow-x: clip !important; }
.st-key-chat_list [data-testid="stHorizontalBlock"] { gap: .25rem !important; flex-wrap: nowrap !important; }
.st-key-chat_list [data-testid="stColumn"] { min-width: 0 !important; }
[class*="st-key-trash_"] button {
    min-height: 32px !important; width: 32px; padding: 0 !important; justify-content: center !important;
    color: var(--sl-muted) !important; opacity: .55; background: transparent !important;
}
[class*="st-key-trash_"] button:hover { opacity: 1; color: #DC2626 !important; background: rgba(220,38,38,0.08) !important; }
[class*="st-key-del_yes_"] button, [class*="st-key-del_no_"] button {
    width: 100% !important; min-height: 34px !important; padding: .3rem .6rem !important;
    border-radius: 10px !important; justify-content: center !important;
}
[class*="st-key-del_yes_"] button { color: #FFFFFF !important; background: #DC2626 !important; border: none !important; }
[class*="st-key-del_yes_"] button:hover { background: #B91C1C !important; }
[class*="st-key-del_no_"] button { color: var(--sl-text) !important; background: var(--sl-surface-solid) !important; border: 1px solid var(--sl-border-strong) !important; }
[class*="st-key-del_no_"] button:hover { background: var(--sl-surface-2) !important; }
[class*="st-key-del_yes_"] button > div, [class*="st-key-del_no_"] button > div { justify-content: center !important; }
[class*="st-key-del_yes_"] button p, [class*="st-key-del_no_"] button p { font-size: .84rem !important; font-weight: 600 !important; }
.sl-confirm {
    font-size: .8rem; color: var(--sl-text); line-height: 1.45;
    padding: .55rem .7rem; margin: .2rem 0 .4rem; border-radius: 10px;
    background: rgba(220,38,38,0.06); border: 1px solid rgba(220,38,38,0.25);
}
[data-testid="stSidebar"] .stButton button:hover { background: var(--sl-surface-2); }
[data-testid="stSidebar"] .stButton button:disabled {
    background: var(--sl-accent-soft); border-color: transparent; color: var(--sl-accent-text);
    opacity: 1; cursor: default; transform: none;
}
[data-testid="stSidebar"] .stButton button p {
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: .87rem; font-weight: 500;
}
.st-key-new_chat button,
.st-key-btn_add_files button, .st-key-btn_load_urls button, .st-key-btn_load_arxiv button {
    position: relative; overflow: hidden;
    justify-content: center !important;
    background: var(--sl-grad) !important; color: var(--sl-on-accent) !important; border: none !important;
    box-shadow: var(--sl-shadow-accent);
    transition: transform .2s ease, box-shadow .2s ease;
}
.st-key-new_chat button:hover,
.st-key-btn_add_files button:hover, .st-key-btn_load_urls button:hover, .st-key-btn_load_arxiv button:hover {
    transform: translateY(-1px) !important; background: var(--sl-grad) !important;
}
.st-key-new_chat button p,
.st-key-btn_add_files button p, .st-key-btn_load_urls button p, .st-key-btn_load_arxiv button p { font-weight: 600 !important; }
.sl-doc {
    display: flex; align-items: center; gap: .5rem;
    font-size: .83rem; color: var(--sl-text);
    padding: .45rem .65rem; margin-bottom: .35rem; border-radius: 12px;
    background: var(--sl-surface); border: 1px solid var(--sl-glass-edge);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.sl-doc svg { color: var(--sl-accent); }

/* ── "Research papers only" notice ─────────────────────────────────────── */
.sl-notice {
    display: flex; align-items: flex-start; gap: .75rem;
    margin: 0 auto 1.6rem; max-width: 680px; text-align: left;
    padding: .9rem 1.1rem; border-radius: 14px;
    background: rgba(217,119,6,0.09);
    border: 1px solid rgba(217,119,6,0.35);
    border-left: 4px solid #D97706;
}
.sl-notice-icon {
    display: grid; place-items: center; flex-shrink: 0;
    width: 30px; height: 30px; border-radius: 9px;
    color: #B45309; background: rgba(217,119,6,0.15);
}
.sl-notice-title { font-size: .9rem; font-weight: 700; color: var(--sl-text); margin-bottom: .15rem; }
.sl-notice-text { font-size: .83rem; line-height: 1.5; color: var(--sl-muted); }
.st-key-scope_info button {
    justify-content: flex-start !important; min-height: 34px !important;
    border-radius: 12px !important; margin-bottom: .55rem;
    background: rgba(217,119,6,0.09) !important;
    border: 1px solid rgba(217,119,6,0.35) !important;
}
.st-key-scope_info button:hover { background: rgba(217,119,6,0.16) !important; }
.st-key-scope_info button p { font-size: .8rem !important; font-weight: 600 !important; color: #B45309 !important; }
.st-key-scope_info button [data-testid="stIconMaterial"] { color: #D97706 !important; }

/* ── Full-screen loader while a source is being added ──────────────────── */
/* While the loader is up, hide any open dialog (and its backdrop) so the loader is in front. */
body:has(.sl-overlay) [data-baseweb="modal"],
body:has(.sl-overlay) div[role="dialog"],
body:has(.sl-overlay) [data-testid="stDialog"] {
    display: none !important;
}
.sl-overlay {
    position: fixed; inset: 0; z-index: 2147483000;
    display: grid; place-items: center;
    background: rgba(10,16,15,0.35);
    backdrop-filter: blur(6px) saturate(120%); -webkit-backdrop-filter: blur(6px) saturate(120%);
    cursor: progress;
}
.sl-loader-card {
    width: min(420px, 90vw); text-align: center;
    padding: 1.8rem 1.6rem 1.5rem; border-radius: 20px;
    background: var(--sl-surface-solid); border: 1px solid var(--sl-border);
    box-shadow: 0 20px 60px rgba(0,0,0,0.25);
}
.sl-spinner {
    width: 46px; height: 46px; margin: 0 auto 1rem; border-radius: 50%;
    border: 4px solid var(--sl-accent-soft); border-top-color: var(--sl-accent);
    animation: sl-spin .8s linear infinite;
}
@keyframes sl-spin { to { transform: rotate(360deg); } }
.sl-loader-title { font-size: 1.05rem; font-weight: 700; color: var(--sl-text); }
.sl-loader-item {
    font-size: .85rem; color: var(--sl-accent-text); margin-top: .3rem;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.sl-loader-bar {
    position: relative; height: 4px; margin: 1.1rem 0 .8rem; border-radius: 999px;
    overflow: hidden; background: var(--sl-accent-soft);
}
.sl-loader-bar { height: 8px; margin: .55rem 0 .6rem; }
.sl-loader-bar span {
    position: absolute; top: 0; bottom: 0; left: 0; border-radius: 999px;
    background: var(--sl-accent); transition: width .4s ease;
    background-image: linear-gradient(110deg, transparent 25%, rgba(255,255,255,0.35) 50%, transparent 75%);
    background-size: 200% 100%; animation: sl-stripe 1.4s linear infinite;
}
@keyframes sl-stripe { to { background-position: -200% 0; } }
.sl-loader-pct {
    font-size: 2.1rem; font-weight: 800; letter-spacing: -0.03em; color: var(--sl-text);
    margin-top: .9rem; line-height: 1; font-variant-numeric: tabular-nums;
}
.sl-loader-pct span { font-size: 1rem; font-weight: 600; color: var(--sl-muted); margin-left: .1rem; }
.sl-loader-stage { font-size: .8rem; font-weight: 500; color: var(--sl-text); margin-bottom: .35rem; }
.sl-loader-hint { font-size: .78rem; line-height: 1.5; color: var(--sl-muted); }

/* ── "Before you upload" modal ─────────────────────────────────────────── */
div[role="dialog"] { border-radius: 20px !important; }
.sl-modal-hero {
    display: flex; gap: .9rem; align-items: flex-start;
    padding: 1rem 1.1rem; border-radius: 16px;
    background: rgba(217,119,6,0.08); border: 1px solid rgba(217,119,6,0.28);
}
.sl-modal-icon {
    display: grid; place-items: center; flex-shrink: 0;
    width: 44px; height: 44px; border-radius: 12px;
    color: #FFFFFF; background: #D97706;
}
.sl-modal-title { font-size: 1.02rem; font-weight: 700; color: var(--sl-text); margin-bottom: .25rem; }
.sl-modal-text { font-size: .86rem; line-height: 1.55; color: var(--sl-muted); }
.sl-chips { display: flex; flex-wrap: wrap; gap: .45rem; margin: .9rem 0 .2rem; }
.sl-chip {
    display: inline-flex; align-items: center; gap: .35rem;
    font-size: .76rem; font-weight: 600; padding: .3rem .7rem; border-radius: 999px;
    color: var(--sl-text); background: var(--sl-surface-2); border: 1px solid var(--sl-border);
}
.sl-chip-ok { color: var(--sl-accent-text); background: var(--sl-accent-soft); border-color: transparent; }
.sl-chip-no { color: #B91C1C; background: rgba(220,38,38,0.07); border-color: rgba(220,38,38,0.2); text-decoration: line-through; text-decoration-thickness: 1px; }
.sl-modal-section {
    font-size: .72rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase;
    color: var(--sl-muted); margin: 1.1rem 0 .3rem;
}
[class*="st-key-sample_card_"] {
    padding: .7rem .8rem; border-radius: 14px;
    background: var(--sl-surface-solid); border: 1px solid var(--sl-border);
    transition: border-color .15s ease;
}
[class*="st-key-sample_card_"]:hover { border-color: var(--sl-accent); }
.sl-sample-title { font-size: .88rem; font-weight: 600; color: var(--sl-text); line-height: 1.35; }
.sl-sample-meta { font-size: .76rem; color: var(--sl-muted); margin-top: .15rem; }
.sl-sample-meta a {
    color: var(--sl-accent-text) !important; text-decoration: none; font-weight: 600;
    display: inline-flex; align-items: center; gap: .2rem;
}
.sl-sample-meta a:hover { text-decoration: underline; }
[class*="st-key-sample_"] button {
    min-height: 34px !important; border-radius: 10px !important; justify-content: center !important;
    background: var(--sl-accent) !important; color: var(--sl-on-accent) !important; border: none !important;
}
[class*="st-key-sample_"] button p { font-size: .82rem !important; font-weight: 600 !important; color: var(--sl-on-accent) !important; }
[class*="st-key-sample_"] button:hover { background: var(--sl-accent-hover) !important; }
[class*="st-key-suggest_card_"] {
    padding: .65rem .7rem; margin-bottom: .2rem; border-radius: 14px;
    background: var(--sl-accent-soft); border: 1px dashed var(--sl-accent);
}
[class*="st-key-suggest_card_"] [data-testid="stVerticalBlock"] { gap: .4rem !important; }
[class*="st-key-suggest_card_"] [data-testid="stHorizontalBlock"] { gap: .35rem !important; flex-wrap: nowrap !important; }
[class*="st-key-suggest_card_"] [data-testid="stColumn"] { min-width: 0 !important; }
.sl-suggest-label {
    display: flex; align-items: center; gap: .3rem;
    font-size: .68rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; color: var(--sl-accent-text);
}
.sl-suggest-title {
    font-size: .84rem; font-weight: 600; color: var(--sl-text); line-height: 1.35; margin-top: .25rem; cursor: help;
}
.sl-suggest-meta { font-size: .72rem; color: var(--sl-muted); margin-top: .1rem; }
.sl-suggest-meta a { color: var(--sl-accent-text) !important; text-decoration: none; font-weight: 600; }
.sl-suggest-meta a:hover { text-decoration: underline; }
[class*="st-key-suggest_add_"] button {
    min-height: 32px !important; justify-content: center !important; border-radius: 10px !important;
    background: var(--sl-accent) !important; border: none !important;
}
[class*="st-key-suggest_add_"] button p, [class*="st-key-suggest_add_"] button [data-testid="stIconMaterial"] {
    color: var(--sl-on-accent) !important; font-size: .8rem !important; font-weight: 600 !important;
}
[class*="st-key-suggest_add_"] button:hover { background: var(--sl-accent-hover) !important; }
[class*="st-key-suggest_next_"] button {
    min-height: 32px !important; justify-content: center !important; border-radius: 10px !important;
    background: var(--sl-surface-solid) !important; border: 1px solid var(--sl-border-strong) !important;
}
.sl-or {
    text-align: center; font-size: .72rem; color: var(--sl-muted); margin: .45rem 0 .35rem;
    display: flex; align-items: center; gap: .5rem;
}
.sl-or::before, .sl-or::after { content: ""; flex: 1; height: 1px; background: var(--sl-border-strong); }
.sl-welcome-modal { text-align: center; margin: -.6rem 0 1.1rem; }
.sl-modal-logo {
    width: 56px; height: 56px; border-radius: 16px; margin: 0 auto .8rem;
    display: grid; place-items: center; color: var(--sl-on-accent); background: var(--sl-accent);
}
.sl-modal-welcome-title { font-size: 1.45rem; font-weight: 800; letter-spacing: -0.03em; color: var(--sl-text); }
.sl-modal-welcome-sub { font-size: .9rem; color: var(--sl-muted); margin-top: .25rem; }
.sl-chip-free { color: #0369A1; background: rgba(14,165,233,0.09); border-color: rgba(14,165,233,0.25); }
.sl-modal-note {
    font-size: .8rem; line-height: 1.5; color: var(--sl-muted);
    padding: .6rem .8rem; margin-top: .7rem; border-radius: 10px;
    background: var(--sl-surface-2); border: 1px dashed var(--sl-border-strong);
}
.st-key-intro_ok button {
    margin-top: .6rem; min-height: 42px !important; border-radius: 12px !important;
    background: var(--sl-accent) !important; border: none !important;
}
.st-key-intro_ok button p { color: var(--sl-on-accent) !important; font-weight: 700 !important; font-size: .92rem !important; }
.st-key-intro_ok button:hover { background: var(--sl-accent-hover) !important; }
.sl-modal-foot { font-size: .78rem; color: var(--sl-muted); margin-top: .7rem; text-align: center; }
.sl-notice-compact { margin: 0 0 .7rem; padding: .65rem .75rem; gap: .55rem; border-left-width: 3px; }
.sl-notice-compact .sl-notice-icon { width: 24px; height: 24px; border-radius: 7px; }
.sl-notice-compact .sl-notice-title { font-size: .8rem; }
.sl-notice-compact .sl-notice-text { font-size: .75rem; }
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has([class*="st-key-trash_doc_"]) {
    gap: .25rem !important; flex-wrap: nowrap !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has([class*="st-key-trash_doc_"]) [data-testid="stColumn"] { min-width: 0 !important; }
[class*="st-key-doc_"] button {
    min-height: 34px !important; border-radius: 12px !important;
    background: var(--sl-surface) !important; border: 1px solid var(--sl-glass-edge) !important;
}
[class*="st-key-doc_"] button:hover { border-color: var(--sl-accent) !important; }
[class*="st-key-doc_"] button [data-testid="stIconMaterial"] { color: var(--sl-accent); }
.st-key-viewer_head {
    position: sticky; top: 2.6rem; z-index: 40;
    padding: .7rem .9rem; margin-bottom: 1rem; border-radius: 16px;
    background: var(--sl-surface-solid); border: 1px solid var(--sl-border);
    box-shadow: var(--sl-shadow-lg);
}
.sl-viewer-title {
    display: flex; align-items: center; gap: .5rem;
    font-size: 1rem; font-weight: 700; color: var(--sl-text);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.sl-viewer-title svg { color: var(--sl-accent); flex-shrink: 0; }
.sl-viewer-meta { font-size: .78rem; color: var(--sl-muted); margin-top: .1rem; }
.st-key-viewer_download button, .st-key-close_viewer button {
    min-height: 38px !important; height: 38px; padding: 0 .9rem !important;
    border-radius: 10px !important; justify-content: center !important; white-space: nowrap;
}
.st-key-viewer_download button p, .st-key-close_viewer button p {
    white-space: nowrap !important; font-size: .86rem !important; font-weight: 600 !important;
}
.st-key-viewer_download button > div, .st-key-close_viewer button > div {
    flex-wrap: nowrap !important; gap: .4rem !important; justify-content: center !important;
}
.st-key-viewer_download button {
    background: var(--sl-surface-solid) !important; color: var(--sl-text) !important;
    border: 1px solid var(--sl-border-strong) !important;
}
.st-key-viewer_download button:hover { border-color: var(--sl-accent) !important; }
.st-key-close_viewer button {
    background: var(--sl-accent) !important; color: var(--sl-on-accent) !important;
    border: 1px solid var(--sl-accent) !important;
}
.st-key-close_viewer button p, .st-key-close_viewer button [data-testid="stIconMaterial"] { color: var(--sl-on-accent) !important; }
.st-key-close_viewer button:hover { background: var(--sl-accent-hover) !important; }
.st-key-pdf_pages [data-testid="stImage"] { margin-bottom: .2rem; }
.st-key-pdf_pages [data-testid="stCaptionContainer"] { text-align: center; margin-bottom: 1rem; }
.st-key-pdf_pages [data-testid="stImage"] img {
    border-radius: 6px; border: 1px solid var(--sl-border); box-shadow: var(--sl-shadow);
}
.sl-count {
    font-size: .68rem; font-weight: 700; padding: .02rem .45rem; border-radius: 999px;
    color: var(--sl-on-accent); background: var(--sl-grad); letter-spacing: 0;
}
.sl-tip {
    font-size: .78rem; color: var(--sl-muted); line-height: 1.55;
    padding: .7rem .8rem; border-radius: 14px;
    background: var(--sl-surface); border: 1px solid var(--sl-glass-edge);
}
.sl-tip code { background: var(--sl-accent-soft); color: var(--sl-accent-text); border-radius: 5px; padding: .05rem .3rem; }

[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: .2rem; }
[data-testid="stTabs"] [data-baseweb="tab"] { font-size: .8rem; font-weight: 500; padding: .3rem .55rem; }
[data-testid="stFileUploaderDropzone"] {
    border-radius: 14px !important; border: 1px dashed var(--sl-border-strong) !important;
    background: var(--sl-surface) !important;
}

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: var(--sl-border-strong); border-radius: 8px; }
::-webkit-scrollbar-track { background: transparent; }

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
"""


def build_css(mode: str) -> str:
    css_vars = " ".join(f"--sl-{k}: {v};" for k, v in PALETTES[mode].items())
    return CSS_TEMPLATE.replace("__VARS__", css_vars)


@st.cache_data(ttl=300, show_spinner=False)
def cached_list_papers(session_id: str) -> list[str]:
    """Qdrant scroll is slow; cache per session and clear after adding sources."""
    return list_papers(session_id)


# ── Original sources (so documents can be opened in the viewer) ──────────────
UPLOADS_DIR = Path("uploads")


def _sources_file(session_id: str) -> Path:
    return UPLOADS_DIR / session_id / "sources.json"


def load_sources(session_id: str) -> dict:
    """title -> {"kind": "file" | "web", "path": ..., "url": ...}"""
    try:
        return json.loads(_sources_file(session_id).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_source(session_id: str, title: str, kind: str, path: Path | None = None, url: str | None = None) -> None:
    sources = load_sources(session_id)
    sources[title] = {"kind": kind, "path": str(path) if path else None, "url": url}
    _sources_file(session_id).parent.mkdir(parents=True, exist_ok=True)
    _sources_file(session_id).write_text(json.dumps(sources, indent=2), encoding="utf-8")


def remove_document(session_id: str, title: str) -> None:
    """Delete one document everywhere: its chunks in Qdrant, saved file and source entry."""
    delete_paper(session_id, title)
    if (st.session_state.get("viewing_doc") or {}).get("title") == title:
        close_document()
    sources = load_sources(session_id)
    source = sources.pop(title, None)
    if source and source.get("path"):
        Path(source["path"]).unlink(missing_ok=True)
        st.session_state.get(f"processed_files_{session_id}", set()).discard(Path(source["path"]).name)
    if _sources_file(session_id).parent.exists():
        _sources_file(session_id).write_text(json.dumps(sources, indent=2), encoding="utf-8")
    for key in [k for k in st.session_state if k.startswith(f"pdf_shown_{session_id}_{title}")]:
        st.session_state.pop(key, None)
    cached_list_papers.clear()


def store_file(session_id: str, name: str, data: bytes) -> Path:
    folder = UPLOADS_DIR / session_id
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / Path(name).name
    dest.write_bytes(data)
    return dest


@st.cache_data(show_spinner=False, max_entries=64)
def render_pdf_pages(path: str, mtime: float, start: int, count: int) -> tuple[int, list[bytes]]:
    """Render PDF pages to PNG with PyMuPDF. Returns (total pages, page images)."""
    import fitz  # PyMuPDF

    with fitz.open(path) as pdf:
        total = pdf.page_count
        pages = [
            pdf[i].get_pixmap(dpi=110).tobytes("png")
            for i in range(start, min(start + count, total))
        ]
    return total, pages


PDF_PAGES_PER_LOAD = 10


def close_document() -> None:
    st.session_state.pop("viewing_doc", None)


def render_document_viewer(session_id: str, title: str) -> None:
    """Document shown in place of the chat, with a Close button to go back."""
    source = load_sources(session_id).get(title)
    path = Path(source["path"]) if source and source.get("path") else None
    is_pdf = bool(path and path.suffix.lower() == ".pdf" and path.exists())

    with st.container(key="viewer_head"):
        c_title, c_dl, c_close = st.columns([5, 1.7, 1.5], gap="small", vertical_alignment="center")
        meta = ""
        if is_pdf:
            total, _ = render_pdf_pages(str(path), path.stat().st_mtime, 0, 0)
            meta = f"{path.name} · {total} page{'s' if total != 1 else ''}"
        elif path and path.exists():
            meta = path.name
        elif source and source.get("kind") == "web":
            meta = "Web page"
        c_title.markdown(
            f'<div class="sl-viewer-title">{icon("file", 16)} {html.escape(title)}</div>'
            f'<div class="sl-viewer-meta">{html.escape(meta)}</div>',
            unsafe_allow_html=True,
        )
        if path and path.exists():
            c_dl.download_button(
                "Download", path.read_bytes(), file_name=path.name,
                icon=":material/download:", use_container_width=True, key="viewer_download",
            )
        if c_close.button("Close", key="close_viewer", icon=":material/close:", use_container_width=True,
                          help="Back to the chat"):
            close_document()
            st.rerun()

    if not source:
        st.info(
            "The original file for this document wasn't saved (it was added before document "
            "preview existed). Re-add it to be able to open it here."
        )
        return

    if source["kind"] == "web":
        st.markdown(f"This source is a web page: [{source['url']}]({source['url']})")
        st.link_button("Open web page", source["url"], icon=":material/open_in_new:")
        return

    if not path or not path.exists():
        st.warning("The saved file is missing on disk.")
        return

    if is_pdf:
        shown_key = f"pdf_shown_{session_id}_{title}"
        shown = st.session_state.get(shown_key, PDF_PAGES_PER_LOAD)
        total, pages = render_pdf_pages(str(path), path.stat().st_mtime, 0, shown)
        with st.container(key="pdf_pages"):
            for i, png in enumerate(pages, start=1):
                st.image(png, width="stretch")
                st.caption(f"Page {i} of {total}")
        if shown < total:
            if st.button(f"Show more pages ({shown} of {total} shown)", use_container_width=True, key="pdf_more"):
                st.session_state[shown_key] = shown + PDF_PAGES_PER_LOAD
                st.rerun()
    else:
        text = path.read_bytes().decode("utf-8", errors="replace")
        with st.container(border=True, key="text_doc"):
            if path.suffix.lower() in (".md", ".markdown"):
                st.markdown(text)
            else:
                st.text(text)


def load_sessions() -> dict:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_sessions(sessions_meta: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions_meta, indent=2), encoding="utf-8")


def _serialize_state(values: dict) -> dict:
    out = {}
    for k, v in values.items():
        if k == "messages":
            out[k] = [
                {
                    "type": type(m).__name__,
                    "content": (
                        m.content[:300]
                        if isinstance(m.content, str)
                        else repr(m.content)[:300]
                    ),
                }
                for m in (v or [])
            ]
        elif k == "retrieved_docs":
            out[k] = [
                {"content": d.page_content[:300], "metadata": d.metadata}
                for d in (v or [])
            ]
        else:
            out[k] = v
    return out


def md_text(text: str) -> str:
    """Escape "$" so prices like "$5 … $24" aren't rendered as LaTeX math by Streamlit."""
    return text.replace("$", r"\$")


MAX_PAPER_PAGES = 15

SCOPE_NOTICE_TITLE = "Built for research papers only"
SCOPE_NOTICE_TEXT = (
    "ScholarLens is designed to read and reason over academic research papers. "
    "Other documents (resumes, invoices, project reports, contracts, notes, etc.) "
    "are not supported, and answers about them may be inaccurate or random. "
    f"It currently runs on free-tier AI models, so papers of up to {MAX_PAPER_PAGES} pages "
    "are supported for now."
)

# Short papers (≤ 15 pages) users can try right away. (title, arXiv id, pages)
SAMPLE_PAPERS = [
    ("Deep Residual Learning for Image Recognition (ResNet)", "1512.03385", 12),
    ("Batch Normalization", "1502.03167", 11),
    ("Generative Adversarial Networks", "1406.2661", 9),
]


def scope_notice(compact: bool = False) -> str:
    cls = "sl-notice sl-notice-compact" if compact else "sl-notice"
    return (
        f'<div class="{cls}" role="note">'
        f'<span class="sl-notice-icon">{icon("alert", 16 if not compact else 14)}</span>'
        f'<div><div class="sl-notice-title">{SCOPE_NOTICE_TITLE}</div>'
        f'<div class="sl-notice-text">{SCOPE_NOTICE_TEXT}</div></div></div>'
    )


def _index(docs, session_id: str, progress, start: float, end: float) -> None:
    """Embed + store chunks, mapping batch progress onto [start, end] of the overall bar."""
    add_paper(
        docs, session_id,
        on_progress=lambda done, total: progress(start + (end - start) * done / total, f"Indexing chunks {done} / {total}"),
    )


def _load_arxiv_into(session_id: str, query: str, progress) -> str:
    progress(0.05, "Downloading and reading the PDF from ArXiv")
    docs = load_arxiv(query)
    progress(0.25, f"Split into {len(docs)} chunks")
    _index(docs, session_id, progress, 0.25, 0.9)
    pdf_url = docs[0].metadata.get("pdf_url") if docs else None
    progress(0.92, "Saving the PDF for the viewer")
    if pdf_url:
        with urllib.request.urlopen(pdf_url, timeout=60) as resp:
            pdf_path = store_file(session_id, f"{pdf_url.rsplit('/', 1)[-1]}.pdf", resp.read())
        save_source(session_id, docs[0].metadata["title"], "file", path=pdf_path, url=pdf_url)
    return docs[0].metadata.get("title") if docs else query


def _load_file_into(session_id: str, name: str, data: bytes, progress) -> str:
    tmp_path = None
    try:
        progress(0.05, "Reading the document")
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(name).suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        docs = load_document(tmp_path)
        for doc in docs:
            doc.metadata["title"] = Path(name).stem
        progress(0.15, f"Split into {len(docs)} chunks")
        _index(docs, session_id, progress, 0.15, 0.95)
        progress(0.97, "Saving the file for the viewer")
        saved = store_file(session_id, name, data)
        save_source(session_id, Path(name).stem, "file", path=saved)
        st.session_state.setdefault(f"processed_files_{session_id}", set()).add(name)
        return name
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def _load_url_into(session_id: str, url: str, progress) -> str:
    progress(0.05, "Fetching the web page")
    docs = load_webpage(url)
    progress(0.2, f"Split into {len(docs)} chunks")
    _index(docs, session_id, progress, 0.2, 0.97)
    title = docs[0].metadata.get("title") or url
    save_source(session_id, title, "web", url=url)
    return title


def queue_load(kind: str, items: list) -> None:
    """Remember what to load and rerun; the next run shows a full-screen loader while loading.

    Rerunning first also closes any open dialog, so nothing stays clickable underneath.
    """
    st.session_state.pending_load = {"kind": kind, "items": items, "session_id": st.session_state.active_session_id}
    st.rerun()


LOADER_TITLES = {"file": "Adding your paper", "url": "Loading web page", "arxiv": "Fetching paper from ArXiv"}


def _loader_html(title: str, current: str, step: int, total: int, pct: float, stage: str) -> str:
    pct_int = max(0, min(100, int(pct * 100)))
    counter = f" · item {step} of {total}" if total > 1 else ""
    return f"""
    <div class="sl-overlay" role="alertdialog" aria-busy="true" aria-live="polite">
      <div class="sl-loader-card">
        <div class="sl-spinner"></div>
        <div class="sl-loader-title">{html.escape(title)}…</div>
        <div class="sl-loader-item">{html.escape(current[:90])}</div>
        <div class="sl-loader-pct">{pct_int}<span>%</span></div>
        <div class="sl-loader-bar"><span style="width:{pct_int}%"></span></div>
        <div class="sl-loader-stage">{html.escape(stage)}{counter}</div>
        <div class="sl-loader-hint">Please keep this tab open. This can take up to a minute.</div>
      </div>
    </div>
    """


def process_pending_load() -> None:
    pending = st.session_state.get("pending_load")
    if not pending:
        return
    if not pending.get("armed"):
        # First run after the click: render the page without loading so the previous
        # run's elements (e.g. the "Before you upload" dialog) are cleared, then rerun
        # at the end of this script to do the actual load behind the overlay.
        pending["armed"] = True
        st.session_state._rerun_for_load = True
        return
    st.session_state.pop("pending_load", None)
    kind, items, sid = pending["kind"], pending["items"], pending["session_id"]
    overlay = st.empty()
    loaders = {"file": _load_file_into, "url": _load_url_into, "arxiv": _load_arxiv_into}
    total = len(items)
    for step, item in enumerate(items, start=1):
        label = item[0] if kind == "file" else item

        def progress(frac: float, stage: str, step=step, label=label) -> None:
            overall = ((step - 1) + frac) / total
            overlay.markdown(
                _loader_html(LOADER_TITLES[kind], label, step, total, overall, stage), unsafe_allow_html=True
            )

        progress(0.0, "Starting")
        try:
            args = item if kind == "file" else (item,)
            done = loaders[kind](sid, *args, progress)
            progress(1.0, "Done")
            st.toast(f"Added {done}", icon=":material/check_circle:")
        except Exception as e:
            st.session_state.setdefault("source_errors", []).append(f"Failed: {str(label)[:60]} — {e}")
    cached_list_papers.clear()
    overlay.empty()


def _scope_body(session_id: str, first_visit: bool = False) -> None:
    """Shared content of the welcome (first visit) and "Before you upload" dialogs."""
    if first_visit:
        st.markdown(
            f"""
            <div class="sl-welcome-modal">
              <div class="sl-modal-logo">{icon("book", 24)}</div>
              <div class="sl-modal-welcome-title">Welcome to ScholarLens</div>
              <div class="sl-modal-welcome-sub">Your workspace for reading, questioning and fact-checking research papers.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        f"""
        <div class="sl-modal-hero">
          <div class="sl-modal-icon">{icon("alert", 22)}</div>
          <div>
            <div class="sl-modal-title">{SCOPE_NOTICE_TITLE}</div>
            <div class="sl-modal-text">{SCOPE_NOTICE_TEXT}</div>
          </div>
        </div>
        <div class="sl-chips">
          <span class="sl-chip sl-chip-ok">{icon("file", 12)} Academic research papers</span>
          <span class="sl-chip">{icon("book", 12)} Up to {MAX_PAPER_PAGES} pages</span>
          <span class="sl-chip sl-chip-free">{icon("sparkle", 12)} Free-tier AI models</span>
          <span class="sl-chip sl-chip-no">{icon("alert", 12)} Resumes, invoices, reports</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if first_visit:
        st.markdown(
            '<div class="sl-modal-note">Because the app runs on free-tier models, answers can sometimes be '
            "slower, and daily usage limits may occasionally pause it for a while.</div>",
            unsafe_allow_html=True,
        )
    if first_visit:
        if st.button("Got it, let's start", key="intro_ok", type="primary", use_container_width=True):
            st.session_state.intro_ack = True
            st.rerun()
        return
    st.markdown('<div class="sl-modal-section">Try a sample paper</div>', unsafe_allow_html=True)
    for i, (paper_title, arxiv_id, pages) in enumerate(SAMPLE_PAPERS):
        with st.container(key=f"sample_card_{i}"):
            c_info, c_add = st.columns([3.2, 1], vertical_alignment="center")
            c_info.markdown(
                f'<div class="sl-sample-title">{html.escape(paper_title)}</div>'
                f'<div class="sl-sample-meta">arXiv {arxiv_id} · {pages} pages · '
                f'<a href="https://arxiv.org/pdf/{arxiv_id}" target="_blank" rel="noopener">'
                f'{icon("download", 12)} PDF</a></div>',
                unsafe_allow_html=True,
            )
            if c_add.button("Add", key=f"sample_{i}", icon=":material/add:", use_container_width=True,
                            help="Add this paper to the current chat"):
                queue_load("arxiv", [arxiv_id])
    st.markdown(
        '<div class="sl-modal-foot">Download a PDF to upload it yourself, or click <b>Add</b> '
        "to load it straight into this chat.</div>",
        unsafe_allow_html=True,
    )


@st.dialog("Before you upload", width="medium")
def scope_dialog(session_id: str) -> None:
    _scope_body(session_id)


@st.dialog(" ", width="medium")
def welcome_dialog(session_id: str) -> None:
    _scope_body(session_id, first_visit=True)


INTRO_COOKIE = "sl_intro_seen"


def maybe_show_welcome(session_id: str) -> None:
    """Open the welcome dialog on a browser's first visit; remember "Got it" in a cookie."""
    if st.session_state.get("intro_ack"):
        if not st.session_state.get("intro_cookie_set"):
            # components.html runs in a same-origin iframe, so this cookie belongs to the app.
            components.html(
                f"<script>document.cookie='{INTRO_COOKIE}=1; max-age=31536000; path=/; SameSite=Lax';</script>",
                height=0,
            )
            st.session_state.intro_cookie_set = True
        return
    if st.session_state.get("intro_shown") or INTRO_COOKIE in st.context.cookies:
        return
    st.session_state.intro_shown = True
    welcome_dialog(session_id)


# Research papers published as web pages (Distill.pub), suggested in the Web tab. (title, url)
WEB_SAMPLES = [
    ("Feature Visualization", "https://distill.pub/2017/feature-visualization/"),
    ("The Building Blocks of Interpretability", "https://distill.pub/2018/building-blocks/"),
    ("Attention and Augmented Recurrent Neural Networks", "https://distill.pub/2016/augmented-rnns/"),
]

SUGGEST_TEXT = {
    "file": ("Add this paper", "or upload your own · research papers only, max {pages} pages"),
    "arxiv": ("Add this paper", "or search ArXiv · title or ID, max {pages} pages"),
    "web": ("Add this page", "or paste your own links · research paper pages only"),
}


def suggestion_card(tab: str) -> None:
    """A random sample paper (one per chat) offered before the user adds their own source."""
    session_id = st.session_state.active_session_id
    samples = WEB_SAMPLES if tab == "web" else SAMPLE_PAPERS
    key = f"suggested_{tab}_{session_id}"
    if key not in st.session_state:
        st.session_state[key] = random.randrange(len(samples))
    add_label, divider = SUGGEST_TEXT[tab]
    if tab == "web":
        title, link = samples[st.session_state[key]]
        meta = f'{link.split("//", 1)[1].split("/", 1)[0]} · web article'
        help_text = (
            f"{title}: a research article published as a web page ({link}). Click Add to load it, "
            "or paste links to your own research paper pages below."
        )
    else:
        title, arxiv_id, pages = samples[st.session_state[key]]
        link = f"https://arxiv.org/pdf/{arxiv_id}"
        meta = f"arXiv {arxiv_id} · {pages} pages"
        help_text = (
            f"{title} (arXiv {arxiv_id}, {pages} pages). Click Add to load it into this chat, "
            f"or add your own research paper below (max {MAX_PAPER_PAGES} pages)."
        )

    with st.container(key=f"suggest_card_{tab}"):
        st.markdown(
            f'<div class="sl-suggest-label">{icon("sparkle", 12)} No paper handy? Try this one</div>'
            f'<div class="sl-suggest-title" title="{html.escape(help_text)}">{html.escape(title)}</div>'
            f'<div class="sl-suggest-meta">{html.escape(meta)} · '
            f'<a href="{link}" target="_blank" rel="noopener">{icon("download" if tab != "web" else "link", 11)} '
            f'{"PDF" if tab != "web" else "Open"}</a></div>',
            unsafe_allow_html=True,
        )
        c_add, c_next = st.columns([3, 1], gap="small")
        if c_add.button(add_label, key=f"suggest_add_{tab}", icon=":material/add:",
                        use_container_width=True, help=help_text):
            if tab == "web":
                queue_load("url", [link])
            else:
                queue_load("arxiv", [arxiv_id])
        if c_next.button("", key=f"suggest_next_{tab}", icon=":material/shuffle:", use_container_width=True,
                         help="Suggest a different paper"):
            st.session_state[key] = (st.session_state[key] + 1) % len(samples)
            st.rerun()
    st.markdown(f'<div class="sl-or">{divider.format(pages=MAX_PAPER_PAGES)}</div>', unsafe_allow_html=True)


def answer_meta(state: dict) -> dict:
    """Small summary of a finished turn for the badges under an answer."""
    docs = state.get("retrieved_docs") or []
    sources = []
    for d in docs:
        md = d.get("metadata", {}) if isinstance(d, dict) else getattr(d, "metadata", {})
        title = md.get("title") or md.get("source") or md.get("url")
        if title and title not in sources:
            sources.append(str(title))
    return {"route": state.get("route"), "sources": sources[:5]}


def render_meta(meta: dict | None) -> None:
    if not meta:
        return
    badges = []
    if meta.get("route") in ROUTE_BADGES:
        icon_name, label = ROUTE_BADGES[meta["route"]]
        badges.append(f'<span class="sl-badge sl-route">{icon(icon_name, 12)} {label}</span>')
    for src in meta.get("sources", []):
        badges.append(f'<span class="sl-badge">{icon("link", 12)} {html.escape(src[:48])}</span>')
    if badges:
        st.markdown(f'<div class="sl-meta">{"".join(badges)}</div>', unsafe_allow_html=True)


def generate_session_name(first_message: str) -> str:
    try:
        response = _rename_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Generate a concise 3-5 word title for a research chat session "
                        "based on the user's first message. Return only the title, "
                        "no punctuation at the end, no quotes."
                    ),
                },
                {"role": "user", "content": first_message[:500]},
            ]
        )
        return response.text.strip()
    except Exception:
        return "New Session"


def maybe_rename_session(session_id: str, first_message: str) -> None:
    if st.session_state.sessions_meta.get(session_id, {}).get("is_named"):
        return
    name = generate_session_name(first_message)
    st.session_state.sessions_meta[session_id]["name"] = name
    st.session_state.sessions_meta[session_id]["is_named"] = True
    save_sessions(st.session_state.sessions_meta)


def create_session() -> str:
    sid = str(uuid.uuid4())
    st.session_state.sessions_meta[sid] = {
        "id": sid,
        "name": "New Session",
        "created_at": datetime.now().isoformat(),
        "is_named": False,
    }
    save_sessions(st.session_state.sessions_meta)
    st.session_state.chats[sid] = []
    st.session_state.turns[sid] = 0
    return sid


def load_session_chats(session_id: str) -> list[dict]:
    """Rebuild the visible chat (user questions + final answers) from the checkpoint.

    The graph also stores internal steps (tool calls, tool results, rewritten
    queries, agent notes); those are skipped. New turns tag the final answer
    with name="answer" and rewrites with name="query_rewrite"; older sessions
    fall back to "last non-empty, non-tool AI message of each turn".
    """
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        if not state or not state.values:
            return []
        messages = state.values.get("messages", [])
        tagged = any(getattr(m, "name", None) == "answer" for m in messages)
        chats: list[dict] = []
        turn = 0
        pending_answer = None

        def flush():
            nonlocal pending_answer, turn
            if pending_answer:
                turn += 1
                chats.append({"role": "assistant", "content": pending_answer, "turn": turn, "graph_state": {}})
            pending_answer = None

        for msg in messages:
            kind = type(msg).__name__
            name = getattr(msg, "name", None)
            text = msg.text if hasattr(msg, "text") else str(msg.content)
            if kind == "HumanMessage":
                if name == "query_rewrite":
                    continue
                flush()
                chats.append({"role": "user", "content": text})
            elif kind in ("AIMessage", "AIMessageChunk"):
                if getattr(msg, "tool_calls", None) or not text.strip():
                    continue
                if tagged and name != "answer":
                    continue
                pending_answer = text
        flush()
        return chats
    except Exception:
        return []


def delete_session(session_id: str) -> None:
    """Remove a session everywhere: chat history, vector store, session list, UI state."""
    try:
        graph.checkpointer.delete_thread(session_id)
    except Exception:
        pass
    try:
        delete_session_collections(session_id)
    except Exception:
        pass
    st.session_state.sessions_meta.pop(session_id, None)
    save_sessions(st.session_state.sessions_meta)
    st.session_state.chats.pop(session_id, None)
    st.session_state.turns.pop(session_id, None)
    st.session_state.pop(f"processed_files_{session_id}", None)
    shutil.rmtree(UPLOADS_DIR / session_id, ignore_errors=True)
    cached_list_papers.clear()

    if st.session_state.active_session_id == session_id:
        remaining = sorted(
            st.session_state.sessions_meta.values(), key=lambda s: s["created_at"], reverse=True
        )
        if remaining:
            switch_session(remaining[0]["id"])
        else:
            st.session_state.active_session_id = create_session()


def switch_session(session_id: str) -> None:
    st.session_state.active_session_id = session_id
    if session_id not in st.session_state.chats:
        st.session_state.chats[session_id] = load_session_chats(session_id)
    if session_id not in st.session_state.turns:
        turn_count = sum(1 for m in st.session_state.chats[session_id] if m["role"] == "assistant")
        st.session_state.turns[session_id] = turn_count


graph = get_graph()

theme = current_theme()
if apply_streamlit_theme(theme) and not st.session_state.get("_theme_synced"):
    # Theme config only reaches the browser on the next run.
    st.session_state._theme_synced = True
    st.rerun()
st.session_state._theme_synced = False
st.markdown(build_css(theme), unsafe_allow_html=True)

# ── Bootstrap ──────────────────────────────────────────────────────────────────
if "sessions_meta" not in st.session_state:
    st.session_state.sessions_meta = load_sessions()
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "turns" not in st.session_state:
    st.session_state.turns = {}
if "active_session_id" not in st.session_state:
    if st.session_state.sessions_meta:
        latest = max(
            st.session_state.sessions_meta.values(),
            key=lambda s: s["created_at"],
        )
        switch_session(latest["id"])
    else:
        sid = create_session()
        st.session_state.active_session_id = sid

active_sid = st.session_state.active_session_id
process_pending_load()
if "pending_load" not in st.session_state:
    maybe_show_welcome(active_sid)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""
        <div class="sl-brand">
          <div class="sl-logo">{icon("book", 17)}</div>
          <div>
            <div class="sl-brand-name">ScholarLens</div>
            <div class="sl-brand-tag">Research workspace</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("New chat", key="new_chat", icon=":material/edit_square:", use_container_width=True):
        new_sid = create_session()
        st.session_state.active_session_id = new_sid
        active_sid = new_sid
        st.rerun()

    st.markdown('<div class="sl-side-title">Recent chats</div>', unsafe_allow_html=True)
    sorted_sessions = sorted(
        st.session_state.sessions_meta.values(),
        key=lambda s: s["created_at"],
        reverse=True,
    )
    pending_delete = st.session_state.get("pending_delete")
    with st.container(height=min(280, 46 * max(len(sorted_sessions), 1) + 10), border=False, key="chat_list"):
        for session in sorted_sessions:
            sid = session["id"]
            is_active = sid == st.session_state.active_session_id
            if pending_delete == sid:
                st.markdown(
                    f'<div class="sl-confirm">Delete <b>{html.escape(session["name"])}</b>? '
                    "This removes its messages and documents.</div>",
                    unsafe_allow_html=True,
                )
                c_yes, c_no = st.columns(2)
                if c_yes.button("Delete", key=f"del_yes_{sid}", icon=":material/delete:", use_container_width=True):
                    with st.spinner("Deleting chat…"):
                        delete_session(sid)
                    st.session_state.pending_delete = None
                    st.rerun()
                if c_no.button("Cancel", key=f"del_no_{sid}", use_container_width=True):
                    st.session_state.pending_delete = None
                    st.rerun()
                continue
            c_name, c_del = st.columns([6, 1], gap="small", vertical_alignment="center")
            if c_name.button(
                session["name"],
                key=f"sess_{sid}",
                icon=":material/chat_bubble:" if is_active else ":material/chat_bubble_outline:",
                use_container_width=True,
                type="secondary",
                disabled=is_active,
            ):
                with st.spinner("Opening chat…"):
                    switch_session(sid)
                st.rerun()
            if c_del.button("", key=f"trash_{sid}", icon=":material/delete:", help="Delete chat"):
                st.session_state.pending_delete = sid
                st.rerun()

    st.markdown('<div class="sl-side-title">Add sources</div>', unsafe_allow_html=True)
    if st.button(
        f"Research papers only · max {MAX_PAPER_PAGES} pages",
        key="scope_info",
        icon=":material/warning:",
        use_container_width=True,
        help="What you can upload, and sample papers to try",
    ):
        scope_dialog(active_sid)
    # Load failures are stored before st.rerun() so they survive the rerun.
    for err in st.session_state.pop("source_errors", []):
        st.error(err)
    tab_files, tab_web, tab_arxiv = st.tabs(
        [":material/upload_file: Files", ":material/language: Web", ":material/school: ArXiv"]
    )

    # ── Files ──────────────────────────────────────────────────────────────────
    with tab_files:
        suggestion_card("file")
        uploaded_files = st.file_uploader(
            "PDF, TXT, or Markdown",
            type=["pdf", "txt", "md", "markdown"],
            accept_multiple_files=True,
            key=f"uploader_{active_sid}",
            label_visibility="collapsed",
        )
        if st.button("Add files", use_container_width=True, key="btn_add_files",
                     help=f"Upload academic research papers (PDF, TXT or Markdown), up to {MAX_PAPER_PAGES} pages each."):
            if uploaded_files:
                done = st.session_state.get(f"processed_files_{active_sid}", set())
                new = [(f.name, f.getvalue()) for f in uploaded_files if f.name not in done]
                if new:
                    queue_load("file", new)
                st.info("These files are already loaded.")
            else:
                st.warning("No files selected.")

    # ── Web pages ──────────────────────────────────────────────────────────────
    with tab_web:
        suggestion_card("web")
        url_input = st.text_area(
            "URLs (one per line)",
            key=f"url_area_{active_sid}",
            height=90,
            label_visibility="collapsed",
            placeholder="https://example.com/paper\none URL per line",
        )
        if st.button("Load URLs", use_container_width=True, key="btn_load_urls"):
            urls = [u.strip() for u in url_input.splitlines() if u.strip()]
            if urls:
                queue_load("url", urls)
            else:
                st.warning("Enter at least one URL.")

    # ── ArXiv ──────────────────────────────────────────────────────────────────
    with tab_arxiv:
        suggestion_card("arxiv")
        arxiv_title = st.text_input(
            "Paper title or ArXiv ID",
            key=f"arxiv_input_{active_sid}",
            label_visibility="collapsed",
            placeholder="1706.03762 or Attention Is All You Need",
        )
        if st.button("Load paper", use_container_width=True, key="btn_load_arxiv"):
            if arxiv_title.strip():
                queue_load("arxiv", [arxiv_title.strip()])
            else:
                st.warning("Enter a paper title or ArXiv ID.")

    # ── Loaded documents ───────────────────────────────────────────────────────
    try:
        doc_titles = cached_list_papers(active_sid)
    except Exception:
        doc_titles = None
    count = len(doc_titles) if doc_titles else 0
    st.markdown(
        f'<div class="sl-side-title">Documents<span class="sl-count">{count}</span></div>',
        unsafe_allow_html=True,
    )
    if doc_titles is None:
        st.caption("Could not load document list — try refreshing.")
    elif doc_titles:
        sources = load_sources(active_sid)
        for i, t in enumerate(doc_titles):
            src = sources.get(t, {})
            doc_icon = (
                ":material/language:" if src.get("kind") == "web"
                else ":material/picture_as_pdf:" if str(src.get("path") or "").lower().endswith(".pdf")
                else ":material/description:"
            )
            if st.session_state.get("pending_doc_delete") == t:
                st.markdown(
                    f'<div class="sl-confirm">Remove <b>{html.escape(t)}</b>? Its chunks and saved file '
                    "will be deleted from this chat.</div>",
                    unsafe_allow_html=True,
                )
                c_yes, c_no = st.columns(2)
                if c_yes.button("Delete", key=f"del_yes_doc_{i}", icon=":material/delete:", use_container_width=True):
                    with st.spinner("Removing document…"):
                        try:
                            remove_document(active_sid, t)
                            st.toast(f"Removed {t}", icon=":material/delete:")
                        except Exception as e:
                            st.session_state.setdefault("source_errors", []).append(f"Could not remove {t[:60]} — {e}")
                    st.session_state.pending_doc_delete = None
                    st.rerun()
                if c_no.button("Cancel", key=f"del_no_doc_{i}", use_container_width=True):
                    st.session_state.pending_doc_delete = None
                    st.rerun()
                continue
            c_doc, c_del = st.columns([6, 1], gap="small", vertical_alignment="center")
            if c_doc.button(t, key=f"doc_{i}", icon=doc_icon, use_container_width=True, help="Open document"):
                st.session_state.viewing_doc = {"session_id": active_sid, "title": t}
                st.rerun()
            if c_del.button("", key=f"trash_doc_{i}", icon=":material/delete:", help="Remove document"):
                st.session_state.pending_doc_delete = t
                st.rerun()
    else:
        st.caption("No documents yet. Add files, web pages, or ArXiv papers above.")

    st.markdown(
        '<div class="sl-side-title">Tip</div>'
        '<div class="sl-tip">Start a message with <code>/btw</code> for a quick side question '
        "that isn't saved to the chat.</div>",
        unsafe_allow_html=True,
    )

# ── Theme toggle (pinned to the page's top-right corner via CSS) ──────────────
is_dark = theme == "dark"
if st.button(
    "Light" if is_dark else "Dark",
    key="theme_toggle",
    icon=":material/light_mode:" if is_dark else ":material/dark_mode:",
    help="Switch to light mode" if is_dark else "Switch to dark mode",
):
    st.session_state.theme = "light" if is_dark else "dark"
    apply_streamlit_theme(st.session_state.theme)
    st.rerun()

# ── Document viewer (replaces the chat until closed) ───────────────────────────
viewing = st.session_state.get("viewing_doc")
if viewing and viewing["session_id"] != active_sid:
    close_document()
    viewing = None
if viewing:
    render_document_viewer(active_sid, viewing["title"])
    if st.session_state.pop("_rerun_for_load", False):
        st.rerun()
    st.stop()

# ── Chat input (read first so example buttons and typed messages share one path) ─
typed_prompt = st.chat_input("Ask about your papers, verify a claim, or search the web…")
prompt = typed_prompt or st.session_state.pop("pending_prompt", None)

# ── Chat display ───────────────────────────────────────────────────────────────
history = st.session_state.chats.get(active_sid, [])
# Welcome screen lives in one slot: when a message is sent it is cleared right away
# instead of lingering (Streamlit keeps the previous run's elements until a run ends).
welcome_slot = st.empty()
if not history and not prompt:
    with welcome_slot.container():
        st.markdown(
            f"""
            <div class="sl-welcome">
              <div class="sl-mark">{icon("book", 24)}</div>
              <span class="sl-pill">{icon("file", 12)} Research workspace</span>
              <h1>What would you like to <span class="sl-grad-text">research</span> today?</h1>
              <p>Ask questions about your papers, verify claims against recent literature,
              or survey the latest work. Every answer is grounded in its sources.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(scope_notice(), unsafe_allow_html=True)
        cols = st.columns(2)
        for i, (ex_icon, title, subtitle, example) in enumerate(EXAMPLE_PROMPTS):
            if cols[i % 2].button(
                f"**{title}** · {subtitle}",
                key=f"example_{i}",
                icon=ex_icon,
                use_container_width=True,
                help=example,
            ):
                st.session_state.pending_prompt = example
                st.rerun()
else:
    welcome_slot.empty()

for msg in history:
    avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "user":
            st.markdown('<span class="sl-user"></span>', unsafe_allow_html=True)
        st.markdown(md_text(msg["content"]))
        if msg["role"] == "assistant":
            render_meta(msg.get("meta"))
            if msg.get("graph_state"):
                with st.expander("Details", expanded=False, icon=":material/data_object:"):
                    st.json(msg["graph_state"])

# ── Handle a new message ───────────────────────────────────────────────────────
if prompt:
    is_btw = prompt.strip().lower().startswith("/btw")

    if is_btw:
        query = prompt.strip()[4:].strip()

        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown('<span class="sl-user"></span>', unsafe_allow_html=True)
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            if not query:
                st.markdown("Please add a question after `/btw`, e.g. `/btw What is attention?`")
            else:
                placeholder = st.empty()
                placeholder.markdown("_Thinking…_")
                response_text = ""
                for chunk in handle_btw(query):
                    response_text += chunk
                    placeholder.markdown(md_text(response_text) + "▌")
                placeholder.markdown(md_text(response_text))
            st.caption(":material/forum: Side question — not saved to this chat.")

    else:
        if active_sid not in st.session_state.chats:
            st.session_state.chats[active_sid] = []
        if active_sid not in st.session_state.turns:
            st.session_state.turns[active_sid] = 0

        is_first_message = len(st.session_state.chats[active_sid]) == 0

        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown('<span class="sl-user"></span>', unsafe_allow_html=True)
            st.markdown(prompt)
        st.session_state.chats[active_sid].append({"role": "user", "content": prompt})
        st.session_state.turns[active_sid] += 1
        current_turn = st.session_state.turns[active_sid]

        if is_first_message:
            maybe_rename_session(active_sid, prompt)

        input_state = {
            "messages": [HumanMessage(content=prompt)],
            "session_id": active_sid,
            "query": prompt,
            "route": None,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "claim_verdict": None,
            "claim_source": None,
            "superseding_papers": [],
            "answer": None,
            "is_relevant": None,
            "rewrite_count": 0,
        }
        config = {"configurable": {"thread_id": active_sid}}

        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            status = st.status("Thinking…", expanded=False)
            placeholder = st.empty()
            response_text = ""
            current_node = None
            state_snapshot, meta = None, None

            # Without this, a failing API call leaves the question in the chat with
            # no answer, and Streamlit's error box vanishes on the next rerun.
            try:
                for chunk, metadata in graph.stream(input_state, config, stream_mode="messages"):
                    node = metadata.get("langgraph_node")
                    if node != current_node and node in NODE_STATUS and not response_text:
                        current_node = node
                        status.update(label=NODE_STATUS[node])
                    if (
                        node == "generate_answer"
                        and hasattr(chunk, "text")
                        and chunk.text
                    ):
                        if not response_text:
                            status.update(label="Done", state="complete")
                        response_text += chunk.text
                        placeholder.markdown(md_text(response_text) + "▌")

                status.update(label="Done", state="complete")

                if not response_text:
                    final_values = graph.get_state(config).values
                    response_text = final_values.get("answer") or "No response generated."

                placeholder.markdown(md_text(response_text))

                final_values = graph.get_state(config).values
                state_snapshot = _serialize_state(final_values)
                meta = answer_meta(state_snapshot)
                render_meta(meta)

                with st.expander("Details", expanded=False, icon=":material/data_object:"):
                    st.json(state_snapshot)
            except Exception as e:
                traceback.print_exc()
                status.update(label="Failed", state="error")
                response_text = (
                    f"⚠️ Couldn't generate a response: {str(e)[:300]}\n\n"
                    "Please send your message again."
                )
                placeholder.markdown(md_text(response_text))

        st.session_state.chats[active_sid].append(
            {
                "role": "assistant",
                "content": response_text,
                "graph_state": state_snapshot,
                "meta": meta,
                "turn": current_turn,
            }
        )

        if is_first_message:
            st.rerun()

# A source was queued this run: rerun now that the page (without the dialog) is drawn.
if st.session_state.pop("_rerun_for_load", False):
    st.rerun()
