"""Rebrand Streamlit's static index.html (tab title + favicon shown before the app loads).

Streamlit serves this page first; app.py's st.set_page_config only takes effect once
the script runs, so without this the tab briefly shows "Streamlit" and its icon.
Re-run after reinstalling/upgrading streamlit (the Dockerfile does this on build).
"""

import re
import shutil
from pathlib import Path

import streamlit

TITLE = "ScholarLens"
FAVICON = Path(__file__).resolve().parent.parent / "assets" / "favicon.svg"

static_dir = Path(streamlit.__file__).parent / "static"
index = static_dir / "index.html"

shutil.copyfile(FAVICON, static_dir / "favicon.svg")

html = index.read_text(encoding="utf-8")
html = re.sub(r"<title>.*?</title>", f"<title>{TITLE}</title>", html, count=1)
html = re.sub(
    r'<link rel="shortcut icon" href="[^"]*" ?/?>',
    '<link rel="shortcut icon" type="image/svg+xml" href="./favicon.svg" />',
    html,
    count=1,
)
index.write_text(html, encoding="utf-8")
print(f"Branded {index}")
