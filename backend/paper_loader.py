import re
import tempfile
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

import pymupdf
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)
_md_splitter = RecursiveCharacterTextSplitter.from_language(
    "markdown", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)


def _stamp_title(docs: list[Document], title: str) -> list[Document]:
    # Zero chunks means nothing would reach the vector store, so fail loudly
    # instead of letting the UI report a silent "Added".
    if not docs:
        raise ValueError(f"No text could be extracted from '{title}'.")
    for doc in docs:
        doc.metadata["title"] = title
    return docs


# Scanned PDFs have no text layer, so empty pages are OCR'd with Tesseract.
# Only installed language packs are used (e.g. Marathi needs tesseract-ocr-mar).
OCR_LANGUAGES = ("eng", "hin")
OCR_DPI = 300


@lru_cache(maxsize=1)
def _ocr_language() -> str | None:
    """Return a Tesseract language string like 'eng+mar', or None if Tesseract is missing."""
    tessdata = pymupdf.get_tessdata()
    if not tessdata:
        return None
    installed = [lang for lang in OCR_LANGUAGES if (Path(tessdata) / f"{lang}.traineddata").exists()]
    return "+".join(installed) or None


def _read_pdf(file_path: str) -> list[Document]:
    """Load PDF pages, OCR-ing any page that has no extractable text."""
    docs = PyMuPDFLoader(file_path).load()
    empty = [doc for doc in docs if not doc.page_content.strip()]
    language = _ocr_language()
    if empty and language:
        with pymupdf.open(file_path) as pdf:
            for doc in empty:
                page = pdf[doc.metadata["page"]]
                textpage = page.get_textpage_ocr(language=language, dpi=OCR_DPI, full=True)
                doc.page_content = page.get_text(textpage=textpage)
    return docs


def load_pdf(file_path: str) -> list[Document]:
    docs = _read_pdf(file_path)
    return _stamp_title(_splitter.split_documents(docs), Path(file_path).stem)


def load_text(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    return _stamp_title(_splitter.split_documents(docs), Path(file_path).stem)


def load_markdown(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    return _stamp_title(_md_splitter.split_documents(docs), Path(file_path).stem)


def load_webpage(url: str) -> list[Document]:
    docs = WebBaseLoader(url, requests_kwargs={"timeout": 30}).load()
    title = (docs[0].metadata.get("title") or url) if docs else url
    return _stamp_title(_splitter.split_documents(docs), title)


def _extract_arxiv_id(query: str) -> str | None:
    """Return bare ArXiv ID (no version suffix) if one appears in the query."""
    m = _ARXIV_ID_RE.search(query)
    if m:
        return re.sub(r"v\d+$", "", m.group(1))
    return None


def _arxiv_api_lookup(arxiv_id: str) -> str:
    """Fetch paper title by ID from the ArXiv Atom API."""
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        xml = resp.read().decode()
    titles = re.findall(r"<title>(.*?)</title>", xml, re.DOTALL)
    return titles[1].strip() if len(titles) > 1 else arxiv_id


def _arxiv_search(query: str) -> str:
    """Search ArXiv Atom API by title phrase and return the top result's bare paper ID."""
    phrase = query.strip('"')
    search_query = urllib.parse.quote(f'ti:"{phrase}"')
    url = f"https://export.arxiv.org/api/query?search_query={search_query}&max_results=1&sortBy=relevance"
    with urllib.request.urlopen(url, timeout=15) as resp:
        xml = resp.read().decode()
    m = re.search(r"<id>https?://arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)</id>", xml)
    if not m:
        raise ValueError(f"No ArXiv paper found for: {query}")
    return re.sub(r"v\d+$", "", m.group(1))


def _load_arxiv_by_id(arxiv_id: str) -> list[Document]:
    """Download and chunk an ArXiv paper PDF by its bare ID."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    with urllib.request.urlopen(pdf_url, timeout=60) as resp:
        pdf_bytes = resp.read()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        docs = _read_pdf(tmp_path)
        if not docs:
            raise ValueError(f"Could not load PDF for ArXiv ID: {arxiv_id}")
        title = (docs[0].metadata.get("title") or "").strip() or _arxiv_api_lookup(arxiv_id)
        chunks = _stamp_title(_splitter.split_documents(docs), title)
        for chunk in chunks:
            chunk.metadata["pdf_url"] = pdf_url
        return chunks
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def load_arxiv(query: str) -> list[Document]:
    arxiv_id = _extract_arxiv_id(query) or _arxiv_search(query)
    return _load_arxiv_by_id(arxiv_id)


def load_document(source: str) -> list[Document]:
    """Dispatch to the appropriate loader based on URL prefix or file extension."""
    if source.startswith(("http://", "https://")):
        return load_webpage(source)
    ext = Path(source).suffix.lower()
    if ext == ".pdf":
        return load_pdf(source)
    if ext == ".txt":
        return load_text(source)
    if ext in (".md", ".markdown"):
        return load_markdown(source)
    raise ValueError(f"Unsupported file type: {ext}")
