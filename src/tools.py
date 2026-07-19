"""
Tool schema definitions and execution logic.

All tool JSON Schema definitions are centrally managed here,
along with search/open execution (Serper API + URL fetching).
"""
from __future__ import annotations

import io
import itertools
import logging
import re
import textwrap

import html2text
import lxml.html
import lxml.etree
import pypdf
import requests

from src.config import SERPER_API_KEY, MAX_PAGE_CHARS_SEP

logger = logging.getLogger(__name__)

# ── Memory Tools ─────────────────────────────────────────

MANAGE_CONTEXT_TOOL = {
    "type": "function",
    "function": {
        "name": "manage_context",
        "description": """\
Compress your working memory. Call this when context is filling up with dead ends, duplicates, or detail you no longer need verbatim.

The system automatically picks the range to compress: everything since your last manage_context call (or since the start of the investigation if this is your first call) up to (but not including) the message that issued this tool call. The system prompt and the original question are always preserved. The original messages in the compressed range are saved to disk as summary_{summary_id}.json (retrievable later with query_memory(summary_id, query)), and a fresh summary — focused on paths explored, reasoning, and conclusions — is returned as the tool result.

The summary text is prefixed with "[summary_id: N]" so you can refer to it directly in subsequent reasoning and pull the raw content back via query_memory(summary_id=N, ...).

This tool takes no arguments.""",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

MEM_TOOLS = [MANAGE_CONTEXT_TOOL]

# re_mem_noquery ablation: same manage_context, but the archived messages are
# discarded (no query_memory) — so the description drops both query_memory
# references and states that compression is irreversible. Used by get_tools
# when disable_query_memory=True. The original MANAGE_CONTEXT_TOOL is untouched.
MANAGE_CONTEXT_TOOL_NOQUERY = {
    "type": "function",
    "function": {
        "name": "manage_context",
        "description": """\
Compress your working memory. Call this when context is filling up with dead ends, duplicates, or detail you no longer need verbatim.

The system automatically picks the range to compress: everything since your last manage_context call (or since the start of the investigation if this is your first call) up to (but not including) the message that issued this tool call. The system prompt and the original question are always preserved. The original messages in the compressed range are discarded and replaced by a fresh summary — focused on paths explored, reasoning, and conclusions — which is returned as the tool result. Compression is permanent: the original text cannot be recovered, so the summary must carry everything you may still need.

The summary text is prefixed with "[summary_id: N]" so you can refer to it directly in subsequent reasoning.

This tool takes no arguments.""",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

MEM_TOOLS_NOQUERY = [MANAGE_CONTEXT_TOOL_NOQUERY]

# ── Browse Tools ─────────────────────────────────────────
# 参考 OpenAI gpt-oss SimpleBrowserTool：search + open 两个独立 tool。

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web using Google. Returns top results with titles, snippets, and URLs. Use the open tool to read the full content of a result.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string",
                },
            },
            "required": ["query"],
        },
    },
}

OPEN_TOOL = {
    "type": "function",
    "function": {
        "name": "open",
        "description": "Open a URL and read the webpage content. Returns the page text with line numbers. Long pages are automatically truncated.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to open and read",
                },
            },
            "required": ["url"],
        },
    },
}

BROWSE_TOOLS = [SEARCH_TOOL, OPEN_TOOL]

# ── Memory Query Tool ────────────────────────────────────

QUERY_MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "query_memory",
        "description": """\
Retrieve detailed information from a previously compressed summary. Each manage_context call \
returns a summary prefixed with [summary_id: N] and saves the original messages to disk as \
summary_{N}.json. This tool loads summary_{summary_id}.json and uses an LLM to extract \
information matching your query from the original (uncompressed) content.

DO NOT CALL THIS TOOL until at least one manage_context call has produced a summary_id.""",
        "parameters": {
            "type": "object",
            "properties": {
                "summary_id": {
                    "type": "integer",
                    "description": "The summary_id (as shown in the [summary_id: N] prefix of a prior manage_context summary) whose original content should be searched.",
                },
                "query": {
                    "type": "string",
                    "description": "What specific information to extract from the original messages of that summary.",
                },
            },
            "required": ["summary_id", "query"],
        },
    },
}

FILE_TOOLS = [QUERY_MEMORY_TOOL]

# ── BrowseComp-Plus Tools ───────────────────────────────
# Descriptions from BrowseComp-Plus/searcher/searchers/base.py lines 57-67

BCP_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Perform a search on the local knowledge corpus. Returns the top 10 hits "
            "with docid, score, and a snippet of the document content. "
            "Snippet length adapts to whether you have seen the document before: "
            "(a) new docs return a 512-token preview; "
            "(b) docs you have seen in the current window return a 128-token preview "
            "and remind you to call get_document for the full text if needed; "
            "(c) docs you have seen earlier but that have since been compressed by "
            "manage_context return a 128-token preview and remind you to call "
            "query_memory to retrieve the relevant earlier summary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string"},
            },
            "required": ["query"],
        },
    },
}


# re_mem_noquery ablation: identical to BCP_SEARCH_TOOL but case (c) points to
# get_document (which still re-fetches from the corpus) instead of query_memory.
BCP_SEARCH_TOOL_NOQUERY = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Perform a search on the local knowledge corpus. Returns the top 10 hits "
            "with docid, score, and a snippet of the document content. "
            "Snippet length adapts to whether you have seen the document before: "
            "(a) new docs return a 512-token preview; "
            "(b) docs you have seen in the current window return a 128-token preview "
            "and remind you to call get_document for the full text if needed; "
            "(c) docs you have seen earlier but that have since been compressed by "
            "manage_context return a 128-token preview and remind you to call "
            "get_document for the full text if you need it again."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string"},
            },
            "required": ["query"],
        },
    },
}


BCP_GET_DOCUMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "get_document",
        "description": (
            "Retrieve a document by its docid. The returned text is capped at 8192 "
            "tokens; longer documents are truncated."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "docid": {"type": "string", "description": "Document ID to retrieve"},
            },
            "required": ["docid"],
        },
    },
}


BCP_BROWSE_TOOLS = [BCP_SEARCH_TOOL, BCP_GET_DOCUMENT_TOOL]


# ── Benchmark Tool Registry ─────────────────────────────

BENCHMARK_TOOLS: dict[str, list[dict]] = {
    "browsecomp": BROWSE_TOOLS,  # search + open (Serper)
    "browsecomp-plus": BCP_BROWSE_TOOLS,  # search + get_document (BM25)
}


def get_tools(benchmark: str, use_memory_tools: bool = False,
              disable_query_memory: bool = False,
              # Legacy aliases — kept so existing CLI flags still work
              use_mem_tool: bool = False, use_memory_tool: bool = False) -> list[dict]:
    """Return the tool set for the given benchmark.

    Args:
        benchmark: registered benchmark name
        use_memory_tools: include memory tools (manage_context + query_memory)
        disable_query_memory: re_mem_noquery ablation — keep manage_context but
            drop query_memory (and use the no-query manage_context / search
            descriptions). Only meaningful when memory tools are enabled.
        use_mem_tool / use_memory_tool: legacy aliases for use_memory_tools
    """
    enabled = use_memory_tools or use_mem_tool or use_memory_tool
    tools = BENCHMARK_TOOLS.get(benchmark)
    if tools is None:
        raise ValueError(
            f"Unknown benchmark: {benchmark}. Available: {list(BENCHMARK_TOOLS.keys())}"
        )
    tools = list(tools)
    if enabled and disable_query_memory:
        # Swap the BCP/DR9K search description (case (c) → get_document) and
        # drop query_memory (FILE_TOOLS) entirely.
        tools = [BCP_SEARCH_TOOL_NOQUERY if t is BCP_SEARCH_TOOL else t for t in tools]
        tools = MEM_TOOLS_NOQUERY + tools
    elif enabled:
        tools = MEM_TOOLS + tools + FILE_TOOLS
    return tools


def get_browse_tools() -> list[dict]:
    """Return browse tools without memory tools."""
    return list(BROWSE_TOOLS)


def get_all_tools() -> list[dict]:
    """Return full tool set (mem tools + browse + memory query)."""
    return MEM_TOOLS + list(BROWSE_TOOLS) + FILE_TOOLS


# ── Search / Open Execution ────────────────────────────────
# Serper API web search + URL fetching (previously in search_tool.py).

SERPER_URL = "https://google.serper.dev/search"

# open_url max chars (~1200-1500 tokens)
MAX_PAGE_CHARS = 5000


def execute_search(query: str, num_results: int = 50) -> str:
    """Call Serper API to execute a Google search.

    Args:
        query: search query
        num_results: number of results (default 50)

    Returns:
        Formatted search results text
    """
    if not SERPER_API_KEY:
        logger.error("SERPER_API_KEY not set")
        return "[SEARCH ERROR: SERPER_API_KEY not configured]"

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "q": query,
        "num": num_results,
    }

    try:
        resp = requests.post(SERPER_URL, headers=headers, json=payload, timeout=15)
        # Surface Serper's response body in the agent-visible error — the
        # generic HTTPError swallows the "Not enough credits" / "Invalid key"
        # message that the agent (and us, watching logs) need to see.
        if not resp.ok:
            body = (resp.text or "")[:400]
            logger.error("Serper API %d: %s", resp.status_code, body)
            return f"[SEARCH ERROR: HTTP {resp.status_code} from Serper — {body}]"
        data = resp.json()
    except requests.RequestException as e:
        logger.error("Serper API request failed: %s", e)
        return f"[SEARCH ERROR: {e}]"

    parts = []

    # Knowledge Graph
    if "knowledgeGraph" in data:
        kg = data["knowledgeGraph"]
        title = kg.get("title", "")
        desc = kg.get("description", "")
        if title:
            parts.append(f"[Knowledge Graph] {title}: {desc}")
        for key, val in kg.get("attributes", {}).items():
            parts.append(f"  - {key}: {val}")

    # Organic results
    organic = data.get("organic", [])
    for i, result in enumerate(organic[:num_results], 1):
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        link = result.get("link", "")
        entry = f"[{i}] {title}"
        if snippet:
            entry += f"\n    {snippet}"
        entry += f"\n    URL: {link}"
        parts.append(entry)

    if not parts:
        return "[SEARCH: No results found]"

    result = "\n\n".join(parts)
    result += "\n\n---\n Use open(url) to read the full content of any result above."
    return result


# ── Web page opening ───────────────────────────────────────

_EMPTY_LINE_RE = re.compile(r"^\s+$", flags=re.MULTILINE)
_EXTRA_NEWLINE_RE = re.compile(r"\n(\s*\n)+")
_SMP_RE = re.compile(r"[\U00010000-\U0001FFFF]", re.UNICODE)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _html_to_text(raw_html: str) -> tuple[str, str]:
    """Convert HTML to clean plain text.

    Returns:
        (text, title)
    """
    raw_html = _SMP_RE.sub("", raw_html)

    try:
        root = lxml.html.fromstring(raw_html)
    except Exception:
        return raw_html[:MAX_PAGE_CHARS], ""

    title_el = root.find(".//title")
    title = (title_el.text or "").strip() if title_el is not None else ""

    for tag_name in ("script", "style", "noscript", "nav", "header", "footer"):
        for el in root.findall(f".//{tag_name}"):
            el.getparent().remove(el)

    clean_html = lxml.etree.tostring(root, encoding="UTF-8").decode()

    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.body_width = 0
    h.ignore_tables = True
    h.unicode_snob = True
    h.ignore_emphasis = True
    text = h.handle(clean_html).strip()

    text = _EMPTY_LINE_RE.sub("", text)
    text = _EXTRA_NEWLINE_RE.sub("\n\n", text)

    return text, title


def _pdf_to_text(data: bytes) -> tuple[str, str]:
    """Extract plain text from PDF bytes.

    Returns:
        (text, title) — title from PDF metadata if present, else empty.
        ("", "") on parse failure → caller falls through to "no readable text".
    """
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
    except Exception as e:
        logger.warning("pypdf.PdfReader failed: %s", e)
        return "", ""

    title = ""
    try:
        meta = reader.metadata
        if meta and meta.title:
            title = str(meta.title).strip()
    except Exception:
        pass

    pages = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception as e:
            logger.warning("pypdf extract_text page %d failed: %s", i, e)
            continue
        t = t.strip()
        if t:
            pages.append(t)

    text = "\n\n".join(pages)
    text = _EMPTY_LINE_RE.sub("", text)
    text = _EXTRA_NEWLINE_RE.sub("\n\n", text)
    return text, title


def _wrap_lines(text: str, width: int = 80) -> list[str]:
    """Wrap text at *width* characters, preserving blank lines."""
    lines = text.split("\n")
    wrapped = itertools.chain.from_iterable(
        (
            textwrap.wrap(
                line, width=width,
                replace_whitespace=False, drop_whitespace=False,
            )
            if line
            else [""]
        )
        for line in lines
    )
    return list(wrapped)


def _join_lines_with_numbers(lines: list[str], offset: int = 0) -> str:
    """Add line numbers L0: L1: ..."""
    return "\n".join(f"L{i + offset}: {line}" for i, line in enumerate(lines))


def execute_open(url: str, max_chars: int = MAX_PAGE_CHARS) -> str:
    """Open a URL and extract the page body text.

    Args:
        url: webpage URL to open
        max_chars: max characters to return

    Returns:
        Page text with line numbers
    """
    max_retries = 2
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=20, allow_redirects=True)
            resp.raise_for_status()
            break
        except requests.exceptions.Timeout as e:
            logger.warning("Timeout fetching %s (attempt %d/%d): %s", url, attempt, max_retries, e)
            if attempt == max_retries:
                return f"[OPEN ERROR: Failed to fetch {url}: {e}]"
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status >= 500 and attempt < max_retries:
                logger.warning("Server error %d fetching %s (attempt %d/%d)", status, url, attempt, max_retries)
                continue
            logger.error("Failed to fetch URL %s: %s", url, e)
            return f"[OPEN ERROR: Failed to fetch {url}: {e}]"
        except requests.RequestException as e:
            logger.error("Failed to fetch URL %s: %s", url, e)
            return f"[OPEN ERROR: Failed to fetch {url}: {e}]"

    content_type = resp.headers.get("Content-Type", "")
    ct = content_type.lower()
    if "pdf" in ct:
        text, title = _pdf_to_text(resp.content)
    elif "text/html" in ct or "application/xhtml" in ct:
        text, title = _html_to_text(resp.text)
    elif "text/plain" in ct or "text/csv" in ct or "application/json" in ct:
        text, title = resp.text, ""
    else:
        return f"[OPEN ERROR: Unsupported content type: {content_type}]"

    if not text.strip():
        return f"[OPEN: Page at {url} returned no readable text]"

    if len(text) > max_chars:
        text = text[:max_chars]
        last_newline = text.rfind("\n")
        if last_newline > max_chars // 2:
            text = text[:last_newline]
        text += "\n\n[... page truncated ...]"

    lines = _wrap_lines(text)
    body = _join_lines_with_numbers(lines)

    header = f"Page: {title}\nURL: {url}\n---\n" if title else f"URL: {url}\n---\n"
    return header + body
