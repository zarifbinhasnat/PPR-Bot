"""Step 9: generate the grounded answer from retrieved chunks (streaming).

This is the "G" in RAG. We take the top reranked chunks, format them into a
context block (with their breadcrumb + source pages so the model can cite),
and ask Gemini to answer using ONLY that context. The system prompt
(RAG_SYSTEM_PROMPT) enforces grounding and citation and tells the model to
admit when the answer isn't present — the main defence against hallucination.

We stream the answer token-by-token so the UI can render it as it arrives,
which is what makes a chat feel responsive.
"""

import json
import re
import sys
from collections.abc import Iterator

from google import genai
from google.genai import types

from ppr_bot.generation.prompts import RAG_SYSTEM_PROMPT, SUGGEST_FOLLOWUPS_PROMPT


def format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks into a numbered, citable context block."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        breadcrumb = meta.get("breadcrumb", "")
        pages = meta.get("source_pages", [])
        page_str = ", ".join(str(p) for p in pages)
        blocks.append(
            f"[Excerpt {i} | {breadcrumb} | page(s): {page_str}]\n{chunk['text']}"
        )
    return "\n\n".join(blocks)


def build_prompt(query: str, chunks: list[dict]) -> str:
    """Assemble the user-turn prompt: the context block + the question."""
    context = format_context(chunks)
    return (
        f"Context excerpts from PPR 2025:\n\n{context}\n\n"
        f"---\n\nQuestion: {query}\n\n"
        f"Answer using only the context above, citing Rule/Schedule and "
        f"page numbers."
    )


def generate_answer_stream(
    query: str,
    chunks: list[dict],
    client: genai.Client,
    model: str,
) -> Iterator[str]:
    """Yield the answer text incrementally as Gemini generates it.

    The system instruction is passed via GenerateContentConfig; the context +
    question go in `contents`. We yield each streamed chunk's `.text`.
    """
    prompt = build_prompt(query, chunks)
    config = types.GenerateContentConfig(system_instruction=RAG_SYSTEM_PROMPT)

    stream = client.models.generate_content_stream(
        model=model, contents=prompt, config=config
    )
    for event in stream:
        if event.text:
            yield event.text


def _parse_suggestions(raw: str) -> list[str]:
    """Pull a JSON array of strings out of a model response.

    Models sometimes wrap JSON in ```json fences or add stray prose, so we
    locate the first '[' ... ']' span and parse that, then keep only short
    non-empty strings. Returns [] if nothing usable is found.
    """
    if not raw:
        return []
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except (ValueError, TypeError):
        return []
    out: list[str] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
            if text and len(text) <= 80:
                out.append(text)
    return out[:3]


def generate_suggestions(
    question: str,
    answer: str,
    client: genai.Client,
    model: str,
) -> list[str]:
    """Return up to 3 suggested follow-up questions (empty list on failure).

    This powers the suggestion chips in the chat UI. It's best-effort: a
    failure here (including a daily-quota 429) must never break a turn, so we
    swallow errors and just return no suggestions — the UI then shows none.
    """
    prompt = SUGGEST_FOLLOWUPS_PROMPT.format(question=question, answer=answer)
    try:
        response = client.models.generate_content(model=model, contents=prompt)
        return _parse_suggestions(response.text or "")
    except Exception as exc:  # non-fatal — suggestions are a nice-to-have
        print(f"[warn] follow-up suggestion generation failed: {exc}", file=sys.stderr)
        return []
