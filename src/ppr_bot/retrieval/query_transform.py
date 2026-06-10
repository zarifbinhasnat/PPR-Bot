"""Step 8: query transformation for conversational retrieval.

In a chat, people ask follow-ups that only make sense in context:

    User: What is the threshold for direct procurement of goods?
    Bot:  ... (answers) ...
    User: What about for works?     <-- ambiguous on its own

If we embed "What about for works?" and search, we get garbage — the query
has no mention of procurement thresholds. So before retrieval we make a cheap
LLM call that rewrites the follow-up into a STANDALONE query using the prior
turns:

    "What is the threshold for direct procurement of works?"

This "condense question" step is what makes multi-turn RAG actually work.
"""

from google import genai

from ppr_bot.generation.prompts import CONDENSE_QUERY_PROMPT


def _format_history(history: list[dict]) -> str:
    """Render recent turns as plain text for the condense prompt."""
    lines = []
    for turn in history:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def condense_query(
    history: list[dict],
    question: str,
    client: genai.Client,
    model: str,
) -> str:
    """Rewrite `question` into a standalone query given conversation `history`.

    With no history (first turn) the question is already standalone, so we
    skip the LLM call entirely — saving latency and a request.
    """
    if not history:
        return question

    prompt = CONDENSE_QUERY_PROMPT.format(
        history=_format_history(history), question=question
    )
    try:
        response = client.models.generate_content(model=model, contents=prompt)
        rewritten = (response.text or "").strip()
        return rewritten or question  # fall back to original if empty
    except Exception:
        # If the rewrite call fails, degrade gracefully to the raw question.
        return question
