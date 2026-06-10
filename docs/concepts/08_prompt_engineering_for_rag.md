# 08 · Prompt engineering for RAG

The generation step is where retrieval pays off — or where it all falls apart
via hallucination. The prompts live in `generation/prompts.py`.

## The grounding system prompt (`RAG_SYSTEM_PROMPT`)

Key instructions, and why each matters:

- **"Answer using ONLY the provided context."** Stops the model from answering
  from its pre-trained memory (which may be wrong for *this* document or
  outdated).
- **"Cite the relevant Rule/Schedule and source page."** Makes answers
  verifiable and builds user trust. The page numbers come from chunk metadata.
- **"If the context doesn't contain the answer, say so."** This single line is
  the most effective anti-hallucination measure — it gives the model
  permission to say "not found" instead of inventing a rule.
- **"Answer in the same language as the question."** The corpus is bilingual;
  users may ask in Bangla or English.

## Building the context block (`answer_generator.format_context`)

Each retrieved chunk is rendered as a numbered, labelled excerpt:

```
[Excerpt 1 | প্রথম অধ্যায় > ২। সংজ্ঞার্থ | page(s): 1]
<chunk text>
```

The breadcrumb + page label is what lets the model cite precisely.

## Streaming generation

`generate_answer_stream()` uses Gemini's `generate_content_stream` and yields
each delta's text. The system instruction is passed via
`GenerateContentConfig(system_instruction=...)`, separate from the user-turn
content (context + question). Streaming is what makes the chat feel alive (see
`concepts/07` for how those tokens reach the browser).

## Things to tune

- How many chunks to include (`TOP_K_RERANK` in config) — more recall vs. more
  noise/cost.
- Whether to include the `contextual_summary` in the displayed context.
- Citation format strictness.
