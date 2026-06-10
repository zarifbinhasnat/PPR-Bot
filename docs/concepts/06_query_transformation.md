# 06 · Query transformation (conversational retrieval)

## The follow-up problem

Retrieval works on the *current* message. But in a conversation, messages lean
on earlier ones:

```
User: What is the threshold for direct procurement of goods?
Bot:  Under Rule 76 … 10 lakh taka …
User: What about for works?      ← meaningless to a search engine alone
```

If you embed "What about for works?" and search, you retrieve noise — there's
no mention of procurement or thresholds in it.

## The fix: condense to a standalone query

Before retrieval, make a cheap LLM call that rewrites the follow-up into a
self-contained question using the recent turns:

```
"What is the threshold for direct procurement of works?"
```

Now retrieval has everything it needs.

## In this codebase

- `retrieval/query_transform.py` — `condense_query(history, question, …)`.
  - On the **first** turn (no history) it skips the LLM call and returns the
    question unchanged — saving latency and a request.
  - If the rewrite call fails, it **degrades gracefully** to the raw question.
- `chat/orchestrator.py` calls this as step 2 of every turn, before retrieval.

## Related techniques (not in v1)

- **Query expansion**: generate paraphrases and search all of them, fusing
  results. More recall, more cost. (A `expand_query` hook is described in the
  plan as optional/off-by-default.)
- **HyDE** (Hypothetical Document Embeddings): embed a *hypothetical answer*
  instead of the question. See `docs/future_extensions.md`.
- **Step-back prompting**: ask a more general question first. Future work.
