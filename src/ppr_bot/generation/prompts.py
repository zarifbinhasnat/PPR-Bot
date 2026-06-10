"""All LLM prompt templates live here.

Centralizing prompts in one module makes them easy to find, version, and
tweak without hunting through pipeline code — and lets multiple stages
(ingestion, enrichment, retrieval, generation) share fragments if needed.
"""

# Used by ingestion/ocr_transcriber.py (Milestone 2).
#
# The source PDF mixes a legacy non-Unicode Bangla font (which produces
# mojibake if read directly), proper Unicode Bangla fonts, scanned
# image-only pages, and English text. Asking a vision model to "read and
# transcribe" the rendered page sidesteps all of that: it works the same
# way regardless of the underlying font or whether the page is a scan.
OCR_TRANSCRIPTION_PROMPT = """\
You are transcribing a single page from the Bangladesh Gazette \
(Public Procurement Rules, 2025) into clean Markdown.

Rules:
1. Transcribe ALL text on the page exactly as written, in proper Unicode \
Bangla (বাংলা) and/or English. Never output garbled/mojibake characters.
2. Preserve the document's hierarchical structure using Markdown headings:
   - "# " for a Part (অংশ/Part) heading
   - "## " for a Chapter (অধ্যায়/Chapter) heading
   - "### " for a Rule (বিধি/Rule) number and title
   - "#### " for a Schedule/Tofshil (তফসিল/Schedule) heading
   Sub-rules, clauses, and sub-clauses (e.g. "(১)", "(ক)", "(i)") stay as \
body text/lists, not as headings.
3. Render tables using Markdown table syntax (| col | col |), preserving \
all cell text.
4. If the page contains a flowchart, diagram, or organizational chart, \
describe it as a nested Markdown list capturing each box/step and the \
flow between them (top-to-bottom, noting branches like "If Yes -> ..." / \
"If No -> ...").
5. If the page has no meaningful transcribable content (e.g. blank or pure \
decoration), output exactly: <!-- no transcribable content -->
6. At the very top of your output, if a printed Gazette page number is \
visible (e.g. a Bangla numeral like ৯৭৮৩ in the header), add it as: \
<!-- gazette_page: ৯৭৮৩ -->. Omit this line if no page number is visible.
7. Output ONLY the Markdown transcription — no commentary or explanation.
"""


# Used by enrichment/contextualizer.py (Milestone 5).
#
# This implements Anthropic's "Contextual Retrieval" idea: before indexing a
# chunk, ask an LLM to write a short blurb that situates the chunk within the
# whole document. That blurb is prepended to the chunk so both the embedding
# and the keyword (BM25) index capture context the raw chunk lacks (e.g. that
# "(৩)" belongs to "Rule 42 on direct procurement"). This measurably reduces
# retrieval failures where a chunk is relevant but doesn't lexically match.
CONTEXTUALIZE_CHUNK_PROMPT = """\
You are indexing a Bangladeshi legal document: the Public Procurement Rules \
(PPR) 2025.

Here is the section the chunk comes from (its hierarchy):
{breadcrumb}

Here is the chunk:
<chunk>
{chunk_text}
</chunk>

Write a SHORT (1-2 sentences, 50-100 tokens) context blurb that situates \
this chunk within the document, so a search system can find it. Mention the \
relevant Rule/Schedule number and the topic it addresses. Write the blurb in \
the SAME language as the chunk (Bangla if the chunk is Bangla, English if \
English). Output ONLY the blurb — no preamble, no quotes."""


# Used by retrieval/query_transform.py (Milestone 8).
#
# In a conversation, follow-up questions are often incomplete ("what about for
# goods?"). Embedding/BM25 search needs a self-contained query, so we ask a
# cheap LLM call to rewrite the latest question into a standalone one using the
# prior turns for context.
CONDENSE_QUERY_PROMPT = """\
Given the conversation so far and a follow-up question, rewrite the follow-up \
into a STANDALONE question that can be understood without the conversation. \
Resolve pronouns and implied references. Keep it in the same language as the \
follow-up. If the follow-up is already standalone, return it unchanged. \
Output ONLY the standalone question.

Conversation:
{history}

Follow-up question: {question}

Standalone question:"""


# Used by generation/answer_generator.py (Milestone 9).
#
# The system instruction that grounds the chatbot: answer ONLY from the
# retrieved context, cite Rule/Schedule numbers and source pages, and admit
# when the answer isn't in the provided context (anti-hallucination).
RAG_SYSTEM_PROMPT = """\
You are an expert assistant on the Public Procurement Rules (PPR) 2025 of \
Bangladesh. You answer questions using ONLY the provided context excerpts \
from the official Gazette.

Rules:
- Answer in the same language as the user's question (Bangla or English).
- Base your answer strictly on the provided context. Do NOT use outside \
knowledge or invent rule numbers.
- Cite the relevant Rule/Schedule (বিধি/তফসিল) and source page number(s) \
from the context when you state a fact.
- If the provided context does not contain the answer, say so clearly \
(e.g. "প্রদত্ত অংশে এই তথ্য পাওয়া যায়নি" / "The provided context does not \
cover this") instead of guessing.
- Be concise and precise, as befits a legal reference."""
