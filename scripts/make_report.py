"""Generate the project report PDF (font size 14, structured for a teaching
video on building a RAG system). Pure reportlab/Platypus — English only, since
the built-in PDF fonts can't render Bengali glyphs (Bangla examples are
described in words)."""

import sys
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, HRFlowable, ListFlowable,
    ListItem, Table, TableStyle,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = "PPR-Bot_RAG_Report.pdf"

INK = colors.HexColor("#16140C")
ACCENT = colors.HexColor("#B98E00")   # readable "brand" gold for text
TEAL = colors.HexColor("#0E6B5B")
MUTED = colors.HexColor("#5E5A49")
PAPER = colors.HexColor("#FCFBF5")
CARDBG = colors.HexColor("#F4F1E6")
RED = colors.HexColor("#B23A2E")

styles = getSampleStyleSheet()

def S(name, **kw):
    styles.add(ParagraphStyle(name, **kw))

S("Cover", fontName="Helvetica-Bold", fontSize=30, leading=36, textColor=INK,
  alignment=TA_CENTER, spaceAfter=10)
S("CoverSub", fontName="Helvetica", fontSize=15, leading=21, textColor=MUTED,
  alignment=TA_CENTER, spaceAfter=6)
S("H1", fontName="Helvetica-Bold", fontSize=21, leading=26, textColor=INK,
  spaceBefore=18, spaceAfter=8)
S("H2", fontName="Helvetica-Bold", fontSize=16, leading=21, textColor=TEAL,
  spaceBefore=14, spaceAfter=5)
S("Body", fontName="Helvetica", fontSize=14, leading=20, textColor=INK,
  spaceAfter=9, alignment=TA_LEFT)
S("RBullet", fontName="Helvetica", fontSize=14, leading=19, textColor=INK,
  leftIndent=6)
S("Small", fontName="Helvetica-Oblique", fontSize=11, leading=15,
  textColor=MUTED, spaceAfter=6)
S("Label", fontName="Helvetica-Bold", fontSize=14, leading=20, textColor=ACCENT)
S("RCode", fontName="Courier", fontSize=11.5, leading=15, textColor=INK,
  backColor=CARDBG, borderPadding=6, spaceAfter=8)

story = []

def h1(t): story.append(Paragraph(t, styles["H1"]))
def h2(t): story.append(Paragraph(t, styles["H2"]))
def body(t): story.append(Paragraph(t, styles["Body"]))
def small(t): story.append(Paragraph(t, styles["Small"]))
def code(t): story.append(Paragraph(t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"), styles["RCode"]))
def gap(h=6): story.append(Spacer(1, h))
def rule(): story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#DAD6C7"), spaceBefore=6, spaceAfter=10))
def bullets(items):
    flow = [ListItem(Paragraph(x, styles["RBullet"]), value="•", leftIndent=14) for x in items]
    story.append(ListFlowable(flow, bulletType="bullet", start="•", leftIndent=10, spaceAfter=9))

def challenge(num, title, symptom, cause, fix, lesson):
    h2(f"{num}. {title}")
    rows = [
        [Paragraph("<b>Symptom</b>", styles["Body"]), Paragraph(symptom, styles["Body"])],
        [Paragraph("<b>Root cause</b>", styles["Body"]), Paragraph(cause, styles["Body"])],
        [Paragraph("<b>Fix</b>", styles["Body"]), Paragraph(fix, styles["Body"])],
        [Paragraph("<b>Lesson</b>", styles["Body"]), Paragraph(lesson, styles["Body"])],
    ]
    t = Table(rows, colWidths=[28*mm, 140*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), CARDBG),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LINEBELOW", (0,0), (-1,-2), 0.5, colors.HexColor("#E0DCCB")),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING", (0,0), (-1,-1), 9),
        ("RIGHTPADDING", (0,0), (-1,-1), 9),
        ("LINEAFTER", (0,0), (0,-1), 0.5, colors.HexColor("#E0DCCB")),
    ]))
    story.append(t)
    gap(12)

# ============================ COVER ============================
gap(120)
story.append(Paragraph("Building a Bilingual Legal RAG Chatbot", styles["Cover"]))
story.append(Paragraph("PPR-Bot — a from-scratch Retrieval-Augmented Generation system over the "
                       "Bangladesh Public Procurement Rules 2025", styles["CoverSub"]))
gap(16)
story.append(Paragraph("A build log &amp; teaching companion: the architecture, the RAG techniques, "
                       "and every real-world challenge hit along the way.", styles["CoverSub"]))
gap(40)
story.append(Paragraph("Prepared as a reference for a video walkthrough on making a RAG system.", styles["Small"]))
story.append(PageBreak())

# ============================ 1. INTRO ============================
h1("1. What we built, in one paragraph")
body("PPR-Bot is a chat assistant that answers questions — in Bangla or English — about a 271-page "
     "Bangladeshi government legal document (the Public Procurement Rules 2025, published in the "
     "official Gazette). It is a <b>Retrieval-Augmented Generation (RAG)</b> system: instead of "
     "trusting a language model to remember the law, we retrieve the exact relevant passages from "
     "the document and force the model to answer <i>only</i> from them, citing the rule number and "
     "page. It is hand-written end-to-end — no LangChain or LlamaIndex — specifically so every step "
     "is visible and teachable. It is served by a FastAPI backend that streams answers to a "
     "KakaoTalk-style chat UI.")
body("The single most important idea to convey in your video: <b>RAG is not one model call. It is a "
     "pipeline.</b> Most of the engineering — and almost all of the hard problems — live in getting "
     "clean text in, chunking it sensibly, indexing it well, and retrieving the right pieces. The "
     "language model is the last and easiest 5%.")

# ============================ 2. WHAT IS RAG ============================
h1("2. The mental model: why RAG, and what the pieces are")
body("A plain chatbot answers from the model's frozen training memory. For a specific legal document "
     "that may be newer than the model — and where a wrong answer is dangerous — that is unacceptable. "
     "RAG fixes this by splitting the job into two halves:")
bullets([
    "<b>Offline (build the knowledge base):</b> turn the PDF into clean text, cut it into searchable "
    "chunks, and build search indexes over those chunks. Done once.",
    "<b>Online (answer a question):</b> retrieve the most relevant chunks for the user's question, "
    "paste them into a prompt as context, and have the model answer strictly from that context.",
])
body("Separating these two is itself a core teaching point: the offline side is a set of resumable "
     "batch scripts that each produce an artifact on disk; the online side is a fast server that only "
     "reads those artifacts. They never get tangled together.")

# ============================ 3. ARCHITECTURE ============================
h1("3. Architecture &amp; data flow")
h2("Offline pipeline (run once, in order)")
bullets([
    "<b>Extraction</b> — render each PDF page to an image and have a vision LLM transcribe it to clean "
    "Markdown (this dodges the document's broken fonts; see Challenge 1).",
    "<b>Chunking</b> — parse the Markdown into the legal hierarchy (Part &gt; Chapter &gt; Rule &gt; "
    "Sub-rule) and cut one chunk per rule/section, keeping a breadcrumb and source page.",
    "<b>Enrichment</b> — for each chunk, an LLM writes a 1–2 sentence 'situating' summary that is "
    "prepended before indexing (Anthropic's Contextual Retrieval).",
    "<b>Indexing</b> — embed every chunk into a dense vector (bge-m3) AND build a keyword index "
    "(BM25). Two complementary indexes.",
])
h2("Online serving (per question)")
bullets([
    "<b>Condense</b> — rewrite a conversational follow-up into a standalone query using chat history.",
    "<b>Hybrid retrieve</b> — run dense (semantic) and sparse (keyword) search, fuse the two ranked "
    "lists with Reciprocal Rank Fusion (RRF).",
    "<b>Rerank</b> — a cross-encoder re-scores the top candidates for precision (now toggleable).",
    "<b>Generate</b> — stream a grounded, cited answer from the retrieved context.",
    "<b>Suggest</b> — propose follow-up questions for the UI.",
])
small("Tech stack: FastAPI + Uvicorn + SSE streaming; Google Gemini for OCR / generation / query "
      "rewriting; BAAI/bge-m3 embeddings and BAAI/bge-reranker-v2-m3 cross-encoder (local, CPU); "
      "rank_bm25 for keyword search; a hand-rolled NumPy vector store; vanilla HTML/CSS/JS frontend.")

# ============================ 4. RAG TECHNIQUES / VARIATIONS ============================
story.append(PageBreak())
h1("4. The RAG techniques used (and why)")
body("This system deliberately layers several 'beyond-basic' RAG techniques. For the video, each one "
     "is a great mini-segment: show the naive version, show why it fails, then add the technique.")

h2("4.1 Vision-LLM OCR ingestion")
body("Rather than extracting the PDF's text layer (which was garbled), we render each page to an image "
     "and ask a vision model to transcribe it. This treats every page — clean text, broken-font text, "
     "or scanned image — identically.")

h2("4.2 Structure-aware chunking")
body("Naive RAG chops text into fixed 500-character windows, which slices rules in half. We instead "
     "chunk along the document's own hierarchy, so each chunk is a coherent rule and carries its "
     "breadcrumb (Part &gt; Chapter &gt; Rule) and source page as metadata for citation.")

h2("4.3 Contextual Retrieval (Anthropic's technique)")
body("A lone chunk like 'the limit shall not exceed 3 lakh taka' is meaningless to a search engine — "
     "which rule? about what? Before indexing, an LLM writes a short blurb situating the chunk "
     "('This is from Rule 76 on direct procurement, stating the goods threshold...') and we prepend "
     "it. Both indexes then capture context the raw chunk lacked. This measurably reduces 'relevant "
     "but unfindable' misses.")

h2("4.4 Hybrid search = dense + sparse")
body("<b>Dense</b> (embeddings) captures meaning — a query about 'purchase limits' matches a chunk "
     "about 'procurement thresholds' even with no shared words. <b>Sparse</b> (BM25) captures exact "
     "terms — rule numbers, specific legal phrases — which embeddings can blur. Real questions need "
     "both, so we run both.")

h2("4.5 Reciprocal Rank Fusion (RRF)")
body("How do you merge two ranked lists whose scores are on totally different scales (cosine "
     "similarity vs BM25)? RRF ignores the raw scores and uses only the <i>rank position</i>: each "
     "item scores 1/(k + rank) in each list, summed. Simple, robust, no tuning. A perfect small "
     "teaching unit (it fits in ten lines and is easy to unit-test).")

h2("4.6 Cross-encoder reranking")
body("Hybrid search is fast but approximate — it compares query and chunk vectors that were computed "
     "separately. A cross-encoder reads the query AND a chunk <i>together</i> and outputs one "
     "relevance score: far more accurate, far slower. So we only run it on the ~10–15 candidates "
     "hybrid search already shortlisted. This 'retrieve cheap, rerank precise' two-stage pattern is "
     "standard in modern RAG.")

h2("4.7 Conversational query condensation")
body("Follow-ups like 'what about for works?' are meaningless to a search engine. Before retrieving, "
     "a cheap LLM call rewrites the follow-up into a standalone query using recent turns. This is what "
     "makes multi-turn RAG actually work.")

h2("4.8 Grounded generation with citations")
body("The system prompt forces the model to answer only from the provided context, cite the rule and "
     "page for each fact, and explicitly say when the answer is not present — the main defence against "
     "hallucination. We observed this working: asked about exact thresholds not yet in the extracted "
     "pages, the bot correctly said the figures were not in its context instead of inventing them.")

h2("4.9 Follow-up suggestions")
body("After answering, a cheap LLM call proposes 3 natural next questions, returned to the UI as "
     "tappable chips. Generating these on the server (not hardcoding them) keeps them relevant to the "
     "actual conversation.")

h2("4.10 A latency/precision toggle (a RAG variation worth discussing)")
body("On a slow CPU the cross-encoder rerank dominated response time. We added a per-request toggle: "
     "with rerank ON you get maximum precision (~30s here); with it OFF you fall back to hybrid+RRF "
     "ordering only (~5s), retrieving the <i>same</i> content with slightly less precise ordering. "
     "This is a real engineering dial every RAG builder should understand: <b>retrieval quality vs "
     "latency vs cost</b>.")

# ============================ 5. CHALLENGES ============================
story.append(PageBreak())
h1("5. The challenges — the heart of the story")
body("This is the most valuable part for a video: real problems, in the order they bit us, each with "
     "its symptom, root cause, and fix. Many of these are not in tutorials because tutorials use clean "
     "data and unlimited APIs.")

challenge("1", "The PDF text was unreadable mojibake",
    "Extracting the PDF's text layer produced garbage like 'evsjv‡`k' instead of Bangla.",
    "The document uses a legacy non-Unicode Bangla font (SutonnyMJ) that maps Bangla glyphs onto "
    "Latin code points. Direct text extraction reads the code points, not the glyphs.",
    "Stop extracting text; render each page to an image and have a vision LLM transcribe what it "
    "<i>sees</i> into proper Unicode. Font-agnostic, and it also handles scanned pages.",
    "Inspect your raw data before building anything. The hardest RAG problems are data problems.")

challenge("2", "Free-tier API quota kept running out mid-job",
    "The 271-page extraction halted partway, every further call returning 429 RESOURCE_EXHAUSTED.",
    "The free Gemini tier caps each model at ~20 requests/day. A 271-page OCR job plus ~170 "
    "enrichment calls vastly exceeds that.",
    "Two fixes: (a) make every batch job <b>resumable</b> — write each page to disk with a status "
    "manifest and skip completed work on re-run; (b) use <b>per-step fallback model chains</b> — each "
    "model name is a separate daily bucket, so trying a second model name doubles the allowance. The "
    "job now legitimately spans several days.",
    "Design long LLM batch jobs to be idempotent and resumable from day one — they <i>will</i> be "
    "interrupted.")

challenge("3", "Using the cheap model for OCR garbled the Bangla",
    "Switching OCR to a cheaper 'lite' model produced romanized mojibake and repetition loops.",
    "Small models can generate fluent text but cannot reliably <i>read</i> dense rendered Bangla "
    "script.",
    "Match the model to the task: full models for vision OCR (quality matters), the cheap lite model "
    "only for plain text generation like chunk-summaries (where it is fine and the daily quota is "
    "bigger).",
    "'Cheaper model' is a per-task decision, not a global one.")

challenge("4", "Some pages were silently saved as empty files",
    "About a dozen pages produced 0-byte Markdown files yet were marked 'done'.",
    "When the model returned an empty response, the code saved the empty string and recorded success.",
    "Treat an empty transcription as an error (raise, don't save), and make the resume check require "
    "the output file to be non-empty before skipping it.",
    "Validate every external response. 'It returned without error' is not the same as 'it returned "
    "something usable'.")

challenge("5", "A model fell into an endless repetition loop",
    "One page became 130 KB of the same character repeated — a runaway generation.",
    "Degenerate decoding: the lite model got stuck repeating a token.",
    "A guard that rejects suspiciously long output or long single-character runs, so the bad output is "
    "never saved and the page is retried.",
    "Add cheap sanity checks on generated output; degenerate loops are common and easy to detect.")

challenge("6", "The job ignored the quota signal and burned through retries",
    "After quota ran out, the job kept retrying every remaining page for ages instead of stopping.",
    "The 429 error text was attached as an exception 'cause' but not included in the message string "
    "the outer loop inspected, so the 'stop on quota' check never matched.",
    "Detect 429/RESOURCE_EXHAUSTED inside the transcribe call and break to the next model immediately "
    "(no backoff sleep); include the underlying error text in the raised message so the orchestrator "
    "can stop the whole job cleanly.",
    "When you catch-and-re-raise, preserve the information downstream code relies on.")

challenge("7", "Requests hung forever on a flaky connection",
    "Both background jobs froze for 10+ minutes with zero progress and no error.",
    "The Google SDK's HTTP client had no bounded timeout, so a stalled network read blocked "
    "indefinitely — indistinguishable from 'still working'.",
    "A single shared client factory that sets a 60-second request timeout, so a stuck call raises "
    "(and is then retried) instead of hanging.",
    "Every network client needs an explicit timeout. 'Hung' is worse than 'failed' because you cannot "
    "even tell it happened.")

challenge("8", "Gazette page furniture leaked into the chunks",
    "Chunks contained repeated running headers, printed page numbers, and a cover-page price stamp "
    "mid-text.",
    "The OCR faithfully transcribed the boilerplate that the printed Gazette repeats on every page; "
    "concatenating pages dropped it into the middle of rule text.",
    "A small filter in the parser that drops these known furniture lines, with unit tests proving real "
    "sub-rule references that look similar are NOT removed.",
    "Clean at the seams. Boilerplate that is harmless on one page becomes noise once pages are joined.")

challenge("9", "The model files would not download (stuck at 0 bytes)",
    "Downloading the embedder/reranker (~4.6 GB) stalled at 0 bytes across every method, while small "
    "files downloaded fine.",
    "The model library's modern chunked transfer protocol hung on this network, even though plain "
    "authenticated HTTPS range requests to the very same servers worked at ~2 MB/s.",
    "Bypass the library's downloader entirely: a small script using plain HTTP range requests with "
    "resume + retries, saving into a local folder the code loads models from directly. Also skipped "
    "the unused ONNX copies to halve the download.",
    "When a high-level helper mysteriously fails, drop one layer down and verify the primitive works. "
    "An auth token also bypasses anonymous rate-limit throttling.")

challenge("10", "Answers took ~30 seconds on CPU",
    "Every reply took about half a minute before streaming.",
    "The 568M cross-encoder reranker scores each (query, chunk) pair with a full forward pass; on a "
    "GPU-less, modest CPU that is several seconds per pair.",
    "Four levers: use all CPU cores (the library defaulted to half); cap the rerank token window; "
    "shrink the candidate pool; warm the models at server startup so the first question is not slow. "
    "Then a user toggle to skip rerank entirely for ~5s answers.",
    "Profile before optimizing — we measured each stage and found 90% of the time was one step. "
    "Latency, cost, and quality are dials, not constants.")

challenge("11", "The browser showed no answer and no error",
    "Typing dots appeared for ~30s, then vanished with a blank chat — no answer, no error.",
    "The server streams Server-Sent Events with CRLF (\\r\\n) line endings, but the browser parser "
    "split frames on '\\n\\n', which never matched — so it silently dispatched zero events. (Server-"
    "side Python tests passed because their line reader tolerated CRLF, hiding the bug.)",
    "Make the SSE parser CRLF-tolerant (split on a \\r?\\n\\r?\\n pattern).",
    "Test the real client, not just a convenient stand-in. The transport format matters down to the "
    "byte.")

challenge("12", "Answers showed literal asterisks instead of formatting",
    "Bullets rendered as '*' and the italic source line showed raw '*...*'.",
    "The frontend formatter only converted **bold**, leaving the model's Markdown bullets and "
    "single-asterisk italics untouched.",
    "A small, safe Markdown renderer: escape HTML first, then render bullets, bold, and italics "
    "(bold before italic so the passes do not collide).",
    "If the model emits Markdown, the UI must render Markdown — and always escape before injecting "
    "HTML.")

challenge("13", "Windows console crashes on Bangla / dashes",
    "Scripts and the server crashed with a cp932 codec error when printing Bangla or an em-dash.",
    "The default Windows console encoding cannot encode those characters.",
    "Reconfigure stdout to UTF-8 with error replacement at the top of every entry point.",
    "Cross-platform text output needs an explicit UTF-8 setup; do not assume the terminal's default.")

# ============================ 6. TEACHING / VIDEO ============================
story.append(PageBreak())
h1("6. How to teach this in a video")
body("A suggested narrative arc that keeps viewers engaged by leading with problems, not jargon:")
bullets([
    "<b>Hook:</b> ask a plain chatbot a specific legal question; watch it hallucinate or refuse. "
    "State the goal: trustworthy, cited answers from a real 271-page law.",
    "<b>Show the data reality:</b> open the PDF, copy-paste the 'text', reveal the mojibake. This "
    "instantly motivates the vision-OCR approach (Challenge 1).",
    "<b>Build the offline pipeline live</b>, one artifact at a time: pages to Markdown, Markdown to "
    "chunks, chunks to indexes. Emphasize 'each step writes a file you can open and inspect'.",
    "<b>Teach each retrieval technique by failure:</b> show keyword-only missing a paraphrase, then "
    "embeddings missing an exact rule number, then hybrid+RRF getting both. Then show rerank fixing a "
    "borderline ordering.",
    "<b>Wire the online side:</b> condense to retrieve to generate, streaming token by token over SSE.",
    "<b>Tell the war stories:</b> the quota wall, the 0-byte pages, the silent SSE bug, the 30-second "
    "wait and the toggle. These are what make the video memorable and honest.",
    "<b>End on the dials:</b> quality vs latency vs cost — the rerank toggle as the concrete example.",
])
h2("Talking points that land well")
bullets([
    "'RAG is a pipeline, not a prompt.' Most effort is data engineering.",
    "'Retrieve cheap, rerank precise' — the two-stage retrieval pattern.",
    "'Contextual Retrieval' — a few cents of LLM calls at index time buys real recall.",
    "'Make batch jobs resumable' — because free tiers and networks will interrupt you.",
    "'Test the real client and validate every external response.'",
])

# ============================ 7. TAKEAWAYS ============================
h1("7. Key takeaways")
bullets([
    "The model is the easy part; clean ingestion, sensible chunking, and good retrieval are where the "
    "quality and the work are.",
    "Hybrid (dense + sparse) + RRF + cross-encoder rerank is a strong, explainable default stack.",
    "Contextual Retrieval is a high-leverage, low-code upgrade.",
    "Engineer for failure: resumable jobs, bounded timeouts, output validation, graceful degradation.",
    "Expose the quality/latency/cost trade-off as a real control instead of hardcoding it.",
    "Ground every answer in retrieved context and force citations; let the model say 'not found'.",
])
gap(10)
rule()
small("Generated for the PPR-Bot project. All code referenced is hand-written and available in the "
      "project repository; the offline artifacts (extracted pages, chunks, embeddings, models) are "
      "regenerated by the pipeline scripts described above.")

doc = SimpleDocTemplate(
    OUT, pagesize=A4,
    leftMargin=22*mm, rightMargin=22*mm, topMargin=20*mm, bottomMargin=18*mm,
    title="PPR-Bot: Building a Bilingual Legal RAG Chatbot",
    author="PPR-Bot project",
)

def footer(canvas, d):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(A4[0]/2, 10*mm, f"PPR-Bot RAG build & teaching report   ·   page {d.page}")
    canvas.restoreState()

doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"Wrote {OUT}")
