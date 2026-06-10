"""Unit tests for the BM25 index and the Bangla-aware tokenizer."""

from ppr_bot.indexing.bm25_index import BM25Index, simple_tokenize


def test_tokenizer_handles_bangla_and_english():
    tokens = simple_tokenize("Rule ৪২: সরাসরি ক্রয় (Direct Procurement)!")
    assert "rule" in tokens  # lowercased
    assert "৪২" in tokens  # Bangla digits kept
    assert "সরাসরি" in tokens  # Bangla word kept
    assert "direct" in tokens
    # Punctuation is stripped, not turned into tokens.
    assert ":" not in tokens and "!" not in tokens


def test_bm25_ranks_relevant_doc_first():
    corpus = [
        "সরাসরি ক্রয় পদ্ধতির সীমা দশ লক্ষ টাকা",
        "উন্মুক্ত দরপত্র পদ্ধতি বিষয়ক বিধান",
        "পরামর্শক সেবা ক্রয়ের নিয়মাবলী",
    ]
    index = BM25Index.build(corpus, ["c0", "c1", "c2"])
    results = index.search("সরাসরি ক্রয়", top_k=3)
    assert results[0][0] == "c0"  # the direct-procurement doc ranks first
