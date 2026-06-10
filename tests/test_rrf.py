"""Unit tests for Reciprocal Rank Fusion — the core hybrid-search math.

RRF is small enough to verify by hand, which makes it a great unit test:
we can compute the expected fused scores ourselves and assert on them.
"""

from ppr_bot.retrieval.hybrid_search import reciprocal_rank_fusion


def test_document_in_both_lists_outranks_single_list_documents():
    # "b" appears high in both lists; "a" only in list 1; "c" only in list 2.
    dense = ["a", "b", "c"]
    sparse = ["b", "c", "d"]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    fused_ids = [cid for cid, _ in fused]
    # b is rank1 in sparse and rank2 in dense -> should be #1 overall.
    assert fused_ids[0] == "b"


def test_scores_match_hand_computed_values():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60))
    # a: 1/(60+1) + 1/(60+2);  b: 1/(60+2) + 1/(60+1)  -> equal
    expected = 1 / 61 + 1 / 62
    assert abs(fused["a"] - expected) < 1e-12
    assert abs(fused["b"] - expected) < 1e-12


def test_single_list_preserves_order():
    fused = reciprocal_rank_fusion([["x", "y", "z"]], k=60)
    assert [cid for cid, _ in fused] == ["x", "y", "z"]


def test_k_constant_changes_weighting():
    # Smaller k makes rank-1 dominate more strongly.
    small_k = dict(reciprocal_rank_fusion([["a", "b"]], k=1))
    assert small_k["a"] > small_k["b"]


def test_empty_input():
    assert reciprocal_rank_fusion([]) == []
