import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import retrieval_bench as rb  # noqa: E402

DOCS = rb.load(rb.DATA / "corpus.json", "docs")
QUERIES = rb.load(rb.DATA / "queries.json", "queries")


def test_chunk_short_text_is_one_chunk():
    assert rb.chunk_words("a b c", 10, 2) == ["a b c"]


def test_chunk_empty_text():
    assert rb.chunk_words("", 10, 2) == []


def test_chunk_overlap_and_coverage():
    text = " ".join(str(i) for i in range(10))
    chunks = rb.chunk_words(text, 4, 1)
    assert chunks[0] == "0 1 2 3"
    assert chunks[1].split()[0] == "3"  # one word shared with the previous chunk
    assert chunks[-1].split()[-1] == "9"  # nothing is dropped at the end
    assert len(chunks) == 3


def test_chunk_rejects_bad_args():
    with pytest.raises(ValueError):
        rb.chunk_words("a b c", 0, 0)
    with pytest.raises(ValueError):
        rb.chunk_words("a b c", 3, 3)


def test_bm25_prefers_rare_matching_term():
    bm = rb.BM25(["the cat sat", "the dog sat", "the dog barked loudly"])
    s = bm.scores("barked")
    assert rb.rank(s) == [2]
    assert s[0] == 0 and s[1] == 0


def test_bm25_no_match_gives_empty_ranking():
    bm = rb.BM25(["alpha beta", "gamma delta"])
    assert rb.rank(bm.scores("zzz")) == []


def test_ngram_matches_word_variants():
    ix = rb.NgramIndex(["refunds are issued quickly", "labels are printed here"])
    assert rb.rank(ix.scores("refunding"))[0] == 0


def test_rank_ties_keep_input_order_and_drop_zeros():
    assert rb.rank([0.5, 0.0, 0.5, 0.9]) == [3, 0, 2]


def test_rrf_rewards_agreement():
    # item 1 is second in both lists, item 0 first in one, item 2 first in the other
    fused = rb.rrf([[0, 1], [2, 1]])
    assert fused[1] > 0
    assert fused[0] == pytest.approx(fused[2])
    assert fused[1] > fused[0]  # rank 2 in both lists beats rank 1 in only one
    assert rb.rrf([[0, 1, 2], [2, 1, 0]])[1] > rb.rrf([[0, 1, 2]])[1]


def test_rrf_empty():
    assert rb.rrf([[], []]) == []


def test_doc_ranking_keeps_best_chunk_per_doc():
    got = rb.to_doc_ranking([0.2, 0.9, 0.5, 0.0], ["a", "b", "a", "c"])
    assert got == ["b", "a"]


def test_data_is_consistent():
    ids = [d["id"] for d in DOCS]
    assert len(ids) == len(set(ids))
    assert all(q["doc"] in ids for q in QUERIES)
    assert {q["kind"] for q in QUERIES} == {"shared-words", "word-variant", "paraphrase"}


def test_data_is_marked_as_example():
    raw = json.loads((rb.DATA / "corpus.json").read_text(encoding="utf-8"))
    assert "EXAMPLE" in raw["_note"]


def test_evaluate_perfect_on_trivial_set():
    docs = [{"id": "x", "text": "zebra stripes"}, {"id": "y", "text": "giraffe neck"}]
    qs = [{"q": "zebra", "doc": "x", "kind": "shared-words"}, {"q": "giraffe", "doc": "y", "kind": "shared-words"}]
    res = rb.evaluate(docs, qs, 10, 0)
    for name in ("bm25", "ngram", "hybrid"):
        assert res[name]["hit@1"] == 1.0
        assert res[name]["mrr"] == 1.0
        assert res[name]["misses"] == []


def test_evaluate_reports_miss():
    docs = [{"id": "x", "text": "zebra stripes"}, {"id": "y", "text": "giraffe neck"}]
    res = rb.evaluate(docs, [{"q": "elephant", "doc": "x", "kind": "paraphrase"}], 10, 0)
    assert res["bm25"]["hit@3"] == 0.0
    assert res["bm25"]["misses"][0]["pos"] is None


def test_known_numbers_on_bundled_data():
    """Pinned results for the bundled example set. A change here means the scoring changed."""
    r80 = rb.evaluate(DOCS, QUERIES, 80, 20)
    assert r80["bm25"]["hit@1"] == pytest.approx(0.85)
    assert r80["hybrid"]["hit@1"] == pytest.approx(0.80)
    assert r80["ngram"]["hit@1"] == pytest.approx(0.70)
    r20 = rb.evaluate(DOCS, QUERIES, 20, 5)
    assert r20["hybrid"]["per_kind"]["shared-words"] < r80["hybrid"]["per_kind"]["shared-words"]


def test_paraphrase_is_the_weak_spot():
    r = rb.evaluate(DOCS, QUERIES, 80, 20)
    for name in ("bm25", "ngram", "hybrid"):
        pk = r[name]["per_kind"]
        assert pk["paraphrase"] < pk["shared-words"]


def test_cli_runs(capsys):
    assert rb.main(["--sizes", "40"]) == 0
    out = capsys.readouterr().out
    assert "| 40 | hybrid |" in out


def test_cli_json(capsys):
    assert rb.main(["--sizes", "40", "--json"]) == 0
    assert "hybrid" in json.loads(capsys.readouterr().out)["40"]
