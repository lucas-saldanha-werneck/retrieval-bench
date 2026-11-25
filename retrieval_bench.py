"""Compare three retrieval strategies on a small labeled set.

Strategies:
  bm25    word-level BM25 (keyword search)
  ngram   TF-IDF cosine on character 3-grams (a lexical stand-in for a vector
          index; it is NOT a semantic embedding model)
  hybrid  reciprocal rank fusion (RRF) of the two lists

Chunks are scored, then collapsed to documents (best chunk wins) so that
different chunk sizes can be compared on the same questions.
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).parent / "data"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Split text into windows of `size` words, each sharing `overlap` words with the last."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and < size")
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return out


def build_chunks(docs: list[dict], size: int, overlap: int) -> list[tuple[str, str]]:
    """Return (doc_id, chunk_text) pairs."""
    return [(d["id"], c) for d in docs for c in chunk_words(d["text"], size, overlap)]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokens(t)) for t in texts]
        self.lens = [sum(c.values()) for c in self.docs]
        self.avg = sum(self.lens) / len(self.lens) if self.lens else 0.0
        df: Counter = Counter()
        for c in self.docs:
            df.update(c.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokens(query)
        out = []
        for c, length in zip(self.docs, self.lens):
            s = 0.0
            for t in q:
                f = c.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * length / self.avg)
                    s += self.idf[t] * f * (self.k1 + 1) / norm
            out.append(s)
        return out


def trigrams(text: str) -> list[str]:
    out = []
    for w in tokens(text):
        p = f"  {w} "
        out.extend(p[i:i + 3] for i in range(len(p) - 2))
    return out


class NgramIndex:
    def __init__(self, texts: list[str]):
        counts = [Counter(trigrams(t)) for t in texts]
        df: Counter = Counter()
        for c in counts:
            df.update(c.keys())
        n = len(counts)
        self.idf = {g: math.log((1 + n) / (1 + f)) + 1 for g, f in df.items()}
        self.vecs = [self._norm(c) for c in counts]

    def _norm(self, counts: Counter) -> dict[str, float]:
        v = {g: (1 + math.log(f)) * self.idf.get(g, 0.0) for g, f in counts.items()}
        length = math.sqrt(sum(x * x for x in v.values()))
        return {g: x / length for g, x in v.items()} if length else {}

    def scores(self, query: str) -> list[float]:
        q = self._norm(Counter(trigrams(query)))
        return [sum(w * v.get(g, 0.0) for g, w in q.items()) for v in self.vecs]


def rank(scores: list[float]) -> list[int]:
    """Indexes sorted best first. Zero-score items are dropped; ties keep input order."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return [i for i in order if scores[i] > 0]


def rrf(rankings: list[list[int]], k: int = 60) -> list[float]:
    """Reciprocal rank fusion. Returns a score per item index (length = max index + 1)."""
    size = 1 + max((i for r in rankings for i in r), default=-1)
    out = [0.0] * size
    for r in rankings:
        for pos, i in enumerate(r, start=1):
            out[i] += 1.0 / (k + pos)
    return out


def to_doc_ranking(chunk_scores: list[float], chunk_docs: list[str]) -> list[str]:
    """Collapse chunk scores to document ids, best chunk per document."""
    best: dict[str, float] = defaultdict(float)
    for s, d in zip(chunk_scores, chunk_docs):
        if s > best[d]:
            best[d] = s
    return [d for d, s in sorted(best.items(), key=lambda kv: (-kv[1], kv[0])) if s > 0]


def evaluate(docs: list[dict], queries: list[dict], size: int, overlap: int, ks=(1, 3)) -> dict:
    """Return {strategy: {"hit@k": ..., "mrr": ..., "per_kind": {...}, "misses": [...]}}."""
    chunks = build_chunks(docs, size, overlap)
    chunk_docs = [d for d, _ in chunks]
    texts = [t for _, t in chunks]
    bm, ng = BM25(texts), NgramIndex(texts)
    result = {}
    for name in ("bm25", "ngram", "hybrid"):
        hits = {k: 0 for k in ks}
        rr_sum = 0.0
        kind_hits: dict[str, list[int]] = defaultdict(list)
        misses = []
        for q in queries:
            if name == "bm25":
                cs = bm.scores(q["q"])
            elif name == "ngram":
                cs = ng.scores(q["q"])
            else:
                cs = rrf([rank(bm.scores(q["q"])), rank(ng.scores(q["q"]))])
                cs = cs + [0.0] * (len(chunks) - len(cs))
            ranking = to_doc_ranking(cs, chunk_docs)
            pos = ranking.index(q["doc"]) + 1 if q["doc"] in ranking else None
            for k in ks:
                if pos is not None and pos <= k:
                    hits[k] += 1
            rr_sum += 1.0 / pos if pos else 0.0
            kind_hits[q.get("kind", "?")].append(1 if pos == 1 else 0)
            if pos != 1:
                misses.append({"q": q["q"], "expected": q["doc"], "got": ranking[:1], "pos": pos})
        n = len(queries)
        result[name] = {
            **{f"hit@{k}": hits[k] / n for k in ks},
            "mrr": rr_sum / n,
            "per_kind": {k: sum(v) / len(v) for k, v in sorted(kind_hits.items())},
            "misses": misses,
        }
    return result


def load(path: Path, key: str) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))[key]


def format_table(results_by_size: dict[int, dict]) -> str:
    lines = ["| chunk words | strategy | hit@1 | hit@3 | MRR |", "|---|---|---|---|---|"]
    for size, res in results_by_size.items():
        for name, r in res.items():
            lines.append(f"| {size} | {name} | {r['hit@1']:.2f} | {r['hit@3']:.2f} | {r['mrr']:.2f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", type=Path, default=DATA / "corpus.json")
    p.add_argument("--queries", type=Path, default=DATA / "queries.json")
    p.add_argument("--sizes", type=int, nargs="+", default=[20, 40, 80, 400])
    p.add_argument("--overlap-ratio", type=float, default=0.25)
    p.add_argument("--misses", action="store_true", help="list questions not answered at rank 1")
    p.add_argument("--json", action="store_true", help="print raw JSON")
    a = p.parse_args(argv)
    docs, queries = load(a.corpus, "docs"), load(a.queries, "queries")
    out = {s: evaluate(docs, queries, s, int(s * a.overlap_ratio)) for s in a.sizes}
    if a.json:
        print(json.dumps(out, indent=2))
        return 0
    print(f"{len(docs)} documents, {len(queries)} questions\n")
    print(format_table(out))
    print("\nhit@1 by question kind (hybrid):")
    for s, res in out.items():
        print(f"  {s:>4} words: " + ", ".join(f"{k}={v:.2f}" for k, v in res["hybrid"]["per_kind"].items()))
    if a.misses:
        s = a.sizes[0]
        print(f"\nHybrid misses at {s} words:")
        for m in out[s]["hybrid"]["misses"]:
            print(f"  - {m['q']!r}: expected {m['expected']}, got {m['got']} (rank {m['pos']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
