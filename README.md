# retrieval-bench

A small script that compares three ways of searching a document set (keyword, character n-gram, and a mix of both) at several chunk sizes, and prints hit@1, hit@3 and MRR for each.

Independent demo, not affiliated with INDI Staffing Services (cliente não diz). I wrote it after reading a "RAG Engineer" job post, to practice the retrieval part of a RAG pipeline: chunking, ranking, fusion and measurement.

All data in this repo is invented. The 13 help-center articles describe a made-up shipping product, and the 20 questions were written by hand.

## Run it

Python 3.12 (the version CI uses). No dependencies to run; `pytest` only for the tests.

```sh
python retrieval_bench.py            # table for chunk sizes 20, 40, 80, 400 words
python retrieval_bench.py --misses   # also list questions not answered at rank 1
python retrieval_bench.py --sizes 30 60 --json
python retrieval_bench.py --corpus my_docs.json --queries my_questions.json
```

Your own files use the same shape as `data/corpus.json` (`{"docs": [{"id", "text"}]}`) and `data/queries.json` (`{"queries": [{"q", "doc", "kind"}]}`).

## How it works

- **Chunking:** fixed windows of N words with 25% overlap.
- **bm25:** word-level BM25.
- **ngram:** TF-IDF cosine over character 3-grams. This stands in for a vector index. It is not a semantic embedding model, so it does not know that "VAT number" and "company tax number" mean the same thing.
- **hybrid:** reciprocal rank fusion (k=60) of the two ranked chunk lists.
- Chunk scores collapse to one score per document (best chunk wins), so every chunk size is judged on the same questions.

## Result on the bundled data

At 80-word chunks, hit@1 is bm25 0.85, hybrid 0.80, ngram 0.70. Hybrid did not beat plain BM25 here.

Questions that reuse the article's words are answered at rank 1 every time by bm25 and hybrid. Questions that paraphrase the article ("customer wants to send an item back") are the weak spot: 7 of the 20 questions, and hit@1 on them is bm25 0.57, hybrid 0.43, ngram 0.29.

Chunk size matters too. At 20-word chunks, hybrid hit@1 on the shared-word questions drops from 1.00 to 0.78.

Run `python retrieval_bench.py --misses` to see the exact failures.

## What was tested

- 19 offline tests (`python -m pytest -q`): chunking edge cases, BM25 ranking, n-gram matching of word variants, rank fusion, collapsing chunks to documents, data consistency, the CLI, and the pinned numbers above on the bundled data.
- The same tests run in GitHub Actions on every push.

## What was not tested

- No real embedding model, vector database or reranker. The n-gram index only matches spelling, so the paraphrase result says nothing about what real embeddings would do.
- 20 questions is a small sample. One question moves hit@1 by 0.05, and the gap between bm25 and hybrid is three questions or fewer. Treat it as a demo, not a benchmark.
- No answer generation and no groundedness scoring (for example RAGAS). It measures retrieval only.
- Only English text, only short articles, nothing real-world messy such as tables or HTML.
- Not tested on a corpus larger than a few dozen documents.
