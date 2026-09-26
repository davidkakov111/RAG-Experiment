# RAG Retrieval Experiment

Minimal script demonstrating RAG retrieval: embed text into vectors, find
the most semantically similar chunks to a question via cosine similarity.
No vector DB, no framework — brute-force, so the mechanics are visible.

## Run

```bash
npm install
node embed-and-search.mjs
```

First run downloads the embedding model (~90MB), cached after that.

## What to look for

Chunk set mixes exact matches, same-topic distractors, keyword-collision
traps, and unrelated noise.
