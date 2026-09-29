# Book Recommender: a learning project

## Who I am and what this is
I am learning RAG by building it myself. The goal is understanding, not speed.
The app: a user types a request like "recommend some superpower fiction books" and gets the
top 3 to 5 books from my dataset, ranked, each with a score, title, author, genre, and a brief summary.
Raw data: data/raw/cmu_books_clean.csv (author, title, genres, summary). 12,055 rows.
Clean data: data/processed/books.jsonl. 11,997 books.

## Teaching rules (follow these every session)
1. ONE FILE PER TURN. Never create or edit more than one file in a single response.
2. BEFORE writing a file, explain in plain language:
   - what this file is for
   - which RAG step it belongs to
   - the concept I should learn from it
   Then show a short plan of the functions inside it.
3. For core logic files (chunking.py, embed.py, retrieve.py), first ask whether I want to
   write it myself. If I say yes, give me function signatures, docstrings and hints,
   but not the implementation. Then review what I wrote.
4. AFTER writing a file:
   - walk through the important parts
   - give me the exact terminal command to run it
   - tell me what output to expect
   - ask me 2 short questions to check I understood
   Then STOP and wait. Do not continue until I type "next".
5. Keep the code simple: plain Python functions, type hints, and comments that explain WHY,
   not WHAT. No classes unless clearly needed.
   Do NOT use LangChain or LlamaIndex. We write the chunker and retrieval ourselves.
6. Never run a command that costs money (OpenAI API calls) without first showing me the
   estimated token count and cost, and waiting for my yes.
7. Do not write LEARNINGS.md for me. After each file, remind me to add an entry in my own words.
8. If I ask for something that breaks these rules, point that out before doing it.
9. When reviewing my code, run it and check the output counts against the expected numbers
   in this file before saying it works.

## Status
- DONE: notebooks/01_explore.ipynb, requirements.txt, .gitignore, src/config.py,
  src/load_data.py (11,997 books), src/chunking.py (43,697 chunks at prefix_230_ov1,
  same 43,697 at noprefix_230_ov1, over-budget breakdown clean at 0 "other" in both),
  tests/test_chunking.py, src/embed.py, src/store.py, src/retrieve.py, src/generate.py, app.py
  (dense retrieval only today, deadline build — rule 1 relaxed for this session, no
  walkthroughs done on retrieve/generate/app, not yet run)
- TOMORROW: BM25 + hybrid retrieval, reranker, eval, experiments (prefix on/off,
  overlap 0/1, chunk size 230/450/whole, dense/BM25/hybrid, rerank on/off)
- LATER: cli (superseded by app.py unless we still want one)

## Build order (one file at a time)
book-recommender/
  CLAUDE.md
  LEARNINGS.md                 (I write this)
  requirements.txt
  .gitignore                   (.venv/, .env, data/processed/, .ipynb_checkpoints/, __pycache__/)
  .env                         (OPENAI_API_KEY, created at the embedding step)
  data/raw/cmu_books_clean.csv
  data/processed/              (generated outputs)
  data/eval/queries.jsonl      (I write the queries BEFORE running retrieval; see Eval)
  notebooks/01_explore.ipynb   (done)
  src/config.py                (all settings in one place)
  src/load_data.py             (step 1: clean, done)
  tests/test_load_data.py
  src/chunking.py              (step 2: chunk)
  tests/test_chunking.py
  src/embed.py                 (step 3: embed)
  src/store.py                 (step 4: Qdrant)
  src/retrieve.py              (step 5: dense, BM25, hybrid)
  src/rerank.py                (step 6: rerank)
  src/generate.py              (step 7: LLM answer)
  src/cli.py                   (step 8: user interface)
  src/evaluate.py              (step 9: eval, last)

## Step 1: load and clean (load_data.py) - DONE
Order: clean markup, then drop junk, then deduplicate.
- Clean: cut from "<!--" to the end; remove "!--...-->" comments (lazy match) and stray
  "<!-", "-->", "~Plot outline description~"; remove runs of "="; remove "|----" and "|";
  collapse whitespace.
- Drop: cleaned summary under 10 words, or "To be added."
- Deduplicate: same (title, author) keeps the longest summary; an identical summary on
  different books drops all copies; never deduplicate on title alone.
- book_id = "b{row:05d}" from the ORIGINAL csv row, so ids never shift. Never use title as an id.
- Genres are stored as a list.
- Expected: 87 to 102 summaries changed, 53 dropped, 3 + 2 deduplicated, 11,997 saved.
- Known issue (deliberately not fixed): 3 books start with wiki table styling
  (Doorways in the Sand, A Fringe of Leaves, Never Call Retreat). Revisit only if eval shows harm.

## Step 2: chunking (chunking.py)
Input: data/processed/books.jsonl. Chunk EACH BOOK SEPARATELY; nothing crosses between books.

Settings (config.py):
- ENCODING = "cl100k_base"          (tiktoken; the text-embedding-3 tokenizer)
- MODEL_MAX_TOKENS = 8191           (hard limit of text-embedding-3-small)
- CHUNK_MAX_TOKENS = 230            (a quality choice to tune, not a model limit)
- OVERLAP_SENTENCES = 1             (experiment: 0 vs 1)
- MIN_CHUNK_TOKENS = 80
- USE_PREFIX = True                 (experiment: True vs False)
- STRATEGY = "sentence"             (experiment: "sentence" vs "whole")

Sentence splitting:
- Split on ., !, ? followed by a space and a capital letter or opening quote.
- Do NOT split after abbreviations: "U.N.", "U.S.", "Dr.", "Mr.", "Mrs.", single initials like "J.".

Strategy "sentence":
- Prefix (when USE_PREFIX): "{title} by {author} ({genres joined by ', '}): "
- Chunk boundaries are always decided using the real prefix, whether or not USE_PREFIX is on.
  USE_PREFIX changes only embedded_text (prefix + raw_text, or raw_text alone) — never where a
  chunk starts or ends. So the prefix on/off experiment isn't secretly also an experiment on chunk size.
- Budget per book = CHUNK_MAX_TOKENS minus that book's prefix tokens (or CHUNK_MAX_TOKENS if no prefix).
- Pack whole sentences until the next sentence would exceed the budget. Never split a sentence.
  Every "does this fit" check counts the tokens of the exact text that will be saved
  (prefix + joined sentences), not a sum of per-sentence token counts — those can differ by a
  token or two at a join boundary, since tokenization isn't strictly additive.
- The next chunk starts with the last OVERLAP_SENTENCES sentence(s) of the previous chunk,
  unless the overlap sentence(s) alone would already exceed the budget together with the next
  sentence — then drop the overlap for that chunk and start fresh.
- If a single sentence alone exceeds the budget, make it its own chunk (safe because
  MODEL_MAX_TOKENS is 8191). Log how many times this happens (expect roughly 40 to 55,
  depending on USE_PREFIX and OVERLAP_SENTENCES).
- If a book's final chunk is under MIN_CHUNK_TOKENS, merge it into the previous chunk
  (without repeating the overlap sentence), but only if the merged chunk stays under
  MODEL_MAX_TOKENS. A merged tail may therefore exceed CHUNK_MAX_TOKENS by up to
  roughly MIN_CHUNK_TOKENS — capping at CHUNK_MAX_TOKENS instead would disable the merge
  in practice, since the whole point is absorbing a too-small leftover.

Strategy "whole":
- One chunk per book: prefix + whole summary.
- If it exceeds MODEL_MAX_TOKENS (about 4 books), truncate to MODEL_MAX_TOKENS tokens and log the titles.

Chunk record (keep chunks lean; book metadata stays in books.jsonl and is joined at store time):
- chunk_id ("{book_id}-{chunk_index}"), book_id, chunk_index,
  raw_text (the summary text only), embedded_text (prefix + raw_text, or raw_text), n_tokens
Output: data/processed/chunks_{strategy}_{CHUNK_MAX_TOKENS}_{prefix|noprefix}_ov{OVERLAP_SENTENCES}.jsonl
Print: total chunks, chunks per book (median, max), token stats, oversized sentences, merged tails.
Expected (sentence, 230, prefix, overlap 1): roughly 40,000 to 45,000 chunks; median about 2 per book.

Tests (tests/test_chunking.py):
- every chunk's raw_text appears inside its own book's summary
- no chunk exceeds its budget, except single oversized sentences and merged tails
- consecutive chunks of the same book share exactly OVERLAP_SENTENCES sentence(s), except:
  a chunk next to an oversized sentence (no overlap carried in or out of it), and
  a chunk where the overlap was dropped because overlap + next sentence didn't fit
- the first chunk of a book contains no text from any other book
- "U.N." and "U.S." do not cause a sentence split
- a summary that fits the budget becomes exactly one chunk
- a sentence longer than the budget becomes exactly one chunk, unsplit
- when USE_PREFIX is False, embedded_text equals raw_text

## Step 3: embedding (embed.py)
- Model: text-embedding-3-small (1,536 dimensions), OpenAI Python SDK, key from .env via python-dotenv.
- The chunks file to embed is a command line argument.
- First, a one-sentence test call to confirm the key works.
- Before the full run: print total tokens and estimated cost ($0.02 per 1M tokens; a full run
  is about 8.5M tokens, about $0.17), then ask me for yes/no.
- Send embedded_text in batches of about 200, with retry and backoff on rate-limit errors.
- Save data/processed/embeddings_{same name as the chunks file}.npy, in the same order as the
  chunks file, plus a .json alongside it: model name, dimensions, chunks file, count, date.
- If the output file already exists, skip the API call entirely, so I never pay twice.
- After embedding, check: number of vectors equals number of chunks, and every vector has length about 1.0.
- Sanity check (not an eval): embed 3 queries, find the top 5 books with plain NumPy cosine
  similarity (best chunk per book), and print them. I compute cosine myself before using any database.

## Later steps (details decided when we get there)
- Step 4 store.py: Qdrant, local mode first. One point per chunk; point id = uuid5(chunk_id);
  payload = chunk fields + title, author, genres from books.jsonl. Payload indexes on book_id, genres, author.
- Step 5 retrieve.py: dense (Qdrant, grouped by book_id), then BM25 (rank_bm25, written myself),
  then hybrid with RRF (written myself first, then Qdrant's built-in). Optional genre filter.
- Step 6 rerank.py: cross-encoder on the top 30 books; 0 to 1 match score; return 3 to 5 above a
  threshold, or say "no good matches". Consider at most 2 books per author.
- Step 7 generate.py: LLM picks and orders 3 to 5 from the candidates, writes a brief summary
  using ONLY the stored summary, returns JSON by book_id. Code checks every book_id is a candidate.
- Step 8 cli.py: prints rank, match score, title, author, genres, brief summary.
- Pending decisions (ask me before the step): which LLM (step 7), which reranker (step 6),
  when to move Qdrant from local mode to Docker (step 4).

## Step 9: eval (last)
- Queries: I write 15 to 20 queries in data/eval/queries.jsonl BEFORE running retrieval for the
  first time, and do not change them after seeing results. Mix: genre or vibe, plot-specific,
  author-style, and the "superpower" keyword trap.
- Expected books: at the end, I find them by searching books.jsonl myself (keywords, reading
  summaries), NOT by copying what the system returned.
- Metrics: recall@5 and MRR for retrieval; for generation, 0 invented books and summaries
  faithful to the stored text.
- Experiments, one change at a time: prefix on/off, overlap 0/1, chunk size 230/450/whole,
  dense/BM25/hybrid, rerank on/off. Record every result in a table in LEARNINGS.md.