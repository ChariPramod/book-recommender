# Book Recommender

A retrieval-augmented book recommender: describe what you want to read, get 3 to 5 books from
a 12k-book dataset, reranked and summarized from the books' own stored summaries.

> Short version for now; the full README comes in a later stage.

## Query flow

Solid nodes are built. Dashed nodes would mark planned parts (none right now).

```mermaid
flowchart TD
    subgraph APP["app.py : Streamlit chat UI"]
        U(["User message"])
        OUT["Show reply + book cards<br/>rank, match score, title, author,<br/>genres, brief summary"]
    end

    subgraph CHAT["chat.py : conversation manager"]
        GATE{"Safety gate<br/>max 500 chars, 30 messages"}
        LIM["Limit message"]
        TALK["LLM reply: small talk,<br/>one clarifying question, or polite redirect"]
        ABOUT["Answer about a shown book<br/>from its stored summary only"]
        NOM["No strong matches:<br/>suggest how to rephrase"]
        MEM[("Session memory in st.session_state<br/>last 10 turns + books shown")]
    end

    subgraph ROUTER["router.py : intent + query rewrite"]
        R["LLM returns intent,<br/>standalone search query, book_ref"]
    end

    subgraph REC["recommend.py : search pipeline"]
        subgraph RET["retrieve.py"]
            E["Embed query<br/>text-embedding-3-small"] --> Q["Qdrant: top 30 books<br/>best chunk per book<br/>max 2 per author"]
        end
        subgraph RR["rerank.py"]
            C["Cohere rerank-v4.0-fast<br/>title + genres + summary<br/>top 10 with match score"] --> T{"Any book with match score<br/>at or above MATCH_THRESHOLD?"}
        end
        subgraph GEN["generate.py"]
            G["LLM_MODEL picks 3 to 5<br/>grounded brief summaries + fit 1-10<br/>code validates every book_id"]
        end
    end

    U --> GATE
    GATE -- blocked --> LIM
    GATE -- ok --> R
    R -- "small_talk / unclear / off_topic" --> TALK
    R -- about_shown_book --> ABOUT
    R -- "recommend / refine<br/>(refine skips books already shown)" --> E
    Q --> C
    Q -. "Cohere fails: similarity order" .-> G
    T -- yes --> G
    T -- no --> NOM
    G --> OUT
    TALK --> OUT
    ABOUT --> OUT
    NOM --> OUT
    LIM --> OUT
    OUT --> MEM
    MEM -. "history" .-> R

    classDef planned stroke-dasharray: 5 5
```

## Files in the query flow

| File | What it does in the query flow | Status |
|---|---|---|
| `app.py` | Streamlit chat UI: keeps the session in `st.session_state`, shows replies and book cards | Built |
| `src/chat.py` | Safety gate, session memory (last 10 turns), one handler per intent | Built (end-to-end run pending the smoke test) |
| `src/router.py` | One strict-JSON LLM call: intent, standalone search query, book_ref | Built (end-to-end run pending the smoke test) |
| `src/recommend.py` | Runs retrieve → rerank → threshold → generate; can exclude books already shown | Built |
| `src/retrieve.py` | Embeds the query; Qdrant top 30 books by best chunk, max 2 per author | Built |
| `src/embed.py` | `embed_texts()`, used here to embed the query | Built |
| `src/rerank.py` | Cohere rerank to a 0 to 1 match score; falls back to similarity order on failure | Built |
| `src/generate.py` | LLM picks 3 to 5 books, writes grounded summaries; code drops unknown book_ids | Built |
| `src/config.py` | Models, limits, threshold, credentials from `.env` | Built |

Offline indexing (run once, not per query): `src/load_data.py` → `src/chunking.py` →
`src/embed.py` → `src/store.py`.
