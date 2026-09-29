# Deployment

> Full deployment steps come in a later stage. This file currently holds notes that must be
> checked before the app is shared publicly.

## Cohere reranker: trial key limits

- Cohere trial API keys are free but **rate-limited**, and Cohere's terms say they are **not
  permitted for production or commercial use**.
- The app does not depend on Cohere being available: if reranking fails (missing key, rate
  limit, network or API error), `src/rerank.py` logs a warning and falls back to embedding
  similarity order. Match scores then show as "unavailable" and `MATCH_THRESHOLD` is not
  applied, since cosine similarity is on a different scale.
- **Before sharing the public link widely, check Cohere's current terms** and decide whether a
  public demo needs a production key.
