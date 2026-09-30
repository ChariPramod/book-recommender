"""Smoke test: one conversation of 6 messages through chat.handle_message(), printing
the intent, rewritten search query, and returned titles with match scores. Not an
eval — a quick look that every intent path works end to end with the real services.

Run from the project root (makes real OpenAI, Qdrant and Cohere calls):
    python -m scripts.smoke_test
"""

import warnings

from src.chat import handle_message, load_resources, new_session

MESSAGES = [
    "hey hello",
    "recommend some superpower fiction books",
    "tell me more about the second one",
    "something darker, not for kids",
    "something good",
    "what's the weather today",
]

# gpt-4o-mini standard tier, from OpenAI's pricing page (checked 2026-09-29)
PRICE_INPUT_PER_1M = 0.15
PRICE_OUTPUT_PER_1M = 0.60


def count_calls(resources: dict) -> dict:
    """Wrap the chat and rerank clients so we can report what the run actually used."""
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "rerank_calls": 0}

    create = resources["openai"].chat.completions.create

    def counted_create(*args, **kwargs):
        response = create(*args, **kwargs)
        usage["llm_calls"] += 1
        usage["input_tokens"] += response.usage.prompt_tokens
        usage["output_tokens"] += response.usage.completion_tokens
        return response

    resources["openai"].chat.completions.create = counted_create

    if resources["rerank"] is not None:
        rerank = resources["rerank"].rerank

        def counted_rerank(*args, **kwargs):
            usage["rerank_calls"] += 1
            return rerank(*args, **kwargs)

        resources["rerank"].rerank = counted_rerank
    return usage


def main() -> None:
    warnings.simplefilter("ignore", DeprecationWarning)  # noise from the cohere SDK on Python 3.14
    resources = load_resources()
    usage = count_calls(resources)
    session = new_session()

    for number, text in enumerate(MESSAGES, start=1):
        reply = handle_message(session, text, resources=resources)
        print(f"\n[{number}] user: {text}")
        print(f"    intent: {reply['intent']}")
        print(f"    search query: {reply['search_query'] or '-'}")
        if reply["books"]:
            for book in reply["books"]:
                score = f"{book['match_score']:.3f}" if book["match_score"] is not None else "n/a"
                print(f"    {book['rank']}. match {score}  {book['title']} by {book['author']}")
        else:
            print(f"    reply: {reply['text']}")

    cost = (
        usage["input_tokens"] / 1_000_000 * PRICE_INPUT_PER_1M
        + usage["output_tokens"] / 1_000_000 * PRICE_OUTPUT_PER_1M
    )
    print(
        f"\nusage: {usage['llm_calls']} LLM calls, {usage['input_tokens']:,} input + "
        f"{usage['output_tokens']:,} output tokens (about ${cost:.4f}), {usage['rerank_calls']} Cohere rerank calls"
    )


if __name__ == "__main__":
    main()
