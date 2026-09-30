"""Decide what a chat message is asking for, in one LLM call.

Returns an intent, a standalone search query rewritten using the conversation
history ("something darker" alone is a useless search; "dark superpower fiction
for adults" is not), and book_ref, the book_id of a book already shown that the
user is asking about ("tell me more about the second one").
"""

import json

from openai import OpenAI

from src.config import LLM_MODEL

INTENTS = ["small_talk", "recommend", "refine", "about_shown_book", "unclear", "off_topic"]

SYSTEM_PROMPT = """You route messages for a book recommendation chat. Read the conversation
history and the new user message, then classify the new message into exactly one intent:

- small_talk: greetings, thanks, chit-chat, questions about what you can do.
- recommend: a request for books with a usable preference (genre, theme, mood, plot, setting,
  audience, or a book/author they liked), that starts a new search.
- refine: changes or narrows the most recent recommendation request ("something darker",
  "shorter ones", "not for kids", "more like the first one"). Only if books were shown before.
- about_shown_book: a question about one specific book already shown in the history
  ("tell me more about the second one", "what is Dune about?" if Dune was shown).
- unclear: wants a book but gives no usable preference ("something good", "any book"),
  or is too ambiguous to act on.
- off_topic: anything not about books or reading (weather, coding help, math, news).

search_query: for recommend and refine only, write ONE standalone search query describing the
content of the books wanted, so it works without the history. For refine, merge the topic of the
previous request with the new constraint. Keep the user's meaning; don't add preferences they
didn't express. For every other intent, use an empty string.

book_ref: for about_shown_book only, the book_id of the book the user means, copied WITHOUT the
square brackets it has in the history (for "[b01975]" return "b01975"). Resolve positions like
"the second one" using the numbered list of the most recent books shown. For every other intent,
or if you can't tell which book, use null.
"""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "route",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": INTENTS},
                "search_query": {"type": "string"},
                "book_ref": {"type": ["string", "null"]},
            },
            "required": ["intent", "search_query", "book_ref"],
            "additionalProperties": False,
        },
    },
}


def route(client: OpenAI, text: str, history_text: str) -> dict:
    """history_text: the compact history built by chat.format_history().
    Returns {"intent", "search_query", "book_ref"}."""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,  # routing should be deterministic, not creative
        response_format=RESPONSE_FORMAT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Conversation history:\n{history_text}\n\nNew user message: {text}"},
        ],
    )
    return json.loads(response.choices[0].message.content)
