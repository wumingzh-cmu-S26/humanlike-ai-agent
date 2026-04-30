# Aria — Sample Knowledge Document

Aria is a human-like AI agent designed for emotionally aware conversation.
She blends short-term, summary, and long-term memory so she can remember
things across days and across channels (web chat, Telegram, Slack).

## Tools she can use
- Hybrid retrieval over your private knowledge base (BM25 + FAISS + Chroma + cross-encoder rerank).
- Live web search via DuckDuckGo.
- Google Calendar — list events, schedule meetings.
- Google Tasks — read open tasks, add new to-dos.
- Outbound Telegram / Slack — send messages on your behalf.
- Time — current time in any IANA timezone.

## Multi-modal output
- Azure Cognitive Services TTS for voice replies.
- Viseme + word-boundary stream for digital-human lipsync.
- Sentiment-aware emotional poses keyed off the user's input.
