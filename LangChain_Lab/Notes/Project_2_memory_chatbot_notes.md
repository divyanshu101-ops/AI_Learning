# Project 2: Memory Chatbot — Notes

## Problem Statement
Build a chatbot that remembers previous messages in a conversation, so a follow-up question (like "what's my name?") can be answered correctly based on earlier context.

---

## Full Code

```python
# Cell 1: Imports
import sys
sys.path.append("..")

from common.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
```

```python
# Cell 2: Prompt template with history placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
```

```python
# Cell 3: Base chain (no parser — full AIMessage kept for now)
model = get_llm(temperature=0.7)
chain = prompt | model
```

```python
# Cell 4: Session store
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
```

```python
# Cell 5: Wrap chain with history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
```

```python
# Cell 6: First turn
config = {"configurable": {"session_id": "user_123"}}

response = chain_with_history.invoke(
    {"input": "Hi! My name is Alice."},
    config=config,
)
print(response.content)
```

```python
# Cell 7: Follow-up (tests memory)
response2 = chain_with_history.invoke(
    {"input": "What is my name?"},
    config=config,
)
print(response2.content)
```

```python
# Cell 8: Inspect raw stored history
for m in store["user_123"].messages:
    print(f"[{m.type}] {m.content}")
```

```python
# Cell 9: New session — should NOT know the name
config2 = {"configurable": {"session_id": "user_456"}}

response3 = chain_with_history.invoke(
    {"input": "What is my name?"},
    config=config2,
)
print(response3.content)
```

```python
# Cell 10: Streaming with history
config = {"configurable": {"session_id": "user_123"}}

for chunk in chain_with_history.stream(
    {"input": "Tell me a one-line fun fact about my name."},
    config=config,
):
    print(chunk.content, end="", flush=True)
```

```python
# Cell 11: Confirm streamed reply got added to history too
for m in store["user_123"].messages:
    print(f"[{m.type}] {m.content}")
```

---

## Concept-by-Concept Breakdown

### 1. The core problem this project solves
In Project 1, every `chain.invoke()` call was **independent** — only one `{text}` was sent each time, so the model had no memory of anything said before. Here, the goal is to send the model the **entire conversation so far**, not just the latest message, so it can use earlier context to answer new questions.

### 2. `MessagesPlaceholder`
In Project 1's prompt, roles were fixed one-liners (`system`, `human`). Here, `MessagesPlaceholder("chat_history")` reserves a spot in the prompt for a **variable-length list of messages** — because a conversation can be 1 turn or 50 turns long, and you don't know in advance how many.

The string `"chat_history"` is just a *name/key* for this slot — it must match the key you use later when supplying data (`history_messages_key="chat_history"`).

### 3. Prompt message order matters
```python
[("system", ...), MessagesPlaceholder("chat_history"), ("human", "{input}")]
```
Order is: rules → past context → newest question. This mirrors how a person would read a conversation — instructions first, then what's already been said, then the current ask.

### 4. `InMemoryChatMessageHistory`
A ready-made LangChain class that stores messages in RAM as a list, in LangChain's own message format (`HumanMessage`, `AIMessage` objects, not plain strings). It's "in-memory" meaning it disappears when the Python process/kernel stops — a real app would swap this for a database-backed history class instead, without changing anything else in the code.

### 5. The session store pattern
```python
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
```
This is a **factory function with caching**: given a `session_id`, it either returns an existing history object or creates a new empty one. This is how one chatbot backend can serve many different users/conversations at once, each with completely separate memory — one dictionary, keyed by session, holding many independent histories.

### 6. `RunnableWithMessageHistory`
This wraps the existing `chain` (`prompt | model`) so that history is **automatically read before** each call and **automatically updated after** each call — no manual appending of messages required.

Its key parameters:
- `get_session_history` — the function (from step 5) it calls to fetch/create the right history object.
- `input_messages_key="input"` — tells it which key in your input dict is the "new user message" (must match the prompt's `{input}` placeholder).
- `history_messages_key="chat_history"` — tells it which prompt placeholder to inject the history into (must match `MessagesPlaceholder("chat_history")`).

### 7. The `config` / `session_id` mechanism
```python
config = {"configurable": {"session_id": "user_123"}}
chain_with_history.invoke({"input": "..."}, config=config)
```
`config` is a separate channel from the actual input data — it's metadata about *how* to run the chain, not part of the conversation content itself. `RunnableWithMessageHistory` reads `session_id` out of `config["configurable"]` and passes it to `get_session_history()` to pick the right history object.

**Important:** the key inside `configurable` must be named exactly `session_id`, because that matches the parameter name in `get_session_history(session_id: str)`. Naming it something else (e.g. `thread_id`) breaks the wiring.

### 8. Why two different sessions behave differently
Calling `invoke()` with `session_id: "user_123"` vs `session_id: "user_456"` routes to two completely separate `InMemoryChatMessageHistory` objects inside `store`. This is why `user_456` has no idea about "Alice" — its history list is empty; nothing is shared between sessions unless you explicitly design it that way.

### 9. Streaming still updates history
Even when using `.stream()` instead of `.invoke()`, `RunnableWithMessageHistory` collects all the streamed chunks internally and, once the full response is generated, still adds the complete final message to history — same as a normal invoke would. This is why re-checking `store["user_123"].messages` after streaming still shows the full reply as one `AIMessage`, not scattered chunks.

---

## Python Concepts Learned

| Concept | Where it appeared | What it does |
|---|---|---|
| Dictionaries as a cache/registry | `store = {}` | Map keys (session IDs) to objects (history instances), avoiding recreation |
| "get-or-create" pattern | `if session_id not in store: store[session_id] = ...` | Common pattern for lazy initialization — create only when first needed |
| Function returning an object | `def get_session_history(...) -> InMemoryChatMessageHistory:` | A factory function — takes an identifier, returns the right object instance |
| Type hints for return values | `-> InMemoryChatMessageHistory` | Documents what type a function produces, not just what it accepts |
| Nested dictionaries | `{"configurable": {"session_id": "user_123"}}` | Grouping related config under a sub-key, common in API/config designs |
| Object attribute access | `store["user_123"].messages` | Accessing a property (`.messages`) on a stored object instance |
| `for` loop over objects | `for m in store["user_123"].messages:` | Iterating over a list of custom objects, not just plain values |
| f-strings with attributes | `f"[{m.type}] {m.content}"` | Pulling multiple attributes off an object into one formatted line |
| Reassigning a variable across cells | `config = {...}` used in Cell 6 and again in Cell 10 | Notebook cells share the same variable scope — redefining a variable overwrites it for later cells |

**Biggest takeaway (beyond LangChain):** this project introduces **state management** — the idea that a system needs to remember something *between* calls, and that state has to be scoped correctly (per-session here) so different users/conversations don't leak into each other. This is a fundamental backend/system design concept: session handling, caching by key, and keeping state isolated are patterns that show up constantly outside of LangChain too (web sessions, user auth, multi-tenant systems).

---

## How This Connects to the Agent (Project 5)

An agent doesn't just think once and answer — it goes through multiple steps (think → call a tool → see result → think again). It needs to remember what it already tried and what the results were, exactly like this chatbot remembers what was already said. The `RunnableWithMessageHistory` pattern here is the same shape of problem the agent will face: keeping track of an evolving sequence of messages/actions across multiple steps, scoped to one run.

- **Project 1** → the agent's prompt/reasoning chain
- **Project 2** → the agent's memory of past steps *(this project)*
- **Project 3** → the agent's structured tool-call decisions
- **Project 4** → the agent's decision loop/branching
- **Project 5** → all of the above combined into one agent
