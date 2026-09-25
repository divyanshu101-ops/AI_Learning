# Project 1: Smart Rewriter — Notes

## Problem Statement
Build a tool that takes any text and rewrites it in a chosen tone (formal, casual, funny, professional email), without changing its meaning.

---

## Full Code

```python
# Cell 1: Imports
import sys
sys.path.append("..")

from common.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
```

```python
# Cell 2: Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a rewriting assistant. Rewrite the user's text in a {tone} tone. "
     "Keep the original meaning exactly. Use at most {max_words} words. "
     "Output ONLY the rewritten text: no preamble, no explanation, no quotes."),
    ("human", "{text}"),
])
```

```python
# Cell 3: Inspect filled template
sample = "Hey, tomorrow's meeting is cancelled. Please tell everyone."

filled = prompt.invoke({"tone": "formal", "max_words": 30, "text": sample})

for m in filled.to_messages():
    print(f"[{m.type}] {m.content}\n")
```

```python
# Cell 4: Build the chain
llm = get_llm(temperature=0.7)
chain = prompt | llm | StrOutputParser()

chain.invoke({"tone": "formal", "max_words": 30, "text": sample})
```

```python
# Cell 5: Reusable function
def rewrite(text: str, tone: str = "formal", max_words: int = 40, temperature: float = 0.7) -> str:
    chain = prompt | get_llm(temperature) | StrOutputParser()
    return chain.invoke({"text": text, "tone": tone, "max_words": max_words})
```

```python
# Cell 6: Try 4 tones
tones = ["formal", "casual", "funny", "professional email"]

for tone in tones:
    print(f"--- {tone.upper()} ---")
    print(rewrite(sample, tone=tone, max_words=40))
    print()
```

```python
# Cell 7: Temperature experiment
for temp in [0, 0.9]:
    print(f"=== temperature = {temp} ===")
    for i in range(3):
        print(f"{i+1}. {rewrite(sample, tone='funny', max_words=30, temperature=temp)}")
    print()
```

```python
# Cell 8: invoke vs stream
chain = prompt | get_llm(0.7) | StrOutputParser()
inputs = {"tone": "professional email", "max_words": 60, "text": sample}

print("INVOKE:")
print(chain.invoke(inputs))

print("\nSTREAM:")
for chunk in chain.stream(inputs):
    print(chunk, end="", flush=True)
```

```python
# Cell 9: max_words test
for n in [10, 25, 60]:
    out = rewrite(sample, tone="formal", max_words=n)
    print(f"max_words={n} -> actual={len(out.split())} words")
    print(out, "\n")
```

---

## Concept-by-Concept Breakdown

### 1. Chat Model
The model object talks to Gemini. You give it a **list of messages**, and it returns an `AIMessage` — not just plain text. `AIMessage` carries `.content` (the actual text) plus metadata (token usage, finish reason, etc.).

### 2. Messages and Roles
- **System** — defines the model's rules/personality ("you are a rewriter, output only the rewritten text").
- **Human** — the user's actual input for this turn.
- **AI** — the model's past replies, used when tracking conversation history.

Keeping system and human separate makes the model treat the rules more seriously than jamming everything into one message.

### 3. Prompt Template (`ChatPromptTemplate`)
Instead of writing the full prompt every time, you define a **template** with placeholders (`{tone}`, `{text}`, `{max_words}`). It works like an f-string, but for a list of role-tagged messages instead of a single string.

`ChatPromptTemplate.from_messages([(role, content), ...])` lets you define both the system and human message templates together.

### 4. Output Parser (`StrOutputParser`)
The model returns an `AIMessage` object, but you usually just want the plain string. The parser extracts `.content` for you. This becomes more important later (Project 3) when parsers extract structured data (JSON/Pydantic) instead of plain strings.

### 5. Runnables and LCEL (the `|` pipe)
Prompt, model, and parser are all **Runnables** — they share the same interface (`invoke`, `stream`, `batch`). Because the interface is shared, they can be chained together like Lego pieces using `|`.

```
dict {tone, text, max_words}
   → Prompt Template  → messages
   → Chat Model        → AIMessage
   → StrOutputParser   → string
```

This pipeline style is called **LCEL (LangChain Expression Language)**. Once chained, the whole thing becomes a single Runnable — so `chain.invoke()` and `chain.stream()` both work on it directly.

Important: the chain's input must be a **dictionary**, and its keys must match the template's placeholder names exactly, or you'll get an error.

### 6. `invoke` vs `stream`
- **invoke** — waits for the complete response, then returns it all at once.
- **stream** — returns the response in small chunks as they're generated (like watching ChatGPT type word-by-word).

Same final answer either way — just different delivery. Streaming matters when you don't want the user to wait for the full response (e.g., chatbot UIs).

### 7. Temperature
Controls how random the model's word choices are.
- **Low (near 0)** — always picks the most likely next word → consistent, predictable output.
- **High (near 0.9–1)** — occasionally picks less likely words → more varied/creative, but sometimes less focused.

### 8. Length control via prompt instruction (`max_words`)
This is a **prompt-level instruction**, not a hard limit — the model tries to follow it but can go slightly over/under. This is different from a token limit at the API level, which cuts the response off mid-sentence if exceeded.

---

## Python Concepts Learned

| Concept | Where it appeared | What it does |
|---|---|---|
| Dictionaries as structured input | `{"tone": ..., "text": ...}` | Pass multiple named values clearly, instead of positional args |
| Type hints | `def rewrite(text: str, ...) -> str:` | Documents expected types; helps IDE autocomplete/errors |
| Default arguments | `tone: str = "formal"` | Caller can omit params and get sensible defaults |
| Keyword arguments | `rewrite(sample, tone=tone, ...)` | Call functions by name, order-independent, more readable |
| `for` loop over a list | `for tone in tones:` | Repeat an action for every item in a list |
| f-strings | `f"--- {tone.upper()} ---"` | Embed variables/expressions directly into strings |
| String methods | `.upper()` | Built-in string transformations |
| Nested loops | `for temp in [...]: for i in range(3):` | Try every combination of two variables (like a grid search) |
| `range(n)` | `range(3)` | Loop a fixed number of times |
| Generators + `for` | `for chunk in chain.stream(inputs):` | Consume values one at a time as they're produced, instead of loading everything into memory at once |
| `print()` special params | `end=""`, `flush=True` | Control newline behavior and force immediate output (needed for streaming) |
| `.split()` + `len()` | `len(out.split())` | Basic word counting / text processing |
| DRY principle | `rewrite()` function | Wrap repeated logic into a single reusable function |

**Biggest takeaway (beyond LangChain):** this project demonstrates **object composition** — combining small, independent pieces (`prompt`, `llm`, `parser`) through a predictable shared interface (`|`, `.invoke()`) to build a larger system. This is a core software engineering idea that applies directly to backend/system design work, not just LangChain.

---

## How This Connects to the Agent (Project 5)

This project's `prompt | llm | parser` pattern is the foundation for the agent's "thinking" step — the agent also has a system prompt that tells it who it is and what it can do. Every later project builds another piece on top of this one:

- **Project 1** → the agent's prompt/reasoning chain
- **Project 2** → the agent's memory of past steps
- **Project 3** → the agent's structured tool-call decisions
- **Project 4** → the agent's decision loop/branching
- **Project 5** → all of the above combined into one agent
