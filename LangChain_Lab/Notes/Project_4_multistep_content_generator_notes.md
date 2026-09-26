# Project 4: Multi-step Content Generator — Notes

## Problem Statement
Given a topic, generate an **outline**, expand it into a **draft**, then from that single draft produce three different outputs at once — a Tweet, a LinkedIn post, and a summary. Additionally, route content down a different path depending on its length (short content posted as-is, long content gets summarized).

---

## Full Code

```python
# Cell 1: Imports
import sys
sys.path.append("..")

from common.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableBranch
```

```python
# Cell 2: Step 1 — outline chain
outline_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a content strategist. Create a short bullet-point outline (4-6 points) for the given topic."),
    ("human", "{topic}"),
])

outline_chain = outline_prompt | get_llm(0.7) | StrOutputParser()
```

```python
# Cell 3: Step 2 — draft chain (takes the outline as input)
draft_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a writer. Expand the given outline into a well-structured article of 150-200 words."),
    ("human", "{outline}"),
])

draft_chain = draft_prompt | get_llm(0.7) | StrOutputParser()
```

```python
# Cell 4: Carry `topic` forward + connect outline -> draft
def add_outline(data: dict) -> dict:
    outline = outline_chain.invoke({"topic": data["topic"]})
    return {"topic": data["topic"], "outline": outline}

def add_draft(data: dict) -> dict:
    draft = draft_chain.invoke({"outline": data["outline"]})
    return {**data, "draft": draft}

pipeline_step1 = RunnableLambda(add_outline) | RunnableLambda(add_draft)
```

```python
# Cell 5: Test steps 1+2
result = pipeline_step1.invoke({"topic": "Why system design matters for backend engineers"})

print("OUTLINE:\n", result["outline"])
print("\nDRAFT:\n", result["draft"])
```

```python
# Cell 6: Step 3 — three parallel output chains
tweet_prompt = ChatPromptTemplate.from_messages([
    ("system", "Turn the given article into a punchy tweet, under 280 characters. Output ONLY the tweet."),
    ("human", "{draft}"),
])

linkedin_prompt = ChatPromptTemplate.from_messages([
    ("system", "Turn the given article into a professional LinkedIn post (3-4 short paragraphs). Output ONLY the post."),
    ("human", "{draft}"),
])

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the given article in exactly 2 sentences. Output ONLY the summary."),
    ("human", "{draft}"),
])

tweet_chain = tweet_prompt | get_llm(0.7) | StrOutputParser()
linkedin_chain = linkedin_prompt | get_llm(0.7) | StrOutputParser()
summary_chain = summary_prompt | get_llm(0.3) | StrOutputParser()
```

```python
# Cell 7: Combine the three with RunnableParallel
parallel_outputs = RunnableParallel(
    tweet=tweet_chain,
    linkedin_post=linkedin_chain,
    summary=summary_chain,
)
```

```python
# Cell 8: Full pipeline — outline -> draft -> parallel outputs
full_pipeline = pipeline_step1 | RunnableLambda(lambda data: {"draft": data["draft"]}) | parallel_outputs

output = full_pipeline.invoke({"topic": "Why system design matters for backend engineers"})

print("TWEET:\n", output["tweet"])
print("\nLINKEDIN:\n", output["linkedin_post"])
print("\nSUMMARY:\n", output["summary"])
```

```python
# Cell 9: Branching — route based on content length
def is_long(data: dict) -> bool:
    return len(data["draft"].split()) > 100

short_path = RunnableLambda(lambda data: {"final": data["draft"], "note": "posted as-is (short)"})
long_path = RunnableLambda(lambda data: {"final": summary_chain.invoke({"draft": data["draft"]}), "note": "summarized (was long)"})

branch = RunnableBranch(
    (is_long, long_path),
    short_path,   # default, if no condition matches
)
```

```python
# Cell 10: Test branching on the long draft
branch_result = branch.invoke({"draft": result["draft"]})

print(branch_result["note"])
print(branch_result["final"])
```

```python
# Cell 11: Test branching on a short input
short_result = branch.invoke({"draft": "System design helps engineers build scalable systems."})

print(short_result["note"])
print(short_result["final"])
```

```python
# Optional: retry wrapper for flaky parallel calls
tweet_chain_safe = (tweet_prompt | get_llm(0.7) | StrOutputParser()).with_retry(
    stop_after_attempt=3
)
```

---

## Concept-by-Concept Breakdown

### 1. The core problem this project solves
Every earlier project had a single chain: one input dict → one output. Here, for the first time, **one chain's output becomes another chain's input** (outline feeds into draft), and at another point, **one input needs to produce several independent outputs at once** (tweet, LinkedIn, summary — none of which depend on each other, just on the same draft).

### 2. Sequential chaining (multi-step pipelines)
```python
outline = outline_chain.invoke({"topic": ...})
draft = draft_chain.invoke({"outline": outline})
```
This is the simplest form of a pipeline: run one chain, take its result, feed it as input to the next chain. It's the same idea as function composition in plain Python (`f(g(x))`), just applied to LLM chains instead of regular functions.

### 3. `RunnableLambda` — wrapping plain functions
```python
RunnableLambda(add_outline)
```
Normal Python functions don't automatically support `.invoke()` or fit into a `|` pipe. `RunnableLambda` wraps any function so it behaves like every other Runnable in LangChain (prompt, model, parser) — meaning it can be freely chained with `|` alongside them. This is the bridge between "plain Python logic" and "LangChain's pipeline world."

### 4. Carrying data forward through a pipeline
```python
def add_outline(data: dict) -> dict:
    outline = outline_chain.invoke({"topic": data["topic"]})
    return {"topic": data["topic"], "outline": outline}

def add_draft(data: dict) -> dict:
    draft = draft_chain.invoke({"outline": data["outline"]})
    return {**data, "draft": draft}
```
Each step doesn't just return the *new* piece of data — it returns the **old data plus the new piece**, so later steps can still access things from earlier (like the original `topic`, even after the draft has been generated). `{**data, "draft": draft}` is a dictionary unpacking pattern: it copies everything already in `data` and adds/overwrites the `draft` key.

### 5. `RunnableParallel` — running independent chains at once
```python
parallel_outputs = RunnableParallel(
    tweet=tweet_chain,
    linkedin_post=linkedin_chain,
    summary=summary_chain,
)
```
Instead of calling three chains one after another (`.invoke()` three separate times), `RunnableParallel` runs them together and collects the results into a single dictionary, keyed by the names you gave (`tweet`, `linkedin_post`, `summary`). Since none of the three outputs depend on each other — only on the same `draft` — running them in parallel is both correct and more efficient than running them in sequence.

### 6. `RunnableBranch` — conditional routing
```python
branch = RunnableBranch(
    (is_long, long_path),
    short_path,
)
```
This behaves like an if/elif/else, but built out of Runnables instead of plain Python `if` statements. It takes a list of `(condition_function, path_to_take)` pairs, checks each condition function against the input in order, and runs the **first path whose condition returns `True`**. The last argument with no condition (`short_path`) is the default/fallback if nothing else matched.

### 7. Condition functions
```python
def is_long(data: dict) -> bool:
    return len(data["draft"].split()) > 100
```
A condition function for `RunnableBranch` just needs to take the same input the branch receives and return `True`/`False`. Here, it's checking word count — a simple, cheap way to make a routing decision without calling the LLM again.

### 8. Combining sequential + parallel + branch into one flow
```python
full_pipeline = pipeline_step1 | RunnableLambda(lambda data: {"draft": data["draft"]}) | parallel_outputs
```
The `RunnableLambda(lambda data: {"draft": data["draft"]})` step here is a small "reshaping" step — it strips down the full running dictionary (which still has `topic`, `outline`, `draft`) to just `{"draft": ...}`, because `parallel_outputs` only expects a `draft` key for its three sub-chains. This is a common real-world need: **reshaping data between pipeline stages** so each stage only receives what it actually expects.

### 9. Debugging with `response_metadata` and `repr()`
When a chain unexpectedly returns something empty, two useful debugging habits:
- `print(repr(value))` shows the *exact* value (quotes, whitespace, `None` vs `""`) instead of a possibly misleading blank line from `print(value)`.
- Calling the chain *without* the output parser (`prompt | llm`, no `| StrOutputParser()`) returns the full `AIMessage`, whose `.response_metadata` often reveals *why* something went wrong (e.g., a finish reason like `MAX_TOKENS` or a safety filter), which a plain string result would hide.

### 10. Retries for flaky parallel calls
```python
tweet_chain_safe = (tweet_prompt | get_llm(0.7) | StrOutputParser()).with_retry(
    stop_after_attempt=3
)
```
LLM API calls can occasionally fail or return empty/unexpected results for transient reasons (network blip, momentary rate limiting, etc.) — especially noticeable when running several calls in parallel. `.with_retry(stop_after_attempt=3)` wraps a chain so it automatically retries up to a set number of times before giving up, instead of failing (or silently returning bad output) on the first hiccup. This is standard practice for any unreliable external call in a real system, not just LLMs.

---

## Python Concepts Learned

| Concept | Where it appeared | What it does |
|---|---|---|
| Function composition (pipelines) | `outline_chain.invoke(...)` result fed into `draft_chain.invoke(...)` | Output of one operation becomes input of the next, chained together |
| Wrapping functions to fit an interface | `RunnableLambda(add_outline)` | Adapts a plain function so it matches the interface (`invoke`, `|`) used by other objects in the same system |
| Dictionary unpacking (`**`) | `{**data, "draft": draft}` | Copies all key-value pairs from `data` into a new dict, then adds/overrides specific keys |
| Returning enriched state instead of replacing it | `add_outline`, `add_draft` | A common pattern in pipelines: each step adds to the accumulated data rather than discarding what came before |
| Lambda functions | `lambda data: {"draft": data["draft"]}` | A short, unnamed function defined inline, useful for small one-off transformations |
| Named parallel execution | `RunnableParallel(tweet=..., linkedin_post=..., summary=...)` | Keyword arguments used to both run things concurrently and label their results in the output dict |
| Predicate/condition functions | `def is_long(data: dict) -> bool:` | A function whose sole job is to return `True`/`False`, used to drive a decision elsewhere |
| List of tuples for conditional dispatch | `RunnableBranch((is_long, long_path), short_path)` | Represents "if condition A, do X; otherwise do Y" using data (tuples) instead of a written `if/else` |
| `repr()` for debugging | `print(repr(test))` | Shows the precise value (including empty strings vs `None`) instead of a potentially ambiguous plain print |
| Retry wrapping | `.with_retry(stop_after_attempt=3)` | Defensive programming pattern — automatically retries an unreliable operation before treating it as failed |

**Biggest takeaway (beyond LangChain):** this project is about **pipeline design** — breaking a bigger task into stages, deciding which stages must run in order versus which can run independently (parallel), and adding conditional logic (branching) to route data differently based on runtime checks. This is exactly the kind of thinking used in real backend systems: multi-step data pipelines, workflow orchestration, and request routing all rely on the same sequential/parallel/branch building blocks, just usually expressed in different tools (queues, workflow engines, etc.) instead of LangChain's Runnables.

---

## How This Connects to the Agent (Project 5)

An agent is essentially a loop that keeps making a decision: "do I have enough to answer, or do I need to call a tool first?" That decision-and-different-path logic is exactly what `RunnableBranch` demonstrated here on a small scale (short draft vs long draft → different paths). The agent will use the same underlying idea, just looped repeatedly and driven by the model's own decision (from Project 3's structured output) instead of a simple word-count check.

- **Project 1** → the agent's prompt/reasoning chain
- **Project 2** → the agent's memory of past steps
- **Project 3** → the agent's structured tool-call decisions
- **Project 4** → the agent's decision loop/branching *(this project)*
- **Project 5** → all of the above combined into one agent
