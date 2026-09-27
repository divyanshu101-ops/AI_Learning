# Project 5: Tool-using Agent — Notes

## Problem Statement
Build an assistant that decides **for itself** whether answering a question requires a tool — and if so, which tool and with what arguments — instead of just generating text. This is the first project where the model takes an **action**, not just produces a response.

---

## Full Code

```python
# Cell 1: Imports
import sys
sys.path.append("..")

from common.llm import get_llm
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
```

```python
# Cell 2: Define Tool 1 — Calculator
@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression, e.g. '235 * 18' or '(10 + 5) / 3'.
    Only use this for arithmetic calculations."""
    try:
        result = eval(expression, {"__builtins__": {}})  # restricted eval, no dangerous builtins
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"
```

```python
# Cell 3: Define Tool 2 — Weather (mocked)
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city name."""
    fake_weather_db = {
        "kolkata": "28°C, light rain",
        "delhi": "34°C, clear sky",
        "mumbai": "30°C, humid",
    }
    city_key = city.lower()
    return fake_weather_db.get(city_key, f"No weather data available for {city}")
```

```python
# Cell 4: Tools list + name-to-tool lookup
tools = [calculator, get_weather]
tools_by_name = {t.name: t for t in tools}   # fixed version (dict, not set)
```

```python
# Cell 5: Bind tools to the model
llm = get_llm(temperature=0)
llm_with_tools = llm.bind_tools(tools)
```

```python
# Cell 6: First test — see the raw tool_calls output
response = llm_with_tools.invoke("What is 235 * 18?")

print(response.content)        # often empty when a tool is requested
print(response.tool_calls)     # the model's structured decision appears here
```

```python
# Cell 7: The agent loop — core logic
def run_agent(user_input: str, max_iterations: int = 5):
    messages = [
        SystemMessage(content="You are a helpful assistant. Use tools when needed, otherwise answer directly."),
        HumanMessage(content=user_input),
    ]

    for step in range(max_iterations):
        ai_response = llm_with_tools.invoke(messages)
        messages.append(ai_response)

        if not ai_response.tool_calls:
            return ai_response.content

        for tool_call in ai_response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            print(f"[Agent decided] calling '{tool_name}' with args {tool_args}")

            selected_tool = tools_by_name[tool_name]
            tool_result = selected_tool.invoke(tool_args)

            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))

    return "Max iterations reached without a final answer."
```

```python
# Cell 8: Test — calculation question
print(run_agent("What is 235 * 18?"))
```

```python
# Cell 9: Test — weather question
print(run_agent("What's the weather like in Kolkata right now?"))
```

```python
# Cell 10: Test — no tool needed
print(run_agent("What is the capital of France?"))
```

```python
# Cell 11: Test — multi-step (needs 2 different tools in one query)
print(run_agent("What is 50 * 3, and also what's the weather in Delhi?"))
```

```python
# Cell 12: Inspect the full message history after a run
messages = [
    SystemMessage(content="You are a helpful assistant. Use tools when needed, otherwise answer directly."),
    HumanMessage(content="What is 100 + 250?"),
]

for step in range(5):
    ai_response = llm_with_tools.invoke(messages)
    messages.append(ai_response)
    if not ai_response.tool_calls:
        break
    for tool_call in ai_response.tool_calls:
        result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

for m in messages:
    print(f"[{m.type}] {getattr(m, 'content', '')}")
```

---

## Concept-by-Concept Breakdown

### 1. The core problem this project solves
Every earlier project produced either free text (Projects 1, 2) or a fixed structured object in one shot (Project 3), and Project 4 followed a fixed sequence of steps. Here, for the first time, the model itself decides **whether an external action is needed at all**, and if so, **repeats the decision-action cycle** an unknown number of times until it has enough to answer. This is the defining trait of an agent: the number of steps isn't fixed in advance.

### 2. `@tool` decorator
```python
@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression..."""
```
This decorator turns an ordinary Python function into something LangChain can hand to a model as an available capability. Two things about the function become critical once decorated:
- The **docstring** isn't just documentation for humans anymore — it's sent to the model as the description of what this tool does and when to use it.
- The **type hints** on parameters (`expression: str`) tell the model what shape of arguments it must provide when calling this tool.

### 3. Restricted `eval()`
```python
result = eval(expression, {"__builtins__": {}})
```
`eval()` normally runs a string as if it were Python code. Passing `{"__builtins__": {}}` as the second argument strips out Python's built-in functions (like `open`, `import`, etc.) from what `eval` can access, so a malicious expression can't do more than basic arithmetic. This is a defensive measure — plain `eval()` on unrestricted input is a well-known security risk in real backend systems, since it can execute arbitrary code.

### 4. Mocked data for the weather tool
```python
fake_weather_db = {"kolkata": "28°C, light rain", ...}
```
Instead of wiring up a real weather API (which needs its own API key and setup), a small dictionary stands in as fake "live data." This let the project focus entirely on the **agent decision loop** without adding unrelated complexity. In a real project, only the *inside* of this function would change (an actual API call) — everything else about how the agent uses it stays the same.

### 5. `tools_by_name` — name-to-object lookup
```python
tools_by_name = {t.name: t for t in tools}
```
When the model decides to call a tool, it gives back the tool's **name as a string** (e.g. `"calculator"`), not the actual Python function object. This dictionary comprehension builds a lookup table so that, given that string name, the actual callable tool object can be found and invoked. Note the deliberate mistake caught during the exercise: `{t.name for t in tools}` (no colon) creates a **set** of just the names — useless for lookup, since a set has no way to map a name back to anything. The colon (`{key: value for ...}`) is what makes it a dictionary comprehension instead of a set comprehension.

### 6. `.bind_tools()`
```python
llm_with_tools = llm.bind_tools(tools)
```
This wraps the model so every future call includes information about which tools are available (their names, descriptions, and expected arguments, taken from each tool's docstring and type hints). The model itself was never modified — this creates a new object that behaves like the model but with tool-awareness attached.

### 7. `response.tool_calls`
```python
response.tool_calls
```
When a model decides it needs a tool, its response usually has an empty or minimal `.content`, and the real decision lives in `.tool_calls` — a list of dictionaries, each with:
- `name` — which tool the model wants to call
- `args` — the arguments to call it with (already matched to the tool's expected parameters)
- `id` — a unique identifier for this specific call, used to match the result back to the request later

This is the same idea as Project 3's structured output (a Pydantic schema), except here LangChain built the schema automatically from each tool's docstring and function signature.

### 8. The agent loop, step by step
```python
for step in range(max_iterations):
    ai_response = llm_with_tools.invoke(messages)
    messages.append(ai_response)

    if not ai_response.tool_calls:
        return ai_response.content

    for tool_call in ai_response.tool_calls:
        ...
```
This is the heart of the whole project. Each iteration:
1. Sends the **entire conversation so far** to the model (not just the latest message — same idea as Project 2's memory).
2. Appends the model's response to that conversation, whether it's a tool request or a final answer.
3. Checks: did the model ask for a tool? If **no** (`tool_calls` is empty), the model considered this a final answer — return it and stop the loop.
4. If **yes**, run every requested tool and feed the result(s) back into the conversation, then loop again so the model can see the results and decide what to do next.

### 9. `ToolMessage` — feeding results back
```python
messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))
```
This is a special message type (alongside `SystemMessage`, `HumanMessage`, `AIMessage`) whose job is specifically to carry a tool's output back into the conversation. The `tool_call_id` field links this result to the *exact* tool call that asked for it — critical when a model requests multiple tools at once (like Cell 11's example), so the model can tell which result answers which request.

### 10. `max_iterations` as a safety limit
```python
def run_agent(user_input: str, max_iterations: int = 5):
```
Since the number of loop iterations isn't predetermined (unlike Project 4's fixed 3-step pipeline), there's a real risk of the model getting stuck calling tools indefinitely. Capping the loop at a fixed number of iterations is a simple safeguard against infinite loops or runaway costs — a pattern that shows up in any real system built around an open-ended decision loop.

### 11. Multi-tool, single-turn requests
```python
"What is 50 * 3, and also what's the weather in Delhi?"
```
A single model response can contain **more than one** `tool_call` at once, if the question needs multiple independent pieces of information. The `for tool_call in ai_response.tool_calls:` loop handles all of them before going back to the model, so it can use all the results together in its next response.

---

## Python Concepts Learned

| Concept | Where it appeared | What it does |
|---|---|---|
| Decorators | `@tool` above a function definition | Wraps a function to add extra behavior/metadata without changing its own code |
| Docstrings used programmatically | `"""Evaluate a basic math expression..."""` | Not just documentation — actively read and used by LangChain to describe the tool to the model |
| Restricted execution environment | `eval(expression, {"__builtins__": {}})` | Limiting what a piece of dynamically executed code is allowed to access, for safety |
| Dictionary lookup by string key | `tools_by_name["calculator"]` | Standard way to go from an identifier (string) to the actual object it refers to |
| Dictionary comprehension | `{t.name: t for t in tools}` | Builds a dict in one line from an iterable, mapping each item's attribute to the item itself |
| Set comprehension (the bug version) | `{t.name for t in tools}` | Builds a *set* of just names — visually similar to a dict comprehension but missing the `:` for key-value pairs, and therefore useless here |
| `for` loop with a safety cap | `for step in range(max_iterations):` | Bounding a loop that would otherwise run an unknown/unbounded number of times |
| Early return inside a loop | `if not ai_response.tool_calls: return ai_response.content` | Exiting a function as soon as a stopping condition is met, rather than always running every iteration |
| `getattr()` with a default | `getattr(m, 'content', '')` | Safely reads an attribute that might not exist on every object type, avoiding an `AttributeError` |
| String conversion for safety | `str(tool_result)` | Ensures whatever a tool returns (number, list, etc.) becomes a plain string before being placed in a message |
| Function with a default parameter for configuration | `def run_agent(user_input: str, max_iterations: int = 5):` | Lets callers override a setting (loop limit) without needing to pass it every time |

**Biggest takeaway (beyond LangChain):** this project is about **building a controlled feedback loop around an unreliable decision-maker**. The model's decisions (which tool, what arguments) aren't guaranteed correct, so the surrounding code needs structure to safely execute those decisions, feed results back, and eventually terminate. This exact shape — decide → act → observe → decide again, with a safety limit — appears throughout real systems: retry logic, state machines, orchestration engines, and yes, agent frameworks like LangGraph, which formalizes this loop instead of hand-rolling it with a `for` loop.

---

## How This Project Combines Everything

| Earlier Project | What it taught | Where it shows up here |
|---|---|---|
| Project 1 | `prompt \| llm \| parser`, chains | The system prompt + `llm_with_tools` is still fundamentally a chain-style call |
| Project 2 | Message history, memory | `messages` list carries the full conversation across every loop iteration |
| Project 3 | Structured output (Pydantic schemas) | `tool_calls` — the model's tool-name + arguments decision is structured output, auto-generated from each tool's docstring/type hints |
| Project 4 | Branching (`RunnableBranch`) | `if not ai_response.tool_calls:` is the same "check a condition, take a different path" idea, just written as plain Python instead of a Runnable |

The agent isn't a new, separate concept — it's the first project where all four earlier building blocks are used **together, inside a loop**, with the model itself deciding how many times the loop runs.
