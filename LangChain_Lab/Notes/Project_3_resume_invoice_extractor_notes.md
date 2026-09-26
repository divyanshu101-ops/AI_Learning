# Project 3: Resume / Invoice Extractor — Notes

## Problem Statement
Take messy, unstructured text (an invoice or resume) and extract it into a **clean, fixed-format object** — with predictable fields — so that another program (database, API) can reliably use it, no matter how the input text is phrased.

---

## Full Code

```python
# Cell 1: Imports
import sys
sys.path.append("..")

from common.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional
```

```python
# Cell 2: Nested schema — one line item
class LineItem(BaseModel):
    name: str = Field(description="Name of the product or service")
    quantity: int = Field(description="Number of units purchased")
    unit_price: float = Field(description="Price per single unit, without currency symbol")
```

```python
# Cell 3: Top-level schema — full invoice
class Invoice(BaseModel):
    invoice_number: str = Field(description="The invoice's unique identifier/number")
    date: str = Field(description="Invoice date in YYYY-MM-DD format")
    billed_to: str = Field(description="Name of the person or company being billed")
    items: list[LineItem] = Field(description="List of all line items in the invoice")
    total: float = Field(description="Total amount due")
    payment_terms: Optional[str] = Field(default=None, description="Payment terms, e.g. 'Net 30', if mentioned")
```

```python
# Cell 4: Model wrapped with structured output
llm = get_llm(temperature=0)
structured_llm = llm.with_structured_output(Invoice)
```

```python
# Cell 5: Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an information extraction assistant. Extract invoice details "
     "from the user's text into the given structure. If a field is not "
     "mentioned in the text, leave it empty or null instead of guessing."),
    ("human", "{text}"),
])

extractor_chain = prompt | structured_llm
```

```python
# Cell 6: Test on sample invoice
sample_invoice = (
    "Invoice #4521, dated March 15 2026. Billed to Rohan Verma. "
    "Items: 2x Wireless Mouse @ $15, 1x Keyboard @ $45. "
    "Total due: $75. Payment terms: Net 30."
)

result = extractor_chain.invoke({"text": sample_invoice})
result
```

```python
# Cell 7: Access fields like a normal Python object
print(result.invoice_number)
print(result.date)
print(result.billed_to)
print(result.total)

for item in result.items:
    print(item.name, item.quantity, item.unit_price)
```

```python
# Cell 8: Convert to dict / JSON
result.model_dump()                 # Python dict
result.model_dump_json(indent=2)    # JSON string, nicely formatted
```

```python
# Cell 9: Test missing-field handling
sample_no_terms = (
    "Invoice #9981, dated Jan 5 2026. Billed to Meera Nair. "
    "Items: 3x Notebook @ $5. Total due: $15."
)

result2 = extractor_chain.invoke({"text": sample_no_terms})
print(result2.payment_terms)   # should be None
```

```python
# Cell 10: .batch() — multiple invoices at once
inputs = [
    {"text": sample_invoice},
    {"text": sample_no_terms},
]

results = extractor_chain.batch(inputs)

for r in results:
    print(r.model_dump_json(indent=2))
    print("---")
```

```python
# Cell 11: Extra — resume schema (reuse same pattern)
class Resume(BaseModel):
    name: str = Field(description="Candidate's full name")
    email: Optional[str] = Field(default=None, description="Candidate's email address")
    skills: list[str] = Field(description="List of skills mentioned")
    years_experience: Optional[int] = Field(default=None, description="Total years of experience, if stated")

resume_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an information extraction assistant. Extract resume details "
     "from the user's text into the given structure. If a field is not "
     "mentioned in the text, leave it empty or null instead of guessing."),
    ("human", "{text}"),
])

resume_llm = get_llm(temperature=0).with_structured_output(Resume)
resume_chain = resume_prompt | resume_llm
```

---

## Concept-by-Concept Breakdown

### 1. The core problem this project solves
In Projects 1 and 2, the output was always **free-form text** — a string that could be phrased however the model liked. Here, the goal flips: no matter how messy or differently-worded the input is, the **output shape must always be the same** — same field names, same types, every time. This is what "structured output" means.

### 2. Pydantic `BaseModel` as a schema
```python
class Invoice(BaseModel):
    invoice_number: str
    ...
```
A Pydantic model is a Python class that defines a **contract**: exactly which fields must exist and what type each one is. When you pass this class to the model via `with_structured_output()`, LangChain converts it into instructions the model can follow, and validates the model's output actually matches the shape — so a mistyped or missing field is caught immediately instead of silently breaking downstream code.

### 3. `Field(description=...)`
```python
name: str = Field(description="Name of the product or service")
```
This isn't just documentation — the description text is actually shown to the model as a hint about what should go in that field. It's like writing a mini-prompt for a single field, which is why clear, specific descriptions (e.g. "date in YYYY-MM-DD format") noticeably improve extraction accuracy.

### 4. Nested schemas
```python
items: list[LineItem] = Field(...)
```
A schema can contain another schema. Here, `Invoice` has a field `items` that isn't a plain type — it's a **list of `LineItem` objects**, where `LineItem` is itself a separate `BaseModel` defined earlier. This lets you represent naturally nested real-world data (an invoice has many items; each item has its own fields) instead of forcing everything into flat fields.

### 5. `Optional[...]` and default values
```python
payment_terms: Optional[str] = Field(default=None, description="...")
```
`Optional[str]` means "this field can be a string, or it can be `None`." Combined with `default=None`, this tells the model (and Pydantic) that it's fine for this field to be missing from the input text — it won't cause a validation error, it'll just come back empty. Without this, a genuinely missing value in the input would force the model to either fail or invent something (hallucinate) to fill the field.

### 6. `with_structured_output()`
```python
structured_llm = llm.with_structured_output(Invoice)
```
This wraps the chat model so that instead of returning an `AIMessage` (like in Projects 1 and 2), invoking it returns a **ready-made instance of your Pydantic class directly** — e.g. `result.total` works immediately, no manual parsing of text into fields required. Internally, it also handles telling the model what schema to produce and validating/re-asking if the output doesn't match.

### 7. Accessing fields like normal Python attributes
```python
result.invoice_number
result.items[0].name
```
Because `result` is a real Pydantic object (not a dict or a string), you access its data with dot notation, exactly like any other Python class instance. Nested access (`result.items` → a list of `LineItem` objects → each has `.name`, `.quantity`) works the same way as any nested object structure in Python.

### 8. `.model_dump()` and `.model_dump_json()`
```python
result.model_dump()               # -> Python dict
result.model_dump_json(indent=2)  # -> formatted JSON string
```
These convert the Pydantic object into a plain dict or a JSON string — useful for storing in a database, sending over an API, or just printing in a readable format. `indent=2` pretty-prints the JSON with 2-space indentation instead of one long line.

### 9. `.batch()`
```python
results = extractor_chain.batch(inputs)
```
Instead of calling `.invoke()` once per invoice in a loop, `.batch()` takes a **list of inputs** and processes all of them, returning a list of results in the same order. This is more efficient than manually looping `.invoke()` calls one at a time, especially as the number of items grows.

### 10. Reusing the extraction pattern for a new schema (Resume)
```python
class Resume(BaseModel):
    ...
resume_chain = resume_prompt | resume_llm
```
The exact same pattern (schema → `with_structured_output` → prompt → chain) is reused for a completely different kind of data. This is the real payoff of learning structured extraction properly: once you understand the pattern, extracting *any* kind of structured data from text becomes "define a new schema, reuse the same 4 lines of wiring" — not starting from scratch each time.

**Note on the intentional bug:** the first draft of Cell 11 mistakenly reused the *invoice* prompt (which told the model to extract invoice details) for the *resume* schema. This is a good reminder that the schema and the prompt's instructions need to match — a correct schema with mismatched instructions will still produce poor or inconsistent results, because the model is still being told the wrong task in plain English even though the output shape is enforced.

---

## Python Concepts Learned

| Concept | Where it appeared | What it does |
|---|---|---|
| Class-based schemas (`BaseModel`) | `class Invoice(BaseModel):` | Define a strict data structure using a class instead of a loose dict |
| Type annotations as validation | `invoice_number: str`, `total: float` | Declares the required type of each field; Pydantic enforces it at runtime, not just for documentation |
| `Optional[X]` from `typing` | `Optional[str]` | Marks a field as "may be `None`" — allows missing data without breaking the schema |
| Default values on typed fields | `Field(default=None, ...)` | Field can be omitted entirely and still fall back to a sensible value |
| Nested classes / composition | `items: list[LineItem]` | One schema can contain a list of another schema — models real-world nested data |
| List type hints | `list[LineItem]`, `list[str]` | Declares not just "a list" but "a list of a specific type of item" |
| Dot notation on custom objects | `result.invoice_number`, `item.quantity` | Standard way to access attributes of any Python object instance, including Pydantic models |
| Object-to-dict/JSON conversion | `.model_dump()`, `.model_dump_json(indent=2)` | Converts a structured object into plain data formats usable elsewhere (databases, APIs, files) |
| Processing a list of inputs at once | `extractor_chain.batch(inputs)` | Passing multiple items together instead of looping one call at a time |
| Reusing a pattern across use cases | `Invoice` schema vs `Resume` schema | Same 3-4 lines of wiring (prompt → structured_llm → chain) applied to a totally different data shape |

**Biggest takeaway (beyond LangChain):** this project is really about **defining contracts for data**. In real backend systems, you constantly need to guarantee that data coming from an unreliable source (user input, an external API, an LLM) matches an exact expected shape before your code trusts it. Pydantic's validation is the same idea used in request/response validation in web frameworks (like FastAPI) — you're not just hoping the data looks right, you're enforcing it.

---

## How This Connects to the Agent (Project 5)

When an agent decides "I need to call the calculator tool with these arguments," that decision is itself a piece of **structured output** — a tool name plus a set of typed arguments, not free-flowing text. The exact mechanism used here (`BaseModel` schema + `with_structured_output`) is what lets an agent's "decision" be something your code can safely act on programmatically, instead of trying to parse meaning out of a sentence like "I think I should use the calculator now."

- **Project 1** → the agent's prompt/reasoning chain
- **Project 2** → the agent's memory of past steps
- **Project 3** → the agent's structured tool-call decisions *(this project)*
- **Project 4** → the agent's decision loop/branching
- **Project 5** → all of the above combined into one agent
