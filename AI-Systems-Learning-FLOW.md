# AI-in-Systems Roadmap: ML → DL → Transformers → GenAI → Agentic AI → MCP

A complete path from mathematical foundations to production AI-system integration, built for someone coming from backend engineering who wants to become a system design engineer who can connect AI into real systems.

---

## Stage 1: Math + Classical ML Foundations

### Linear Algebra (vectors, matrices, eigenvalues)
**Takeaway:** You understand that every neural network layer is just a matrix transformation, and eigenvalues explain why some transformations are stable and others blow up.
**Feeds forward into:** Backpropagation, attention mechanisms (which are matrix multiplications), and understanding why certain architectures suffer from vanishing/exploding gradients.

### Probability & Statistics
**Takeaway:** You can reason about model outputs as probability distributions rather than fixed answers — this is the mental model that makes "why does the model sometimes hallucinate" make sense.
**Feeds forward into:** Loss functions (cross-entropy is a probability concept), sampling strategies in LLMs (temperature, top-p), and evaluation metrics.

### Calculus (gradients, partial derivatives)
**Takeaway:** You understand *why* gradient descent works — that a gradient tells you the direction of steepest increase, and you're moving against it to minimize loss.
**Feeds forward into:** Backpropagation, optimizer behavior (Adam, SGD), and debugging training instability.

### Core ML: Linear/Logistic Regression, Gradient Descent
**Takeaway:** These are the simplest possible "learn from data" systems — once you understand them fully, every neural network is just this idea stacked and made nonlinear.
**Feeds forward into:** Every deep learning model is regression at its core; understanding this early makes DL feel like an extension, not a new topic.

### Bias-Variance Tradeoff, Regularization, Train/Test Splits
**Takeaway:** You learn to diagnose *why* a model fails (underfitting vs overfitting) instead of just tweaking things randomly.
**Feeds forward into:** Fine-tuning LLMs without destroying their general ability, and designing evaluation pipelines for agents later.

### Classical Algorithms (decision trees, SVMs, k-NN, clustering)
**Takeaway:** You get intuition for non-neural approaches, which helps you recognize when a full LLM is overkill and a simple classifier would do — a system design skill, not just an ML one.
**Feeds forward into:** Making pragmatic architecture decisions later (not every "AI feature" needs a transformer).

---

## Stage 2: Deep Learning

### Neural Net Basics (forward pass, backprop, activation functions)
**Takeaway:** You can trace exactly how a prediction is made and how error flows backward to update weights — no more "black box" feeling.
**Feeds forward into:** Understanding transformer internals, since transformers are just a specific, very deep neural network.

### Loss Functions & Optimizers (SGD, Adam)
**Takeaway:** You understand the practical knobs that control training speed and stability, and why almost everyone defaults to Adam.
**Feeds forward into:** Fine-tuning LLMs, where choosing the wrong optimizer/learning rate wastes real GPU money.

### CNNs (vision intuition)
**Takeaway:** You learn the idea of "local feature detection + hierarchy," even if you're not doing vision work — it's a template for how specialized architectures beat generic ones.
**Feeds forward into:** Multimodal models (vision-language models) later, if you ever touch image-generating or image-understanding AI.

### RNNs / LSTMs (sequence intuition)
**Takeaway:** You understand *why* sequence models existed before transformers, and specifically why they struggled with long-range dependencies (vanishing gradients over time steps).
**Feeds forward into:** This is what makes attention "click" — attention was invented specifically to fix what LSTMs couldn't do.

### Regularization (dropout, batch norm, weight decay)
**Takeaway:** You learn techniques to keep large networks from memorizing training data instead of generalizing.
**Feeds forward into:** Fine-tuning LLMs on small custom datasets without overfitting — a very real problem when you fine-tune on your own company's data.

### PyTorch Fluency
**Takeaway:** You can implement any of the above from scratch instead of only calling library functions — this is what separates "used an LLM" from "understands an LLM."
**Feeds forward into:** Every subsequent stage — you'll read model code, not just API docs.

---

## Stage 3: Transformers

### Attention Mechanism (the core concept)
**Takeaway:** You understand how a model decides which parts of the input matter most for each output token — this single idea explains 90% of modern AI behavior.
**Feeds forward into:** Literally everything after this point — GenAI, agents, RAG, and MCP tool-selection all build on "the model attends to relevant context."

### Self-Attention vs Cross-Attention, Multi-Head Attention
**Takeaway:** You see how attention scales to capture multiple types of relationships in parallel (multi-head) and how a model can attend across two different sequences (cross-attention, e.g. translating one language to another).
**Feeds forward into:** Understanding RAG (the model cross-attends between your query and retrieved documents).

### Positional Encoding
**Takeaway:** You learn why transformers, despite processing all tokens in parallel, still understand word order.
**Feeds forward into:** Understanding context-window limits and why long documents get truncated or chunked in RAG pipelines.

### Encoder-only (BERT) vs Decoder-only (GPT) vs Encoder-Decoder (T5)
**Takeaway:** You learn to match architecture to task — BERT-style for understanding/classification, GPT-style for generation, T5-style for transformation tasks like translation/summarization.
**Feeds forward into:** Picking the right model family for a production task instead of defaulting to "just use GPT for everything."

### Efficient Attention Variants (FlashAttention, sparse attention)
**Takeaway:** You understand why naive attention is O(n²) and doesn't scale to long contexts, and how modern implementations get around that memory bottleneck.
**Feeds forward into:** Inference serving decisions later (Stage 6) — this is exactly what tools like vLLM optimize under the hood.

### Scaling Laws
**Takeaway:** You learn that model performance improves predictably with more data/parameters/compute — this turns "should we use a bigger model" into a cost/benefit calculation instead of a guess.
**Feeds forward into:** Real-world tradeoff decisions between model size, latency, and hosting cost when you design an AI-backed system.

---

## Stage 4: Generative AI (LLMs)

### Pretraining vs Fine-Tuning vs Instruction-Tuning vs RLHF
**Takeaway:** You understand the pipeline that turns a raw next-token predictor into a helpful assistant — each stage adds a different kind of behavior.
**Feeds forward into:** Knowing when you actually need fine-tuning versus when prompting or RAG solves the problem more cheaply.

### Tokenization, Context Windows, Sampling (temperature, top-k/top-p)
**Takeaway:** You understand the literal unit the model operates on, why prices are per-token, and how sampling settings control creativity vs determinism.
**Feeds forward into:** Cost estimation and latency tuning for any production LLM feature — this is a direct system design input.

### Prompt Engineering Fundamentals
**Takeaway:** You learn to reliably shape model output through structure (roles, examples, constraints) rather than trial and error.
**Feeds forward into:** Agent design, where prompts become the "code" that defines an agent's reasoning behavior.

### Embeddings + Vector Search
**Takeaway:** You understand how meaning gets converted into numbers that can be compared for similarity — the foundation of semantic search.
**Feeds forward into:** RAG and vector databases directly (Stage 5 and 6) — this is the bridge concept.

### Fine-Tuning Approaches: Full Fine-Tune vs LoRA/QLoRA
**Takeaway:** You learn that you rarely need to retrain a whole model — parameter-efficient methods adapt a model cheaply by training small additional weight matrices.
**Feeds forward into:** Practical, budget-aware customization of models for a specific company's data or tone.

### Function/Tool Calling & Structured Output (JSON mode, schema validation)
**Takeaway:** You learn how a model can reliably produce machine-readable output instead of free text, which is what makes it usable inside real software.
**Feeds forward into:** This is the direct technical mechanism behind agentic tool use and MCP — a tool call is just structured output the client executes.

### Evaluation (benchmarks + automated eval pipelines)
**Takeaway:** You learn to measure model/agent quality systematically instead of eyeballing a few outputs — building your own eval set for your own use case.
**Feeds forward into:** Production trust — this is what tells you an agent is safe to ship, and it's a skill interviewers specifically probe for.

### DPO/RLHF (conceptual)
**Takeaway:** You understand why an "aligned" chat model behaves differently (more helpful, more refusal-aware) than a raw pretrained model.
**Feeds forward into:** Realistic expectations when you fine-tune or prompt a model — you know which behaviors come from training, not just prompting.

---

## Stage 5: Agentic AI

### Core Agent Loop (ReAct: reason → act → observe → repeat)
**Takeaway:** You understand the fundamental control loop that turns a static LLM into a system that can take multi-step action toward a goal.
**Feeds forward into:** Every agent framework (LangChain, LangGraph) is an implementation of this loop with more structure — you'll read their code with real understanding.

### RAG (Retrieval-Augmented Generation)
**Takeaway:** You learn how to ground model answers in your own data instead of relying on what the model memorized during training.
**Feeds forward into:** This is the single most common production AI pattern — nearly every "AI feature" in a real company is RAG plus a UI.

### Frameworks: LangChain → LangGraph
**Takeaway:** LangChain teaches you composable building blocks (chains, prompts, tools); LangGraph teaches you to model agent behavior as an explicit state graph instead of an implicit loop.
**Feeds forward into:** LangGraph's graph model maps directly onto how you'll design multi-agent systems and how MCP-connected agents route between tools.

### Memory (short-term vs long-term)
**Takeaway:** You learn how agents maintain context across a conversation (short-term) and across sessions (long-term, often vector-store backed).
**Feeds forward into:** This is architecturally identical to session/state management in backend systems — your existing skills transfer directly here.

### Planning & Multi-Agent Orchestration
**Takeaway:** You learn how complex tasks get decomposed into sub-tasks handled by specialized agents, and how they hand off work to each other.
**Feeds forward into:** This is distributed systems thinking applied to AI — directly related to your task-queue project's orchestration ideas.

### Evaluation/Observability (LangSmith or similar)
**Takeaway:** You learn to trace *why* an agent took a specific action, not just whether the final output was correct.
**Feeds forward into:** Production debugging — without traces, a misbehaving agent is nearly impossible to fix.

---

## Stage 6: Production AI Systems (the backend/system-design layer)

### Vector Databases in Production (HNSW indexing, pgvector vs Pinecone/Weaviate/Qdrant)
**Takeaway:** You learn how similarity search is actually implemented at scale, and the tradeoffs between using Postgres extensions vs dedicated vector DBs.
**Feeds forward into:** Direct system design skill — sharding, indexing, and read/write scaling decisions you already know from databases, now applied to embeddings.

### Inference Serving (vLLM/TGI, batching, KV-cache, quantization)
**Takeaway:** You learn how to actually host a model efficiently — batching requests, reusing computed attention (KV-cache), and shrinking model size (GGUF/int8) without major quality loss.
**Feeds forward into:** This is literally "system design for AI" — throughput, latency, and cost tradeoffs, the same category of problem as your distributed task queue.

### LLMOps (streaming, semantic caching, cost/token tracking, rate limiting, observability)
**Takeaway:** You learn to treat an LLM endpoint like any other production dependency — one with unusual failure modes (hallucination, latency spikes, cost blowouts) that need monitoring.
**Feeds forward into:** This is where your backend engineering instincts (rate limiting, caching, tracing) apply almost unchanged to AI systems.

### Security (prompt injection, tool-call sandboxing, RAG data leakage)
**Takeaway:** You learn the AI-specific attack surface — untrusted text can manipulate an agent's behavior, and tools with real access need the same sandboxing discipline as any untrusted-input system.
**Feeds forward into:** Directly relevant once your agents have real tool access (Stage 5) or MCP servers (below) — this is what makes an agent safe to deploy.

---

## Stage 7: MCP (Model Context Protocol)

### The Problem MCP Solves
**Takeaway:** You understand MCP as a standardization layer — instead of every AI app writing custom integration code for every tool, tools expose a common protocol any MCP-compatible client can use.
**Feeds forward into:** This reframes "connecting AI to systems" as an API design problem you already know how to solve, just with a specific contract.

### Client-Server Architecture (servers expose tools/resources, clients consume them)
**Takeaway:** You learn the actual shape of an MCP integration — a server declares what it can do, and a client (like Claude) discovers and calls those capabilities dynamically.
**Feeds forward into:** This is the same discovery pattern as service registries in distributed systems — familiar territory from a system design angle.

### Transport Mechanics (stdio vs SSE/HTTP)
**Takeaway:** You learn how the client and server actually talk — local process communication (stdio) for local tools, HTTP-based streaming (SSE) for remote/hosted servers.
**Feeds forward into:** Choosing the right transport when you build your own MCP server for a real system (local dev tool vs a hosted API integration).

### Tool Design as API Design (idempotency, error contracts, versioning)
**Takeaway:** You learn that a good MCP tool follows the exact same design discipline as a good REST/gRPC endpoint — clear contracts, predictable errors, safe retries.
**Feeds forward into:** This is where all your backend engineering experience pays off directly — you're not learning a new discipline, you're applying your existing one to a new consumer (an LLM instead of a frontend).

### Building Your Own MCP Server
**Takeaway:** Hands-on, you'll wrap something you already know how to build (a DB query, an internal API) as an MCP tool and connect it to a real client.
**Feeds forward into:** This is the capstone — it's the literal "connect AI within systems" skill you set out to learn, closing the loop from Stage 1's math back to a shippable integration.

---

## Suggested Order of Attack

1. Stages 1–2 in parallel with short daily sessions (they reinforce each other)
2. Stage 3 with hands-on implementation (build a tiny transformer from scratch — this is the highest-leverage single exercise in the whole roadmap)
3. Stage 4–5 project-first: build small LangChain projects as you learn each concept (matches your existing plan)
4. Stage 6 alongside your task-queue/backend work — these concepts map almost one-to-one onto skills you're already building
5. Stage 7 last, as the integration capstone once you have both an agent and solid backend instincts to connect it to
