# HOMELAB & JARVIS System Architecture

## Overview
HOMELAB is a modular, research-grade platform comprising two cleanly decoupled layers:
1. **`ai_core`**: A native, self-built Machine Learning framework implemented in PyTorch. It contains zero operating system logic and exposes a unified model interface (`BaseModel`).
2. **`jarvis_core`**: An AI Operating System managing task planning, memory, agents, plugins, and execution via an asynchronous Event Bus.

```
+-------------------------------------------------------------------------+
|                              jarvis_core                                |
|   +-----------+  +--------+  +---------------+  +-------------------+   |
|   | Planner   |  | Memory |  | Agent Manager |  | Plugins & Tools   |   |
|   +-----+-----+  +---+----+  +-------+-------+  +---------+---------+   |
+---------|------------|---------------|--------------------|-------------+
          |            |               |                    |
          +------------+-------+-------+--------------------+
                               |
                        [ Async Event Bus ]
                               |
+------------------------------+------------------------------------------+
|                              ai_core                                    |
|   +--------------------+  +-------------------+  +-------------------+  |
|   |  BPETokenizer      |  | ModelRegistry     |  | InferenceEngine   |  |
|   +--------------------+  +-------------------+  +-------------------+  |
|   |  JarvisTransformer |  | Trainer & Loss    |  | LogitSampler      |  |
|   +--------------------+  +-------------------+  +-------------------+  |
+-------------------------------------------------------------------------+
```

## Component Breakdown

### `ai_core` (Neural Network Framework)
- `BaseModel`: Abstract Base Class specifying model lifecycle (`forward`, `generate`, `get_num_params`, `save_checkpoint`, `load_checkpoint`).
- `JarvisTransformer`: Decoder-only transformer architecture with Causal Self-Attention, RMSNorm/LayerNorm, SwiGLU MLP, and Rotary Positional Embeddings (RoPE).
- `ModelRegistry`: Dynamic loader pattern allowing runtime instantiation of models from versioned YAML configs (`ModelRegistry.load("jarvis0")`).
- `InferenceEngine`: Autoregressive generation manager with temperature, Top-K, Top-P (Nucleus) sampling and KV-cache capabilities.
- `Trainer` & `Evaluator`: Complete training pipeline with AdamW optimizer, cosine learning rate scheduler, cross-entropy loss computation, and evaluation metrics (Perplexity, Tokens/sec, Memory usage).

### `jarvis_core` (AI Operating System)
- `ExecutivePlanner`: Task decomposition, DAG building, and dynamic plan execution.
- `ExperienceMemory`: SQLite & vector-backed memory storing conversations, decisions, execution context, modified files, terminal output, error logs, and fixes.
- `AgentManager`: Multi-agent dispatcher managing specialized agents (`CodingAgent`, `TaskAgent`, `ResearchAgent`).

### `bus` (Asynchronous Event Bus)
- Event-driven pub/sub architecture (`bus.EventBus`) ensuring zero direct tight coupling between Planner, Memory, Plugins, and Agents.

### `plugins` (Standardized Plugin System)
- Unified interface (`BasePlugin`):
  - `name()` -> string
  - `description()` -> string
  - `permissions()` -> list[string]
  - `run(action, payload)` -> dict / async result
