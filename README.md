# HOMELAB: Local AI Research Engine & JARVIS AI Operating System

HOMELAB is a modular, research-grade local AI platform built from scratch in PyTorch for Apple Silicon (macOS). It cleanly decouples a native machine learning engine (**`ai_core`**) from an AI Operating System (**`jarvis_core`**) connected via an Asynchronous Event Bus and standardized Plugin ecosystem.

```
                    USER / SYSTEM
                          │
                   Executive Planner
                          │
          ┌───────────────┼───────────────┐
          │               │               │
     Memory (SQLite)  Event Bus      Plugins
          │               │               │
          └───────────────┼───────────────┘
                          │
                [ ModelRegistry ]
                          │
                [ ai_core Engine ]
            Native PyTorch Transformer
             (Jarvis-0 / Jarvis-1 / ...)
```

---

## 🚀 Key Features & Principles

- **Pure Machine Learning Engine (`ai_core`)**: Zero OS dependencies. Native BPE tokenizer, decoder-only transformer with Causal Self-Attention, RMSNorm, SwiGLU MLP, and Rotary Positional Embeddings (RoPE).
- **Extensible `BaseModel` & `ModelRegistry`**: Load models dynamically (`ModelRegistry.load("jarvis0")`).
- **AI Operating System (`jarvis_core`)**: Goal decomposition, task DAG execution, and multi-agent dispatching.
- **Asynchronous Event Bus (`bus`)**: Decoupled pub/sub event stream (`TaskSubmittedEvent`, `TaskCompletedEvent`, `MemoryStoredEvent`).
- **Standardized Plugin Ecosystem (`plugins/`)**: Single unified interface (`name()`, `description()`, `permissions()`, `run()`).
- **Experience Memory (`jarvis_core.memory`)**: SQLite database logging conversations, decisions, reasons, commands executed, error logs, and fixes.
- **Telemetry & Observability (`observability/`)**: Structured logging, metric tracking, and performance profiling.

---

## 📁 Repository Structure

```
HOMELAB/
├── ai_core/           # Native Machine Learning Engine (PyTorch)
│   ├── tokenizer/     # Subword Byte-Pair Encoding (BPE)
│   ├── transformer/   # BaseModel ABC, JarvisTransformer
│   ├── attention/     # Causal Self-Attention & RoPE
│   ├── layers/        # RMSNorm, SwiGLU MLP
│   ├── trainer/       # AdamW Trainer & Cosine LR Scheduler
│   ├── evaluator/     # Perplexity & Loss Evaluation
│   ├── sampler/       # Temperature, Top-K, Top-P (Nucleus)
│   ├── inference/     # Autoregressive Generation Engine
│   └── registry/      # ModelRegistry dynamic loader
├── jarvis_core/       # AI Operating System Runtime
│   ├── planner/       # Executive Planner
│   └── memory/        # SQLite Experience Memory
├── bus/               # Asynchronous Pub/Sub Event Bus
├── plugins/           # Standardized Plugin Ecosystem (filesystem, terminal)
├── observability/     # Metrics, Logging & Benchmarks
├── config/            # Versioned YAML model configs (jarvis0.yaml, jarvis1.yaml)
├── docs/              # Research documentation & architecture guides
├── scripts/           # Training & evaluation runner scripts
└── tests/             # Pytest test suite
```

---

## ⚡ Quickstart

### 1. Setup Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
PYTHONPATH=. pytest -v tests/
```

### 3. Train Jarvis-0 Mini Baseline Model
```bash
python3 scripts/train_jarvis_0.py
```

---

## 📖 Research Documentation

- [System Architecture](docs/architecture.md)
- [Training Methodology](docs/training.md)
- [Tokenizer Design](docs/tokenizer.md)
- [Dataset Pipeline](docs/datasets.md)
- [Model Evolution Roadmap](docs/roadmap.md)
- [Experiment Research Log](docs/research_log.md)
- [Performance Benchmarks](docs/benchmarks.md)
