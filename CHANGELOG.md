# Changelog: HOMELAB & JARVIS AI Operating System

All notable changes to the HOMELAB local AI research engine and JARVIS AI Operating System are documented in this file.

## [0.4.0] - 2026-07-22
### Added
- **KV-Cache Accelerated Autoregressive Inference (`ai_core/attention/` & `ai_core/inference/`)**: Incremental single-token decoding supporting past Key-Value state caches in `CausalSelfAttention`, `TransformerBlock`, and `JarvisTransformer`, achieving up to 526.7 tokens/sec generation throughput.
- **Byte-Level Lossless BPE Tokenizer (`ai_core/tokenizer/`)**: Refactored `BPETokenizer` with UTF-8 byte fallback lookup and pre-tokenization regex splitting, achieving 0.0% unknown token rate ($\% <unk> = 0.0\%$) and 100% roundtrip fidelity.
- **Logit Penalty Sampler (`ai_core/sampler/`)**: Enhanced `LogitSampler` with `repetition_penalty`, `frequency_penalty`, and `presence_penalty` parameters to suppress repetitive text loops.
- **Synthetic Technical Dataset Pipeline (`ai_core/datasets/`)**: Built `SyntheticTechnicalCorpus` for Python code, shell execution sequences, and system design instruction data.
- **Model Intelligence Evaluator (`ai_core/evaluator/`)**: Standardized `ModelIntelligenceEvaluator` measuring perplexity, cross-entropy loss, KV-cache speedup ratios, and zero-unk tokenizer fidelity.
- **Updated Test Suite**: Added `tests/test_kv_cache.py` and `tests/test_model_evaluator.py` bringing Pytest coverage to 27/27 passing tests.

## [0.3.0] - 2026-07-21
### Added
- **Capabilities Framework (`jarvis_core/capabilities/`)**: Standardized `BaseCapability` interface declaring `can_handle()`, `plan()`, `execute()`, `verify()`, and `repair()`. Added `CodeCapability`, `FileCapability`, `TerminalCapability`, and `CapabilityRegistry`.
- **Automated Repair Engine (`jarvis_core/planner/repair_engine.py`)**: Automated error critique and repair loop up to `max_attempts=3`.
- **Structured JSON Telemetry (`observability/structured_logger.py`)**: Execution logger emitting JSON trace logs.
- **End-to-End Task Evaluation & Version Benchmarking (`ai_core/evaluator/`)**: Task competence harness (`E2EHarness`) and version comparative benchmarking (`VersionBenchmark`).
- **CLI Runner**: `scripts/benchmark_tasks.py`.

## [0.2.0] - 2026-07-21
### Added
- **RAG Pipeline (`jarvis_core/rag/`)**: Query rewriter, document chunk retriever, reranker, and token-budget context builder.
- **Continuous Knowledge Indexing (`jarvis_core/knowledge/`)**: Document parser (Markdown, Python AST, JSON, YAML, HTML), overlapping chunker, knowledge indexer, and directory file watcher.
- **Context Manager & Prompt Compiler (`jarvis_core/context/`)**: Token budget manager and structured prompt compiler (`<sys>` / `[RELEVANT_KNOWLEDGE]` / `[SYSTEM_STATE]` / `<usr>` / `<bot>`).
- **State Manager & Tool Selector (`jarvis_core/state/` & `plugins/`)**: Live OS tracker and schema-driven tool argument validator.
- **Reproducible Dataset Pipeline (`datasets/`)**: Text cleaning, line-level deduplication, train/val/test splits, and SHA-256 JSON dataset manifests (`datasets/manifests/`).
- **CLI Runners**: `scripts/build_dataset.py`, `scripts/train.py`, `scripts/evaluate.py`.

## [0.1.0] - 2026-07-21
### Added
- **`ai_core` Machine Learning Engine**: Pure PyTorch implementation of Byte-Pair Encoding subword tokenizer (`BPETokenizer`), decoder-only Transformer (`JarvisTransformer`), Causal Self-Attention, RMSNorm, SwiGLU MLP, Rotary Positional Embeddings (RoPE), `BaseModel`, `ModelRegistry`, `LogitSampler`, `InferenceEngine`, AdamW `Trainer`, and `CheckpointManager`.
- **`jarvis_core` AI Operating System**: Executive Planner (`ExecutivePlanner` & `TaskDAG`), SQLite Experience Memory (`ExperienceMemory`), Event Bus (`bus.EventBus`), and Agent Manager (`AgentManager`).
- **Plugins**: `FilesystemPlugin` and `TerminalPlugin`.
- **Production Backend**: FastAPI REST server (`backend/main.py`).
- **Initial Verification**: Empirical claim verification script `scripts/verify_system.py` and 24 passing Pytest unit/integration tests.
