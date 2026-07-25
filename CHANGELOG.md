# Changelog: HOMELAB & JARVIS AI Operating System

All notable changes to the HOMELAB local AI research engine and JARVIS AI Operating System are documented in this file.

## [0.11.0] - 2026-07-26
### Added
- **Okapi BM25 Sparse Keyword Scorer (`jarvis_core/rag/bm25.py`)**: Built `BM25Scorer` with $k_1=1.5, b=0.75$ term frequency scaling and inverse document frequency (IDF) calculation for exact symbol search.
- **Hybrid Reciprocal Rank Fusion Retriever (`jarvis_core/rag/hybrid_retriever.py`)**: Built `HybridRetriever` fusing sparse BM25 keyword rankings and dense vector similarity scores via Reciprocal Rank Fusion ($k=60$).
- **Cross-Encoder Reranker Module (`jarvis_core/rag/cross_encoder.py`)**: Built `CrossEncoderReranker` performing fine-grained term co-occurrence density scoring, query coverage weighting, and exact symbol matching.
- **Dynamic Context Compressor (`jarvis_core/rag/compressor.py`)**: Built `ContextCompressor` enforcing strict token budget caps ($B_{\text{max}}$) while preserving exact file citations (`[path:L10-L20]`).
- **Benchmark Suite & Test Coverage (`scripts/benchmark_v110_rag.py` & `tests/test_hybrid_rag.py`)**: Created benchmark runner measuring sparse BM25 latency (0.06 ms) and hybrid RRF latency (0.24 ms), bringing Pytest coverage to **59/59 passing tests**.

## [0.10.0] - 2026-07-26
### Added
- **4-bit NormalFloat (NF4) & 8-bit Quantization Engine (`ai_core/quantization/quant_utils.py`)**: Built information-theoretically optimal NF4 quantile lookup mapping and block-wise scaling ($N=64$) routines in pure PyTorch, packing two 4-bit indices per byte (`uint8`) and achieving **7.11x memory compression ratio** with minimal quantile error ($\text{MSE} < 0.009$).
- **Quantized Linear Projection Module (`ai_core/quantization/quantized_linear.py`)**: Implemented `QuantizedLinear` layer storing base weight matrices in packed uint8 format, providing `from_linear()` conversion and on-the-fly dequantization forward projection.
- **QLoRA Fine-Tuning Module (`ai_core/quantization/qlora.py`)**: Built `QLoRALinear` attaching 16-bit trainable low-rank adapters ($A$ and $B$) onto frozen 4-bit quantized base projections, enabling full parameter-efficient fine-tuning with **< 0.8% trainable parameter fraction**.
- **Quantization Benchmark Suite & Test Coverage (`scripts/benchmark_v100_quantization.py` & `tests/test_quantization.py`)**: Created benchmark runner measuring memory footprint compression, MSE, forward latency, and gradient updates, bringing Pytest coverage to **55/55 passing tests**.

## [0.9.0] - 2026-07-26
### Added
- **Grammar-Constrained Tool Sampler (`ai_core/sampler/grammar_sampler.py`)**: Implemented `GrammarConstrainedSampler` and `JSONGrammar` architecture providing dynamic token logit masking during autoregressive sampling, guaranteeing 100% JSON schema validity for tool calls without output syntax errors.
- **Isolated Execution Sandbox Suite (`jarvis_core/sandbox/`)**: Built `ExecutionPolicy`, `ExecutionResult`, `BaseSandbox`, `PythonSandbox`, and `BashSandbox` providing isolated asynchronous subprocess execution with resource caps (10.0s timeouts, SIGKILL cleanup, 100KB buffer limits), stdout/stderr capturing, and command injection defenses.
- **Capabilities Execution Integration (`jarvis_core/capabilities/`)**: Upgraded `CodeCapability` and `TerminalCapability` to execute code and shell commands exclusively through `PythonSandbox` and `BashSandbox` without direct subprocess calls.
- **Enhanced Repair Loop & Telemetry (`jarvis_core/planner/repair_engine.py` & `observability/sandbox_telemetry.py`)**: Upgraded `RepairEngine` with failure classification (`SYNTAX_ERROR`, `IMPORT_ERROR`, `TIMEOUT_EXCEEDED`, `PERMISSION_ERROR`, `RUNTIME_ERROR`) and structured telemetry trace logging under `observability/telemetry/`.
- **Benchmark Suite & Tests (`scripts/benchmark_v090_sandbox.py`)**: Added benchmark script measuring execution latency, process overhead, repair loop status, and JSON grammar validity, bringing Pytest suite coverage to **51/51 passing tests**.

## [0.8.0] - 2026-07-26
### Added
- **Production Dataset Quality Pipeline (`ai_core/datasets/dataset_pipeline.py`)**: Built versioned dataset generator with SHA-256 provenance tracking, quality scoring (0.0–1.0), exact line deduplication, near-duplicate character trigram filtering, and 80/10/10 train/val/test splitters.
- **Curriculum Learning Pipeline v3 (`ai_core/datasets/curriculum.py`)**: Implemented progressive 3-stage curriculum learning pipeline (Stage 1: Syntax & Code, Stage 2: Instruction Following & Tool Use, Stage 3: CoT Reasoning & Needle Retrieval) with difficulty metadata.
- **Gradient Accumulation & ETA Tracking in Trainer (`ai_core/trainer/trainer.py`)**: Upgraded `Trainer` with `grad_accum_steps`, checkpoint resume (`resume_from_checkpoint`), top-K checkpoint pruning (`prune_checkpoints`), and per-step training ETA calculations.
- **Automated Research Experiment Tracker (`ai_core/metrics/experiment_tracker.py`)**: Structured JSON experiment recorder tracking git commit hash, model version, dataset version, hyperparameters, losses, device, RAM, and benchmark results.
- **Multi-Task Capability Evaluation Harness (`ai_core/evaluator/instruction_benchmarks.py`)**: Evaluates HumanEval-lite Pass@1/Pass@5, MBPP-lite accuracy, CoT math/logic reasoning score, and Needle-In-A-Haystack retrieval across 2048-token context windows.
- **Automated Checkpoint Promotion Gate (`ai_core/evaluator/regression_gate.py`)**: Automated promotion gate blocking weaker checkpoints unless validation loss or benchmark scores improve.
- **Updated Test Suite**: Added `tests/test_instruction_curriculum.py` and `tests/test_instruction_benchmarks.py` bringing Pytest coverage to **45/45 passing tests**.

## [0.7.0] - 2026-07-26
### Added
- **Speculative Decoding Engine (`ai_core/inference/speculative.py`)**: Built `SpeculativeDecoder` using Jarvis-0 as a fast draft model to propose $K$ candidate tokens per step, validated in parallel by Jarvis-1 with rejection sampling, achieving **1.53x throughput speedup** (57.41 tokens/sec vs 37.64 tokens/sec) and 75.0% draft acceptance rate while preserving exact output distribution fidelity.
- **Paged KV-Cache Memory Manager (`ai_core/inference/paged_kv.py`)**: Implemented `BlockAllocator` and `PagedKVCache` managing physical memory block tables ($B=16$), eliminating virtual memory fragmentation for long-context sequences.
- **Speculative Benchmark Suite (`scripts/benchmark_speculative.py`)**: Created benchmark runner measuring throughput, latency, draft acceptance rate, and speedup factors.
- **Updated Test Suite**: Added `tests/test_paged_kv.py` and `tests/test_speculative_decoding.py` bringing Pytest coverage to **42/42 passing tests**.

## [0.6.0] - 2026-07-26
### Added
- **Grouped-Query Attention (GQA) & 2048-Context Scaling (`ai_core/attention/multi_head.py` & `ai_core/config/model_config.py`)**: Added `n_kv_heads` support and `repeat_kv` projection to `CausalSelfAttention`, enabling $4\times$ memory footprint reduction during KV-cache decoding and context window expansion to 2048 tokens.
- **Multi-Domain Technical Corpus v2 Pipeline (`ai_core/datasets/multi_domain_corpus.py`)**: Enhanced dataset provenance with line-level deduplication, minimum sequence length filtering, expanded Chain-of-Thought (CoT) reasoning data, and extended JSON manifest statistics (`deduplication_ratio`, `entropy`, `token_count`).
- **Pass@k, CoT Reasoning, and DAG Planning Evaluation Harness (`ai_core/evaluator/domain_benchmarks.py`)**: Upgraded domain evaluation suite with AST Python code syntax validation (`Pass@k`), CoT reasoning step verification (`<thought>`, `<plan>`), and DAG dependency ordering checks.
- **AMP Autocast & Stability Tracking (`ai_core/trainer/trainer.py`)**: Integrated `torch.amp.autocast` for MPS/CPU mixed precision training, per-step gradient norm tracking, and NaN/Inf loss spike protection.
- **Jarvis-1 Architecture Config & End-to-End Runner (`config/jarvis1.yaml` & `scripts/train_jarvis_1.py`)**: Added Jarvis-1 engine configuration (`n_heads=12, n_kv_heads=4, max_seq_len=2048`) and end-to-end training, benchmark evaluation, and regression checking runner.
- **Updated Test Suite (`tests/test_gqa_attention.py`)**: Added unit tests for GQA shape transformations, KV-cache head reduction, and numerical equivalence, bringing Pytest coverage to 38/38 passing tests.

## [0.5.0] - 2026-07-22
### Added
- **14-Domain Technical Corpus & Provenance Tracker (`ai_core/datasets/multi_domain_corpus.py`)**: Built synthetic corpus generator and SHA-256 manifest builders for all 14 research domains (*Programming, Linux, Networking, Cybersecurity, AWS, Docker, Terraform, Kubernetes, SQL, Algorithms, Mathematics, Reasoning, Technical Documentation, System Design*) saved under `datasets/manifests/`.
- **Server-Sent Events (SSE) Streaming Inference (`ai_core/inference/streamer.py` & `backend/main.py`)**: Implemented `TokenStreamer` and `POST /api/v1/infer/stream` endpoint for token-by-token real-time streaming with first-token latency and memory profiling.
- **7-Category Domain Benchmark Harness (`ai_core/evaluator/domain_benchmarks.py`)**: Structured file-backed task evaluation harness for *model, coding, reasoning, planning, tool, rag, and memory* categories.
- **Automated Model Regression Framework (`ai_core/evaluator/regression_framework.py`)**: Automated regression detector comparing current benchmark metrics against historical baseline runs and flagging degradation $>5.0\%$.
- **CI/CD Quality Gate Script (`scripts/ci_regression_gate.py`)**: Automated build gate script failing CI pipelines upon performance regressions.
- **Structured JSON Training Dashboard (`ai_core/trainer/trainer.py`)**: Persisted structured training logs under `training_logs/run_XXX.json` tracking loss, perplexity, lr, grad_norm, and tokens/sec.
- **Updated Test Suite**: Added `tests/test_multi_domain_dataset.py`, `tests/test_streaming_inference.py`, `tests/test_domain_benchmarks.py`, and `tests/test_ci_regression_gate.py` bringing Pytest coverage to 31/31 passing tests.

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
