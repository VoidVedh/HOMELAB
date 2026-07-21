# Performance Benchmarks & Telemetry Standards

## Benchmark Metrics Tracker

| Model Version | Parameters | Training Loss | Val Perplexity | Tokens / Sec | Inference Latency (ms) | Memory Peak (MB) |
|---|---|---|---|---|---|---|
| Jarvis-0 | ~5.2M | TBD | TBD | TBD | TBD | TBD |

## Telemetry Metrics Definitions
- **Training Loss**: Cross-entropy loss computed on training batches.
- **Validation Perplexity**: $\text{PPL} = \exp(\mathcal{L}_{val})$.
- **Tokens/Sec**: Throughput during training & autoregressive text generation.
- **Inference Latency**: Time to first token (TTFT) and inter-token generation delay (ms).
- **Memory Peak**: Peak RAM / VRAM allocation in Megabytes (MB).
