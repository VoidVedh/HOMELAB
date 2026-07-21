#!/usr/bin/env python3
"""
CLI runner for End-to-End Task Competence & Version Benchmarking.
Usage: python3 scripts/benchmark_tasks.py
"""
import os
import sys
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_core.evaluator.e2e_harness import E2EHarness
from ai_core.evaluator.version_benchmark import VersionBenchmark

def main():
    print("=" * 70)
    print("      JARVIS-2 END-TO-END TASK COMPETENCE & VERSION BENCHMARK")
    print("=" * 70)

    # 1. Run E2E Harness
    print("\n[*] Executing End-to-End Capability Evaluation Harness...")
    harness = E2EHarness()
    results = asyncio.run(harness.run_e2e_eval())

    print(f"[*] Overall Pass Rate: {results['pass_rate'] * 100}% ({results['passed_count']}/{results['total_e2e_tasks']} tasks passed)")
    for detail in results["details"]:
        status = "PASSED" if detail["passed"] else "FAILED"
        print(f"  └─ Task '{detail['goal']}' ({detail['capability']}) -> {status}")

    # 2. Compare Version Metrics
    print("\n[*] Version Benchmark Comparisons:")
    vb = VersionBenchmark()
    comparison = vb.compare_versions()
    for ver, metrics in comparison.items():
        print(f"  • {ver}: Pass Rate={metrics['task_pass_rate']*100}% | Latency={metrics['latency_ms']}ms | Tok/sec={metrics['tok_per_sec']} | Self-Repair={metrics['self_repair_supported']}")

    print("\n[✓] E2E Task Competence & Version Benchmark Completed!")

if __name__ == "__main__":
    main()
