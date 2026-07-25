import os
from typing import Dict, List, Any
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.datasets.dataset_pipeline import DatasetQualityPipeline

STAGE_1_SAMPLES = [
    {"domain": "Programming", "difficulty": "easy", "text": "def add(a: int, b: int) -> int:\n    \"\"\"Return sum of two integers.\"\"\"\n    return a + b\n"},
    {"domain": "Linux", "difficulty": "easy", "text": "systemctl status jarvis.service\njournalctl -u jarvis --no-pager -n 50\nchmod 755 scripts/deploy.sh\n"},
    {"domain": "SQL", "difficulty": "easy", "text": "SELECT id, name, created_at FROM users WHERE status = 'ACTIVE' ORDER BY id DESC LIMIT 10;\n"},
    {"domain": "APIs", "difficulty": "easy", "text": "curl -X POST http://localhost:8080/api/v1/infer -H 'Content-Type: application/json' -d '{\"prompt\":\"hello\"}'\n"}
]

STAGE_2_SAMPLES = [
    {
        "domain": "Tool_Use",
        "difficulty": "medium",
        "text": "<usr>List files in project directory</usr>\n<bot><action>filesystem:list_dir</action><payload>{\"path\": \".\"}</payload></bot>"
    },
    {
        "domain": "Planning",
        "difficulty": "medium",
        "text": "<usr>Build dataset and run tests</usr>\n<bot><plan>1. Execute dataset pipeline -> 2. Run pytest test suite</plan></bot>"
    },
    {
        "domain": "Terminal_Reasoning",
        "difficulty": "medium",
        "text": "<usr>Check GPU/MPS device status</usr>\n<bot><action>terminal:run_command</action><payload>{\"command\": \"python3 -c 'import torch; print(torch.backends.mps.is_available())'\"}</payload></bot>"
    }
]

STAGE_3_SAMPLES = [
    {
        "domain": "CoT_Reasoning",
        "difficulty": "hard",
        "text": "<usr>Implement topological sort algorithm</usr>\n<bot><thought>\nGoal: Topological sort of DAG.\nStep 1: Compute in-degree for all vertices.\nStep 2: Collect vertices with in-degree 0 in queue.\nStep 3: Process queue and update neighbor in-degrees.\n</thought>\n<execute>\ndef topological_sort(graph):\n    in_degree = {u: 0 for u in graph}\n    for u in graph:\n        for v in graph[u]:\n            in_degree[v] += 1\n    queue = [u for u in in_degree if in_degree[u] == 0]\n    res = []\n    while queue:\n        u = queue.pop(0)\n        res.append(u)\n        for v in graph[u]:\n            in_degree[v] -= 1\n            if in_degree[v] == 0:\n                queue.append(v)\n    return res\n</execute></bot>"
    },
    {
        "domain": "Needle_In_Haystack",
        "difficulty": "hard",
        "text": "# [Technical Document Header]\n" + "Architecture specification text line.\n" * 40 + "\nNEEDLE_KEY: JARVIS_SECRET_TOKEN_98765\n" + "Architecture specification text line.\n" * 40 + "\nQuery: What is the NEEDLE_KEY?\nAnswer: JARVIS_SECRET_TOKEN_98765"
    }
]

class CurriculumV3Pipeline:
    """
    Curriculum Learning Pipeline v3 structuring dataset generation into 3 difficulty stages.
    Stage 1: Syntax, Vocabulary, APIs, Shell, Python, SQL
    Stage 2: Instruction Following, Tool Use, Terminal Reasoning, Planning
    Stage 3: Long-Context Needle Retrieval, Multi-Step CoT Reasoning, Task Decomposition
    """

    def __init__(self, tokenizer: BPETokenizer, version: str = "3.0.0"):
        self.tokenizer = tokenizer
        self.version = version
        self.pipeline = DatasetQualityPipeline(version=version)

    def generate_curriculum_stage(self, stage: int, multiplier: int = 10) -> List[Dict[str, Any]]:
        if stage == 1:
            base_samples = STAGE_1_SAMPLES
        elif stage == 2:
            base_samples = STAGE_2_SAMPLES
        else:
            base_samples = STAGE_3_SAMPLES

        raw_samples = []
        for i in range(multiplier):
            sample = dict(base_samples[i % len(base_samples)])
            sample["curriculum_stage"] = stage
            sample["source"] = f"HOMELAB Curriculum Stage {stage} Synthetic v3"
            raw_samples.append(sample)

        return raw_samples

    def build_full_curriculum(self, output_dir: str = "datasets/manifests/curriculum") -> Dict[str, Any]:
        all_raw = []
        for stage in [1, 2, 3]:
            all_raw.extend(self.generate_curriculum_stage(stage=stage, multiplier=15))

        res = self.pipeline.process_raw_samples(all_raw, output_dir=output_dir)
        return res
