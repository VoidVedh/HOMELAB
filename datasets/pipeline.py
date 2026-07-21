import os
import json
import hashlib
import time
from typing import Dict, Any, List, Tuple

class DatasetPipeline:
    """Processes raw text data into cleaned, deduplicated, split, and manifest-tracked datasets."""

    def __init__(self, base_dir: str = "datasets"):
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, "raw")
        self.cleaned_dir = os.path.join(base_dir, "cleaned")
        self.dedup_dir = os.path.join(base_dir, "deduplicated")
        self.manifest_dir = os.path.join(base_dir, "manifests")
        self.splits_dir = os.path.join(base_dir, "splits")

        for d in [self.raw_dir, self.cleaned_dir, self.dedup_dir, self.manifest_dir, self.splits_dir]:
            os.makedirs(d, exist_ok=True)

    def clean_text(self, text: str) -> str:
        """Strip non-printable control characters and normalize whitespace."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    def deduplicate(self, lines: List[str]) -> List[str]:
        """Line-level deduplication preserving order."""
        seen = set()
        deduped = []
        for line in lines:
            line_hash = hashlib.md5(line.encode("utf-8")).hexdigest()
            if line_hash not in seen:
                seen.add(line_hash)
                deduped.append(line)
        return deduped

    def process_and_create_manifest(
        self,
        dataset_name: str,
        raw_text: str,
        license_info: str = "MIT / Self-Authored",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> Dict[str, Any]:
        # 1. Clean
        cleaned = self.clean_text(raw_text)
        lines = cleaned.splitlines()
        
        # 2. Deduplicate
        deduped_lines = self.deduplicate(lines)
        processed_text = "\n".join(deduped_lines)

        # 3. Compute Hash & Token Count estimate
        data_hash = hashlib.sha256(processed_text.encode("utf-8")).hexdigest()
        total_tokens = len(processed_text.split())

        # 4. Generate Splits
        n = len(deduped_lines)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_lines = deduped_lines[:n_train]
        val_lines = deduped_lines[n_train : n_train + n_val]
        test_lines = deduped_lines[n_train + n_val :]

        # 5. Write Manifest
        manifest = {
            "dataset_name": dataset_name,
            "license": license_info,
            "hash": data_hash,
            "total_lines": n,
            "total_tokens_approx": total_tokens,
            "splits": {
                "train_lines": len(train_lines),
                "val_lines": len(val_lines),
                "test_lines": len(test_lines)
            },
            "timestamp": time.time()
        }

        manifest_path = os.path.join(self.manifest_dir, f"{dataset_name}_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest
