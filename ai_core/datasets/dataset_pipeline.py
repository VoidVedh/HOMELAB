import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

class DatasetQualityPipeline:
    """
    Production-quality dataset pipeline with versioning, SHA-256 manifests, provenance,
    quality scoring, duplicate/near-duplicate filtering, and metadata tracking.
    """

    def __init__(self, version: str = "3.0.0", min_quality_score: float = 0.5):
        self.version = version
        self.min_quality_score = min_quality_score

    @staticmethod
    def compute_sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def calculate_quality_score(sample_text: str, domain: str) -> float:
        """
        Calculates a quality score [0.0 - 1.0] based on length, structure, and readability.
        """
        if not sample_text or len(sample_text.strip()) < 10:
            return 0.0

        score = 0.5
        lines = [line for line in sample_text.splitlines() if line.strip()]
        
        # Reward adequate length & line count
        if len(sample_text) > 50:
            score += 0.2
        if len(lines) > 2:
            score += 0.1

        # Reward domain-specific keywords or structured tags
        if domain in ["Programming", "Code"] and any(kw in sample_text for kw in ["def ", "class ", "return", "import"]):
            score += 0.2
        elif domain == "Reasoning" and any(kw in sample_text for kw in ["<thought>", "<plan>", "Step 1", "Goal:"]):
            score += 0.2
        elif domain == "Tool_Use" and any(kw in sample_text for kw in ["payload", "action", "filesystem"]):
            score += 0.2

        return min(1.0, round(score, 2))

    @staticmethod
    def is_near_duplicate(text: str, seen_trigrams: set) -> bool:
        """Near-duplicate detection using normalized character 3-grams."""
        normalized = "".join(text.lower().split())
        if len(normalized) < 10:
            return False
        trigrams = set(normalized[i:i+3] for i in range(len(normalized)-2))
        if not trigrams:
            return False
        intersection = trigrams.intersection(seen_trigrams)
        jaccard_sim = len(intersection) / max(len(trigrams.union(seen_trigrams)), 1)
        if jaccard_sim > 0.85:
            return True
        seen_trigrams.update(trigrams)
        return False

    def process_raw_samples(
        self,
        raw_samples: List[Dict[str, Any]],
        output_dir: str = "datasets/manifests/v3"
    ) -> Dict[str, Any]:
        processed_samples = []
        exact_seen = set()
        trigram_seen = set()
        
        duplicates_removed = 0
        quality_filtered = 0

        for idx, sample in enumerate(raw_samples):
            text = sample.get("text", "")
            sha256 = self.compute_sha256(text)

            # 1. Exact Duplicate Filtering
            if sha256 in exact_seen:
                duplicates_removed += 1
                continue
            exact_seen.add(sha256)

            # 2. Near-Duplicate Filtering
            if self.is_near_duplicate(text, trigram_seen):
                duplicates_removed += 1
                continue

            domain = sample.get("domain", "General")
            quality_score = self.calculate_quality_score(text, domain)

            # 3. Minimum Quality Threshold
            if quality_score < self.min_quality_score:
                quality_filtered += 1
                continue

            tokens_count = len(text.split())  # Approximate token count

            sample_metadata = {
                "sample_id": f"sample_{self.version.replace('.', '')}_{idx:05d}",
                "dataset_version": self.version,
                "domain": domain,
                "difficulty": sample.get("difficulty", "medium"),
                "quality_score": quality_score,
                "curriculum_stage": sample.get("curriculum_stage", 1),
                "source": sample.get("source", "HOMELAB Internal Synthetic v3"),
                "licence": "MIT",
                "sha256": f"sha256:{sha256}",
                "token_count": tokens_count,
                "text": text
            }
            processed_samples.append(sample_metadata)

        # 4. Train / Validation / Test Split (80% / 10% / 10%)
        total = len(processed_samples)
        train_end = int(total * 0.8)
        val_end = train_end + int(total * 0.1)

        train_split = processed_samples[:train_end]
        val_split = processed_samples[train_end:val_end]
        test_split = processed_samples[val_end:]

        manifest = {
            "dataset_version": self.version,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_raw_samples": len(raw_samples),
            "processed_samples": total,
            "duplicates_removed": duplicates_removed,
            "quality_filtered": quality_filtered,
            "splits": {
                "train_count": len(train_split),
                "val_count": len(val_split),
                "test_count": len(test_split)
            },
            "manifest_checksum": self.compute_sha256(json.dumps([s["sha256"] for s in processed_samples]))
        }

        os.makedirs(output_dir, exist_ok=True)
        manifest_path = os.path.join(output_dir, f"manifest_v{self.version}.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"manifest": manifest, "samples": processed_samples}, f, indent=2)

        return {
            "manifest": manifest,
            "manifest_path": manifest_path,
            "train_split": train_split,
            "val_split": val_split,
            "test_split": test_split
        }
