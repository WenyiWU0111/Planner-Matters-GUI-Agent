"""
Build the planner FAISS index from discrete_summary.json (Step 2 of the memory pipeline).

Run after precompute_takeaways.py. Loads discrete_summary.json, creates text embeddings
via CLIP, builds a FAISS index, and saves it to the given output path for use by the
planner at runtime (plan_with_memory.py).

Also defines ExperienceMemorySimple: simplified memory that builds/loads an index from
discrete_summary.json for planner retrieval (retrieve_similar_tasks).
"""

import argparse
import json
import logging
import os
import sys

import faiss
import numpy as np

# Ensure project root is on path when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.help_functions import CLIPTextSimilarity

logger = logging.getLogger("memory.planner")


class ExperienceMemorySimple:
    """
    Simplified experience memory for the planner: builds an index from discrete_summary.json
    (intent + keywords per task) and provides retrieve_similar_tasks() for plan generation.
    """

    def __init__(self, summary_json_path, faiss_index_path=None):
        self.summary_json_path = summary_json_path
        self.clip_similarity = CLIPTextSimilarity()
        self.memories = []
        self.embeddings = None
        self.faiss_index = None

        if faiss_index_path is None:
            print("Generating new memory index from discrete_summary.json...")
            self._load_summary_data()
            self._create_faiss_index()
            if self.faiss_index is None:
                logger.warning("No memories loaded, FAISS index was not created.")
        else:
            print(f"Loading memory index from {faiss_index_path}...")
            self.load_index(faiss_index_path)

    def _load_summary_data(self):
        """Load all tasks from discrete_summary.json and concatenate intent and keywords."""
        print(f"Loading summary data from: {self.summary_json_path}")
        try:
            with open(self.summary_json_path, "r") as f:
                summary_data = json.load(f)
            for task_id, task_data in summary_data.items():
                intent = task_data.get("intent", "")
                if not intent:
                    logger.info(f"Skipping {task_id} because intent is empty")
                    continue
                keywords = task_data.get("keywords", [])
                keywords_str = " ".join(keywords) if keywords else ""
                combined_text = f"{intent} {keywords_str}".strip()
                self.memories.append({
                    "task_id": task_id,
                    "intent": intent,
                    "keywords": keywords,
                    "combined_text": combined_text,
                    "steps": task_data.get("steps", []),
                    "src": task_data.get("src", ""),
                })
            print(f"Total tasks loaded: {len(self.memories)}")
        except Exception as e:
            logger.error(f"Error loading summary data: {e}")
            print(f"Error loading summary data: {e}")

    def _create_faiss_index(self):
        """Create FAISS index for fast similarity search."""
        print("Creating FAISS index for all memories...")
        if not self.memories:
            print("No memories to create FAISS index for")
            return
        combined_texts = [m["combined_text"] for m in self.memories]
        print("Generating text embeddings...")
        self.embeddings = self.clip_similarity.get_text_embeddings(combined_texts)
        logger.info(f"Created embeddings matrix with shape: {self.embeddings.shape}")
        faiss.normalize_L2(self.embeddings)
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(self.embeddings.astype("float32"))
        print(f"Created FAISS index with {self.faiss_index.ntotal} vectors")

    def retrieve_similar_tasks(self, query_text, similar_num=3):
        """Retrieve similar tasks by text similarity (FAISS)."""
        if not self.memories or self.faiss_index is None:
            logger.info("No memories available for retrieval")
            return []
        query_embedding = self.clip_similarity.get_text_embeddings([query_text])
        faiss.normalize_L2(query_embedding)
        similarities, indices = self.faiss_index.search(
            query_embedding.astype("float32"), similar_num
        )
        selected = []
        for score, idx in zip(similarities[0], indices[0]):
            if idx != -1:
                memory = self.memories[idx].copy()
                memory["similarity_score"] = float(score)
                selected.append(memory)
                logger.info(f"Score: {score:.4f} - Task ID: {memory['task_id']} - Intent: {memory['intent']}")
        return selected

    def save_index(self, filepath):
        """Save FAISS index, embeddings, and memory data."""
        if self.faiss_index is None:
            print("No FAISS index to save")
            return
        out_dir = os.path.dirname(filepath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        faiss.write_index(self.faiss_index, f"{filepath}.faiss")
        if self.embeddings is not None:
            np.save(f"{filepath}.embeddings.npy", self.embeddings)
        memory_data = {
            "memories": self.memories,
            "embeddings_shape": self.embeddings.shape if self.embeddings is not None else None,
        }
        with open(f"{filepath}.json", "w") as f:
            json.dump(memory_data, f, indent=2)
        print(f"Saved FAISS index, embeddings, and memory data to {filepath}")

    def load_index(self, filepath):
        """Load FAISS index, embeddings, and memory data."""
        try:
            self.faiss_index = faiss.read_index(f"{filepath}.faiss")
            embeddings_path = f"{filepath}.embeddings.npy"
            if os.path.exists(embeddings_path):
                self.embeddings = np.load(embeddings_path)
                print(f"Loaded embeddings with shape: {self.embeddings.shape}")
            else:
                print("Embeddings file not found, reconstructing from FAISS index...")
                self.embeddings = self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal)
            with open(f"{filepath}.json", "r") as f:
                memory_data = json.load(f)
            self.memories = memory_data["memories"]
            print(f"Loaded FAISS index and memory data from {filepath}; {self.faiss_index.ntotal} vectors, {len(self.memories)} memories")
        except Exception as e:
            print(f"Error loading index from {filepath}: {e}")
            print("Falling back to creating new index...")
            self._load_summary_data()
            self._create_faiss_index()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build planner FAISS index from discrete_summary.json (Step 2 of memory pipeline)."
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default=os.environ.get("DISCRETE_SUMMARY_PATH", "discrete_summary.json"),
        help="Path to discrete_summary.json from precompute_takeaways.py",
    )
    parser.add_argument(
        "--output_index",
        type=str,
        default=None,
        help="Directory + prefix for saved index (e.g. memory_index/simple_text). "
             "If not set, builds index in memory and does not save (for testing).",
    )
    parser.add_argument(
        "--load_existing",
        type=str,
        default=None,
        help="If set, load existing FAISS index from this path instead of building from summary.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.summary_json):
        print(f"Error: summary file not found: {args.summary_json}")
        print("Run precompute_takeaways.py first (Step 1).")
        sys.exit(1)

    if args.load_existing:
        print(f"Loading existing index from {args.load_existing}...")
        memory = ExperienceMemorySimple(args.summary_json, faiss_index_path=args.load_existing)
        if args.output_index:
            memory.save_index(args.output_index)
            print(f"Re-saved index to {args.output_index}")
    else:
        # Build new index from summary (faiss_index_path=None triggers build only; main() does the save)
        memory = ExperienceMemorySimple(args.summary_json, faiss_index_path=None)
        if args.output_index:
            memory.save_index(args.output_index)
            print(f"Saved index to {args.output_index}")
        elif memory.faiss_index is not None:
            default_path = f"memory_index/simple_text_{memory.faiss_index.ntotal}"
            os.makedirs("memory_index", exist_ok=True)
            memory.save_index(default_path)
            print(f"Saved index to {default_path} (use --output_index to set a custom path).")


if __name__ == "__main__":
    main()
