"""
Memory module: discrete takeaway memories for the planner in the multi-agent GUI framework.

Pipeline:
  1. precompute_takeaways.py  → discrete_summary.json
  2. experience_memory_planner.py → FAISS index
  3. plan_with_memory.py (used at agent runtime)
"""

from memory.experience_memory_planner import ExperienceMemorySimple
from memory.plan_with_memory import (
    extract_history_context,
    generate_plan_with_memory,
    update_plan_with_memory,
)
from memory.help_functions import CLIPTextSimilarity, CLIPMultimodalSimilarity

# Legacy: raw-trajectory experience memory (same retrieval mechanism as planner memory)
from memory.experience_memory import ExperienceMemory

__all__ = [
    "ExperienceMemorySimple",
    "ExperienceMemory",
    "generate_plan_with_memory",
    "update_plan_with_memory",
    "extract_history_context",
    "CLIPTextSimilarity",
    "CLIPMultimodalSimilarity",
]
