"""Reward functions for VERL GRPO training.

Backward-compatible re-exports from docparse_dataset.
"""

from docparse_dataset.rewards.levenshtein import normalized_levenshtein_reward
from docparse_dataset.runtime.verl_export import compute_score

__all__ = ["compute_score", "normalized_levenshtein_reward"]
