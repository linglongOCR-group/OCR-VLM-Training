from __future__ import annotations

from tools.data_management.registry.base import Registry
from tools.data_management.rewards.base import RewardAdapter


def default_reward_registry() -> Registry[RewardAdapter]:
    from tools.data_management.rewards.levenshtein import NormalizedLevenshteinReward

    registry: Registry[RewardAdapter] = Registry()
    reward = NormalizedLevenshteinReward()
    registry.register(reward.name, reward)
    return registry
