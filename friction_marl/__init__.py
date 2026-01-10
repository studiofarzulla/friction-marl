"""Friction MARL package."""

from friction_marl.envs.resource_allocation import ResourceAllocationEnv
from friction_marl.agents.iql import IQLAgent

__all__ = ["ResourceAllocationEnv", "IQLAgent"]
