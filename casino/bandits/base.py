"""
Base class definitions for bandits
"""

from abc import ABC, abstractmethod


class Bandit:

    @abstractmethod
    def __init__(self, n_arms: int, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def draw(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def reward(self, arm: int) -> None:
        raise NotImplementedError
