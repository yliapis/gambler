"""
Base class definitions for bandits
"""

import numpy as np

from abc import ABC, abstractmethod


class BanditAgent(ABC):

    @abstractmethod
    def __init__(self, n_arms: int, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def draw(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def reward(self, arm: int) -> None:
        raise NotImplementedError
