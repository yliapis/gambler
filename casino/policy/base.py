"""
Base class definitions for bandits
"""

import numpy as np

from abc import ABC, abstractmethod

from typing import Optional, Sequence, Union


class Policy(ABC):

    @abstractmethod
    def scores(self) -> Sequence[float]:
        raise NotImplementedError

    def sample(self) -> int:
        return np.argmax(self.scores())

    def sample_k(self, k: int=1) -> Sequence[int]:
        return np.argpartition(-self.scores(), k=k)[:k]

    @abstractmethod
    def reward(self, arm: int, reward: float=1):
        pass
