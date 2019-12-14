"""
Base class definitions for bandits
"""

import numpy as np

from abc import ABC, abstractmethod

from typing import overload


class Agent(ABC):

    @abstractmethod
    def __init__(self, n_arms: int, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def draw(self, *args, **kwargs) -> int:
        raise NotImplementedError

    @abstractmethod
    def reward(self, arm: int, reward: float, *args, **kwargs) -> None:
        raise NotImplementedError


class BanditAgent(Agent):

    @abstractmethod
    def __init__(self, n_arms: int, *args, **kwargs) -> None:
        raise NotImplementedError

    @overload
    @abstractmethod
    def draw(self) -> int:
        raise NotImplementedError

    @overload
    @abstractmethod
    def reward(self, arm: int) -> None:
        raise NotImplementedError


class ContextualBanditAgent(Agent):

    @abstractmethod
    def __init__(self, n_arms: int, *args, **kwargs) -> None:
        raise NotImplementedError

    @overload
    @abstractmethod
    def draw(self, context: np.ndarray) -> int:
        raise NotImplementedError

    @overload
    @abstractmethod
    def reward(self, arm: int, context: np.ndarray) -> None:
        raise NotImplementedError
