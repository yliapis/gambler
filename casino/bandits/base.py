"""
Base class definitions for bandits
"""

import numpy as np

from abc import ABC, abstractmethod


class GenericBandit(ABC):

    @abstractmethod
    def __init__(self, n_arms: int, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def draw(self, *args, **kwargs) -> int:
        raise NotImplementedError

    @abstractmethod
    def reward(self, arm: int, *args, **kwargs) -> None:
        raise NotImplementedError


class Bandit(GenericBandit):

    @abstractmethod
    def __init__(self, n_arms: int, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def draw(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def reward(self, arm: int) -> None:
        raise NotImplementedError


class ContextualBandit(GenericBandit):

    @abstractmethod
    def __init__(self, n_arms: int, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def draw(self, context: np.ndarray) -> int:
        raise NotImplementedError

    @abstractmethod
    def reward(self, arm: int, context: np.ndarray) -> None:
        raise NotImplementedError
