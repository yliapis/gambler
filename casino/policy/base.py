"""
Base class definitions for bandits

TODO: find a way to marge Policy and ContextualPolicy
to either be the same class, or inherit from the same
class.
"""

import numpy as np

from abc import ABC, abstractmethod


class Policy(ABC):

    @abstractmethod
    def _get_scores(self) -> np.ndarray:
        """
        This is an array of reward scores for each arm. It may either
        be drawn stochastically with each call or be deterministic.
        This is used by the policy to select an arm.

        Returns
        -------
        scores : np.array of float
            array of scores for each arm.
        """
        raise NotImplementedError

    def sample(self) -> int:
        """
        Draw an arm from the multi-arm bandit.

        Returns
        -------
        arm : int
            integer id of arm drawn
        """
        return np.argmax(self._get_scores())

    def sample_k(self, k: int=1) -> np.ndarray:
        """
        Draw multiple arms from the multi-arm bandit.

        Returns
        -------
        arms : array of int
            k arms drawn (random order)
        """
        return np.argpartition(
            -self._get_scores(),
            k=k,
        )[:k]

    @abstractmethod
    def reward(self, arm: int, reward: float=1.0) -> None:
        """
        Set a reward for a given arm. In the case of the bernoulli bandit,
        a miss can be a reward of 0, and a hit can be a reward of 1.

        Parameters
        ----------
        arm : int
            integer id of arm to reward
        reward : float
            size of reward
        """
        raise NotImplementedError


class ContextualPolicy(ABC):

    @abstractmethod
    def _get_scores(self, context: np.ndarray) -> np.ndarray:
        """
        This is an array of reward scores for each arm. It may either
        be drawn stochastically with each call or be deterministic.
        This is used by the policy to select an arm.

        Parameters
        ----------
        context : np.array of float
            feature vector representing the context

        Returns
        -------
        scores : np.array of float
            array of scores for each arm.
        """
        raise NotImplementedError

    def sample(self, context: np.ndarray) -> int:
        """
        Draw an arm from the multi-arm bandit.

        Parameters
        ----------
        context : np.array of float
            feature vector representing the context

        Returns
        -------
        arm : int
            integer id of arm drawn
        """
        return np.argmax(self._get_scores(context))

    def sample_k(self, context: np.ndarray, k: int=1) -> np.ndarray:
        """
        Draw multiple arms from the multi-arm bandit.

        Parameters
        ----------
        context : np.array of float
            feature vector representing the context

        Returns
        -------
        arms : array of int
            k arms drawn (random order)
        """
        return np.argpartition(
            -self._get_scores(context),
            k=k,
        )[:k]

    @abstractmethod
    def reward(self, arm: int, context: np.ndarray, reward: float=1.0) -> None:
        """
        Set a reward for a given arm. In the case of the bernoulli bandit,
        a miss can be a reward of 0, and a hit can be a reward of 1.

        Parameters
        ----------
        arm : int
            integer id of arm to reward
        reward : float
            size of reward
        context : np.array of float
            feature vector representing the context
        """
        raise NotImplementedError
