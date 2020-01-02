
import numpy as np
from scipy import stats

from casino.policy.base import ContextualPolicy

from typing import Optional


class LinUCB(ContextualPolicy):
    """
    LinUCB algorithm
    relevant blog post:
        http://john-maxwell.com/post/2017-03-17/
    """

    def __init__(self, n_arms: int, alpha: float, context_dim: Optional[int]) -> None:

        self.n_arms = n_arms
        self._alpha = alpha

        if context_dim:
            self._init_algo(context_dim)

    def _get_scores(self, context: np.array) -> np.array:

        if not hasattr(self, "context_dim"):
            self._init_algo(len(context))

        alpha = self._alpha
        x = context
        
        scores = np.array([
            x @ Ainv @ b + alpha * np.sqrt(x @ Ainv @ x)
            for Ainv, b in zip(self._Ainv, self._b)
        ])

        return scores

    def _init_algo(self, context_dim: int) -> None:

        self.context_dim = context_dim

        self._A = np.zeros((self.n_arms, context_dim, context_dim))
        self._A[:] = np.eye(context_dim)[None, ...]
        self._Ainv = np.zeros((self.n_arms, context_dim, context_dim))
        self._Ainv[:] = np.eye(context_dim)[None, ...]
        
        self._b = np.zeros((self.n_arms, context_dim))

    def reward(self, arm: int, context: np.array, reward: float=1.0) -> None:
        x = context
        self._b[arm] += reward * x
        self._A[arm] += np.outer(x, x)
        self._Ainv[arm] = np.linalg.inv(self._A[arm])
