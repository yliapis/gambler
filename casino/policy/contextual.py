
import numpy as np
from scipy import stats

from casino.policy.base import ContextualPolicy

from typing import Optional


class LinUCB(ContextualPolicy):
    """
    LinUCB algorithm:
    paper:
        Li et al, 2010: A Contextual-Bandit Approach to
            Personalized News Article Recommendation
        https://arxiv.org/pdf/1003.0146.pdf
    relevant blog post:
        http://john-maxwell.com/post/2017-03-17/

    The current implementation is the disjoint version
    """

    def __init__(
        self,
        n_arms: int,
        alpha: float,
        method="disjoint",  # either disjoint or hybrid
        context_dim: Optional[int]=None
    ) -> None:

        self.n_arms = n_arms
        self._alpha = alpha

        if method not in ("disjoint", "hybrid"):
            raise NotImplementedError(f"method {method} is not implemented")
        # TODO: remove once hybrid method is implemented
        if method is "hybrid":
            raise NotImplementedError("hybrid method still WIP")

        self.method = method

        if context_dim:
            self.__init_algo(context_dim)

    def __init_algo(self, context_dim: int) -> None:

        self._context_dim = context_dim

        self._A = np.zeros((self.n_arms, context_dim, context_dim))
        self._A[:] = np.eye(context_dim)[None, ...]
        self._Ainv = np.zeros((self.n_arms, context_dim, context_dim))
        self._Ainv[:] = np.eye(context_dim)[None, ...]
        
        self._b = np.zeros((self.n_arms, context_dim))

    def _get_scores(self, context: np.array) -> np.array:

        if not hasattr(self, "_context_dim"):
            self.__init_algo(len(context))

        alpha = self._alpha
        x = context
        
        scores = np.array([
            x @ Ainv @ b + alpha * np.sqrt(x @ Ainv @ x)
            for Ainv, b in zip(self._Ainv, self._b)
        ])

        return scores

    def reward(self, arm: int, context: np.array, reward: float=1.0) -> None:
        x = context
        self._b[arm] += reward * x
        self._A[arm] += np.outer(x, x)
        self._Ainv[arm] = np.linalg.inv(self._A[arm])


class NaiveBayes(ContextualPolicy):

    def __init__(
        self,
        n_arms: int,
        context_dim: Optional[int]=None
    ) -> None:

    raise NotImplementedError("WIP")


class AdPredictor(ContextualPolicy):
    """
    Based on Microsoft Research's paper
    paper:
        Graepel et al 2010: Web-Scale Bayesian Click-Through Rate Prediction for Sponsored Search
            Advertising in Microsoft’s Bing Search Engine
        https://www.microsoft.com/en-us/research/wp-content/uploads/
            2010/06/AdPredictor-ICML-2010-final.pdf
    """
    def __init__(
        self,
        n_arms: int,
        context_dim: Optional[int]=None
    ) -> None:

    raise NotImplementedError("WIP")
