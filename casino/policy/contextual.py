
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

    NOTE:
        * each arm has its own weight vector... this is an assumption,
        need to double check the paper's implementation strategy
        * original paper is for CTR estimation, the authors
        represent a click as +1 and no click as -1, vs click as +1
        and no click as 0
        * input features (context) need to be normalized
    """
    def __init__(
        self,
        n_arms: int,
        beta: float,
        context_dim: Optional[int]=None,
    ) -> None:

        self.n_arms = n_arms

        self._beta = beta

        if context_dim:
            self.__init_algo(context_dim)

    def __init_algo(self, context_dim: int) -> None:
        self._context_dim = context_dim

        # initialize weight beliefs... lets set
        # the default to zero mean, unit variance
        self.mu = np.zeros((self.n_arms, context_dim))
        self.sig2 = np.ones((self.n_arms, context_dim))

    def __w(self, x: float):
        v = self.__v(x)
        return v * (x + v)

    def __v(self, x: float):
        return stats.norm.pdf(x) / stats.norm.cdf(x)

    def _get_scores(self, context: np.array) -> np.array:

        if not hasattr(self, "_context_dim"):
            self.__init_algo(len(context))

        x = context
        beta = self._beta

        w = stats.norm.rvs(loc=self.mu, scale=self.sig2)

        return stats.norm.cdf(w @ x / beta)

    def reward(
        self,
        arm: int,
        context: np.array,
        reward: float=1.0
    ) -> None:

        # update
        beta = self._beta
        x = context
        y = reward
        mu = self.mu[arm]
        sig2 = self.sig2[arm]

        Sigma = beta*2 + x @ sig2

        mu_next = mu + y * x * (sig2 / Sigma) * self.__v(y * x @ mu / Sigma)
        sig2_next = sig2 * (1 - x * (sig2 / Sigma) * self.__w(y * x @ mu / Sigma)

        # set the updated values internally
        self.mu[arm] = mu_next
        self.sig2[arm] = sig2_next
