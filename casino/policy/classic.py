
import numpy as np
from scipy import stats

from casino.policy.base import Policy


class Random(Policy):

    def __init__(self, n_arms: int) -> None:
        self.n_arms = n_arms

    def _get_scores(self) -> np.array:
        return np.random.rand(self.n_arms)

    def reward(self, arm: int, reward: float=0.0) -> None:
        pass


class EpsilonGreedy(Policy):

    def __init__(self, n_arms: int, epsilon: float=0.01) -> None:
        self._shots = 2 * np.ones((n_arms,))
        self._hits = np.ones((n_arms,))
        self.epsilon = epsilon
        self.n_arms = n_arms

    def _get_scores(self) -> np.ndarray:
        return self._hits / self._shots

    def sample(self) -> int:
        if np.random.rand() < self.epsilon:  # explore
            return np.random.randint(self.n_arms)
        else:  # exploit
            return np.argmax(self._get_scores())

    def sample_k(self, k: int=1) -> int:
        raise NotImplementedError

    def reward(self, arm: int, reward :float=1.0) -> None:
        self._shots[arm] += 1
        self._hits[arm] += reward


class ThompsonSampling(Policy):

    def __init__(self, n_arms: int) -> None:
        self._shots = 2 * np.ones((n_arms,))
        self._hits = np.ones((n_arms,))
        self.n_arms = n_arms

    @property
    def _misses(self) -> np.ndarray:
        return self._shots - self._hits

    def _get_scores(self) -> np.ndarray:
        return stats.beta.rvs(self._hits, self._misses)

    def reward(self, arm: int, reward: float=1.0) -> None:
        self._shots[arm] += 1
        self._hits[arm] += reward


class UCB1(Policy):

    def __init__(self, n_arms: int) -> None:
        self.n_arms = n_arms

        self._shots = 2 * np.ones((n_arms,))
        self._hits = np.ones((n_arms,))
        self._timestep = 0

    def _uncertainty(self) -> np.ndarray:
        return np.sqrt(
            (2 * np.log(self._timestep + 1)) /
            self._shots
        )

    def _get_scores(self) -> np.ndarray:
        return (self._hits / self._shots) + self._uncertainty()

    def reward(self, arm: int, reward: float=1.0) -> None:
        self._shots[arm] += 1
        self._hits[arm] += reward

        self._timestep += 1


class Exp3(Policy):

    def __init__(self, n_arms: int, gamma: float=0.01) -> None:
        self.n_arms = n_arms
        self.gamma = gamma

        self.__arms = np.arange(n_arms)
        self.w = np.ones((n_arms,))

    def _get_scores(self):
        return (
            (1 - self.gamma) * (self.w / np.sum(self.w)) +
            self.gamma * (1 / self.n_arms)
        )

    def sample(self) -> int:
        return np.random.choice(self.__arms, p=self._get_scores())

    def sample_k(self, k: int=1) -> np.ndarray:
        return np.random.choice(
            self.__arms,
            p=self._get_scores(),
            size=(k,),
            replace=False,
        )

    def reward(self, arm: int, reward: float=0.0) -> None:
        if reward:
            reward_estimate = reward / self._get_scores()[arm]
            self.w[arm] *= np.exp(self.gamma * reward_estimate / self.n_arms)

