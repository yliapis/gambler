
import numpy as np
from scipy import stats

from casino.policy.base import Policy


class EpsilonGreedy(Policy):

    def __init__(self, n_arms: int, epsilon: float) -> None:
        self._shots = 2 * np.ones((n_arms,))
        self._hits = np.ones((n_arms,))
        self.epsilon = epsilon
        self.n_arms = n_arms

    def scores(self) -> np.ndarray:
        return self._hits / self._shots

    def draw(self) -> int:
        if np.random.rand() < self.epsilon:  # explore
            return np.random.randint(self.n_arms)
        else:  # exploit
            return np.argmax(self.scores())

    def reward(self, arm: int) -> None:
        self._hits[arm] += 1


class ThompsonSampling(Policy):

    def __init__(self, n_arms: int) -> None:
        self._shots = 2 * np.ones((n_arms,))
        self._hits = np.ones((n_arms,))
        self.n_arms = n_arms

    @property
    def _misses(self) -> np.ndarray:
        return self._shots - self._hits

    def scores(self):
        return stats.beta.rvs(self._hits, self._misses)

    def draw(self) -> int:
        sample = self.scores()
        arm = np.argmax(sample)
        self._shots[arm] += 1
        return arm

    def reward(self, arm: int) -> None:
        self._hits[arm] += 1


class UCB1(Policy):

    def __init__(self, n_arms: int) -> None:
        self.n_arms = n_arms

        self._shots = 2 * np.ones((n_arms,))
        self._hits = np.ones((n_arms,))
        self._timestep = 0

    def _uncertainty(self) -> np.array:
        return np.sqrt(
            (2 * np.log(self._timestep + 1)) /
            self._shots
        )

    def scores(self) -> np.array:
        return (self._hits / self._shots) + self._uncertainty()

    def draw(self) -> int:
        arm = np.argmax(self.scores())

        self._shots[arm] += 1
        self._timestep += 1

        return arm

    def reward(self, arm: int) -> None:
        self._hits[arm] += 1


class Exp3(Policy):

    def __init__(self, n_arms: int, gamma: float) -> None:
        self.n_arms = n_arms
        self.gamma = gamma

        self.__arms = np.arange((n_arms,))
        self.w = np.ones((n_arms,))

    def scores(self):
        return (
            (1 - self.gamma) * (self.w / np.sum(self.w)) +
            self.gamma * (self.gamma / self.n_arms)
        )

    def draw(self) -> int:
        return np.random.choice(self.__arms, p=self.scores())

    def reward(self, arm: int) -> None:
        reward_estimate = 1 / self.probabilities()[arm]
        self.w[arm] *= np.exp(self.gamma * reward_estimate / self.n_arms)

