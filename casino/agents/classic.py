
import numpy as np
from scipy import stats

from casino.agents.base import BanditAgent


class EpsilonGreedy(BanditAgent):

    def __init__(self, n_arms: int, epsilon: float) -> None:
        self._shots = 2 * np.ones((n_arms,))
        self._hits = np.ones((n_arms,))
        self.epsilon = epsilon
        self.n_arms = n_arms

    @property
    def theta(self) -> np.ndarray:
        return self._hits / self._shots

    def draw(self) -> int:
        if np.random.rand() < self.epsilon:  # explore
            return np.random.randint(self.n_arms)
        else:  # exploit
            return np.argmax(
                self.theta
            )

    def reward(self, arm: int) -> None:
        self._hits[arm] += 1


class ThompsonSampling(BanditAgent):

    def __init__(self, n_arms: int) -> None:
        self._shots = 2 * np.ones((n_arms,))
        self._hits = np.ones((n_arms,))
        self.n_arms = n_arms

    @property
    def _misses(self) -> np.ndarray:
        return self._shots - self._hits

    def draw(self) -> int:
        sample = stats.beta.rvs(self._hits, self._misses)
        arm = np.argmax(sample)
        self._shots[arm] += 1
        return arm

    def reward(self, arm: int) -> None:
        self._hits[arm] += 1


class UCB1(BanditAgent):

    def __init__(self, n_arms: int) -> None:
        self.n_arms = n_arms

        self._shots = 2 * np.ones((n_arms,))
        self._hits = np.ones((n_arms,))
        self._timestep = 0

    @property    
    def _uncertainty(self) -> np.array:
        return np.sqrt(
            (2 * np.log(self._timestep + 1)) /
            self._shots
        )

    @property
    def theta(self) -> np.array:
        return self._hits / self._shots

    def draw(self) -> int:
        score = self.theta + self._uncertainty
        arm = np.argmax(score)

        self._shots[arm] += 1
        
        self._timestep += 1

        return arm

    def reward(self, arm: int) -> None:
        self._hits[arm] += 1


class Exp3(BanditAgent):

    def __init__(self, n_arms: int, gamma: float) -> None:
        self.n_arms = n_arms
        self.gamma = gamma

        self.__arms = np.arange((n_arms,))
        self.w = np.ones((n_arms,))

    def probabilities(self):
        return (
            (1 - self.gamma) * (self.w / np.sum(self.w)) +
            self.gamma * (self.gamma / self.n_arms)
        )

    def draw(self) -> int:
        return np.random.choice(self.__arms, p=self.probabilities())

    def reward(self, arm: int) -> None:
        reward_estimate = 1 / self.probabilities()[arm]
        self.w[arm] *= np.exp(self.gamma * reward_estimate / self.n_arms)

