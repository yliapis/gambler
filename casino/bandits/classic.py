
import numpy as np
from scipy import stats

from base import Bandit


class EpsilonGreedy(Bandit):

    def __init__(self, n_arms: int, epsilon: float):
        self._shots = 2 * np.ones((n_arms,))
        self._hits = np.ones((n_arms,))
        self.epsilon = epsilon
        self.n_arms = n_arms

    @property
    def theta(self):
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


class ThompsonSampling(Bandit):

    def __init__(self, n_arms: int, epsilon: float):
        self._shots = 2 * np.zeros((n_arms,))
        self._hits = np.zeros((n_arms,))
        self.epsilon = epsilon
        self.n_arms = n_arms

    @property
    def _misses(self):
        return self._shots - self._hits

    def draw(self) -> int:
        sample = stats.beta.rvs(self._hits, self._misses)
        arm = np.argmax(sample)
        self._shots[arm] += 1
        return arm

    def reward(self, arm: int) -> None:
        self._hits[arm] += 1


class UCB(Bandit):

    def __init__(self, n_arms: int, variant: str="UCB1"):

        self.n_arms = n_arms
        self.variant = variant

        self._shots = 2 * np.zeros((n_arms,))
        self._hits = np.zeros((n_arms,))
        self._timestep = 0

    @property    
    def _uncertainty(self) -> np.array:
        if self.variant == "UCB1":
            return np.sqrt(
                (2 * np.log(self._timestep + 1)) /
                self._shots
            )
        else:
            raise NotImplementedError

    @property
    def theta(self):
        return self._hits / self._shots

    def draw(self) -> int:
        score = self.theta + self._uncertainty
        arm = np.argmax(score)

        self._shots[arm] += 1
        
        self._timestep += 1

        return arm

    def reward(self, arm: int) -> None:
        self._hits[arm] += 1
