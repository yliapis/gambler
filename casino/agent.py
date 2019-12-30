
from typing import Any, Dict, Optional, Tuple

import casino.policy


POLICY_TABLE = {
    "epsilon_greedy": casino.policy.EpsilonGreedy,
    "exp3": casino.policy.Exp3,
    "random": casino.policy.Random,
    "thompson_sampling": casino.policy.ThompsonSampling,
    "ucb1": casino.policy.UCB1,
}


class Agent:

    def __init__(
        self,
        n_arms: int,
        policy: str="thompson_sampling",
        *args: Tuple,
        **kwargs: Dict[str, Any],
        ) -> None:

        self.policy = POLICY_TABLE[policy](n_arms, *args, **kwargs)

        self.last_drawn = -1

        self.n_draws = 0
        self.total_reward = 0

    def draw(self) -> int:
        self.n_draws += 1
        arm = self.policy.sample()
        self.last_drawn = arm
        return arm

    def reward(self, reward: float=0.0, arm: Optional[int]=None) -> None:
        self.total_reward += reward
        if arm is None:
            arm = self.last_drawn
        self.policy.reward(arm, reward)
