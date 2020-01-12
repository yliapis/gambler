
from typing import Any, Dict, Optional, Tuple

from gambler import policy
from gambler.policy.base import is_contextual


POLICY_TABLE = {
    "epsilon_greedy": policy.EpsilonGreedy,
    "exp3": policy.Exp3,
    "random": policy.Random,
    "thompson_sampling": policy.ThompsonSampling,
    "ucb1": policy.UCB1,
    "linucb": policy.LinUCB,
    "adpredictor": policy.AdPredictor,
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

    def draw(
        self,
        context: Optional[np.ndarray]=None
        ) -> int:

        if is_contextual(self.policy):
            arm = self.policy.sample(context=context)
        else:        
            arm = self.policy.sample()

        self.n_draws += 1
        self.last_drawn = arm

        return arm

    def reward(
        self,
        reward: float=0.0,
        arm: Optional[int]=None,
        context: Optional[np.ndarray]=None
    ) -> None:

        self.total_reward += reward
        if arm is None:
            arm = self.last_drawn

        if is_contextual(self.policy):
            self.policy.reward(arm, reward, context=context)
        else:
            self.policy.reward(arm, reward)
