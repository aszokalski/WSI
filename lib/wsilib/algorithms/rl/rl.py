from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env
from typing import TypeVar
from dataclasses import dataclass
from typing import List

ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")
StepType = TypeVar("StepType")


@dataclass(frozen=True)
class Rewards:
    on_success: float
    on_fail: float
    on_nothing: float = 0


ClassicRewards = Rewards(1, 0)
HolePenaltyRewards = Rewards(1, -1)


@dataclass
class AgentRunResult:
    position: int
    reward: float
    terminated: bool
    truncated: bool
    info: dict

    @classmethod
    def from_step(cls, step: StepType):
        return cls(*step)


@dataclass
class AgentMultipleResult:
    runs: List[AgentRunResult]

    def plot(self):
        plt.plot([r.reward for r in self.runs])
        plt.show()


@dataclass
class Agent:
    env: Env
    n_episodes: int
    learning_rate: float
    epsilon: float
    epsilon_decay: float
    discount_factor: float
    rewards: Rewards = ClassicRewards
    min_epsilon: float = 0.1

    def __post_init__(self):
        self._epsilon_decay_rate = 1 / (self.n_episodes * self.epsilon_decay)
        self._q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))

    def _random_action(self):
        return self.env.action_space.sample()

    def get_action(self, obs: ObsType):
        if np.random.random() < self.epsilon:
            return self._random_action()
        return int(np.argmax(self._q_values[obs]))

    def _update_q(
        self, obs: ObsType, action: ActType, reward: float, next_obs: ObsType
    ):
        next_best_q = np.max(self._q_values[next_obs])
        delta = (
            reward + self.discount_factor * next_best_q - self._q_values[obs][action]
        )
        self._q_values[obs][action] += self.learning_rate * delta

    def _update_epsilon(self):
        self.epsilon = max(self.epsilon - self._epsilon_decay_rate, self.min_epsilon)

    def train(self) -> AgentMultipleResult:
        results = []
        for e in range(self.n_episodes):
            res = self._run(training=True)
            results.append(res)
            self._update_epsilon()
        return AgentMultipleResult(results)

    def test(self, n_test_epochs):
        successful_runs = 0
        for i in range(n_test_epochs):
            res = self._run(training=False)
            successful_runs += res.reward == self.rewards.on_success
        return round(successful_runs / n_test_epochs, 2)

    def _run(self, training: bool) -> AgentRunResult:
        start = self.env.reset()
        res = AgentRunResult(start[0], 0, False, False, start[1])
        terminal = False
        while not terminal:
            action = self.get_action(res.position)
            new_res = AgentRunResult.from_step(self.env.step(action))
            reward = self.calculate_reward(new_res)
            if training:
                self._update_q(res.position, action, reward, new_res.position)
            terminal = new_res.terminated or new_res.truncated
            res = new_res
        return res

    def calculate_reward(self, res: AgentRunResult):
        if res.truncated:
            return self.rewards.on_fail
        if res.terminated:
            return self.rewards.on_success if res.reward > 0 else self.rewards.on_fail
        return self.rewards.on_nothing
