"""Regret matching in the full information setting."""

import numpy as np


class FullInfoRegretMatchingStrategies:
    def __init__(self, payoffs, default_p=None):
        # n = num_actions
        self.n = np.shape(payoffs)[0]
        # default action probabilities
        if default_p is None:
            self.default_p = np.full(self.n, 1.0 / self.n)
        else:
            self.default_p = default_p

        # payoffs
        self.payoffs = payoffs
        self._update_p = self.ext_regret_matching
        self.reset()

    def reset(self, default_p: np.ndarray | None = None):
        # default action probabilities
        if type(default_p) is str and default_p == "randomize":
            p = np.random.rand(self.n)
            self.default_p = p / np.sum(p)
        elif default_p is not None:
            self.default_p = np.copy(default_p)

        # regret of each expert
        self.ext_regret = np.zeros(self.n)
        self.int_regret = np.zeros((self.n, self.n))

        # current action probabilities
        self.p = np.copy(self.default_p)
        self.oldp = np.zeros_like(self.p)

        # number of iterations
        self.t = 0

        # history of play
        self.average_strategy = np.copy(self.p)

        # last action
        self.action = None

    def regret(self, opponent_strategy):
        # Calculate expected payoffs for all pure actions against the opponent
        # self.payoffs is (N, M), opponent_strategy is (M,) -> pure_payoffs is (N,)
        pure_payoffs = self.payoffs @ opponent_strategy
        expected_payoff = self.p @ pure_payoffs

        # external regret
        tr_ex = pure_payoffs - expected_payoff

        # internal regret
        # Formula: p_i * (payoff of j - payoff of i)
        # Broadcasting creates an NxN matrix where row i, col j is the regret of swapping i -> j
        tr_int = self.p[:, None] * (pure_payoffs[None, :] - pure_payoffs[:, None])

        return tr_ex, tr_int

    def update_rule(self, opponent_strategy):
        self.t += 1

        # update regret vector/matrix
        tr_ex, tr_int = self.regret(opponent_strategy)
        self.ext_regret += (tr_ex - self.ext_regret) / self.t
        self.int_regret += (tr_int - self.int_regret) / self.t

        # update average strategy
        self.average_strategy += (self.p - self.average_strategy) / self.t

        # updated for next round
        self._update_p()

    def decision(self) -> np.ndarray:
        return self.p

    def ext_regret_matching(self):
        self.oldp[:] = self.p
        positive_regret = np.maximum(0, self.ext_regret)
        normalizer = np.sum(positive_regret)
        if normalizer <= 0:
            self.p[:] = self.default_p
        else:
            self.p[:] = positive_regret / normalizer


class FullInfoRegretMatchingActions(FullInfoRegretMatchingStrategies):
    def regret(self, opp_act: np.ndarray):

        # expected payoff for current action
        potential_payoffs = self.payoffs @ opp_act
        actual_payoff = self.action @ potential_payoffs

        # current iteration external regret
        tr_ex = potential_payoffs - actual_payoff

        # current iteration internal regret
        # only the action actually played accumulates internal regret
        tr_int = np.outer(self.action, tr_ex)

        return tr_ex, tr_int

    def decision(self) -> np.ndarray:
        action = np.zeros_like(self.p) 
        action[np.random.choice(self.n, p=self.p)] = 1

        self.action = action
        return self.action
