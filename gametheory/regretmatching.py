import numpy as np

class RegretMatchingStrategies:
    def __init__(self, payoffs, default_p = None):
        # n = num_actions
        self.n = np.shape(payoffs)[0]
        # default action probabilities 
        if default_p is None:
            self.default_p = np.full(self.n, 1. / self.n)
        else:
            self.default_p = default_p
        # payoffs
        self.payoffs = payoffs
        self._update_p = self.ext_regret_matching
        self.reset()

    def reset(self, default_p=None):
        # default action probabilities 
        if default_p == "randomize":
            p = np.random.rand(self.n)
            self.default_p = p / np.sum(p)
        elif default_p is not None:
            self.default_p = default_p
            
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
        # expected payoff for current strategy
        outcomes = np.outer(self.p, opponent_strategy)
        expected_payoff = np.einsum('ij,ij', outcomes, self.payoffs)

        # get current iteration external regret
        pure_outcomes = np.outer(np.ones(self.n), opponent_strategy)
        pure_payoffs = np.sum(pure_outcomes * self.payoffs, axis=1)
        trex = pure_payoffs - expected_payoff

        # get current iteration internal regret
        outsum = np.vstack([outcomes.sum(axis=1)]*self.payoffs.shape[1])
        o = np.zeros_like(outcomes)
        cond_outcomes = np.divide(outcomes, outsum, out=o, where=outsum!=0)
        trint = cond_outcomes @ self.payoffs - expected_payoff

        return trex, trint

    def decision(self):
        return self.p

    def update_rule(self, opponent_strategy):
        # update time to divide by correct number
        self.t += 1

        # update regret vector/matrix
        trex, trint = self.regret(opponent_strategy)
        self.ext_regret += (trex - self.ext_regret)/self.t
        self.int_regret += (trint - self.int_regret)/self.t

        # update strategy
        self._update_p()
        self.average_strategy += (self.p - self.average_strategy) / self.t
    
    def ext_regret_matching(self):
        self.oldp[:] = self.p
        positive_regret = np.maximum(0, self.ext_regret)
        normalizer = np.sum(positive_regret)
        if normalizer <= 0:
            self.p[:] = self.default_p
        else:
            self.p[:] = positive_regret / normalizer


class RegretMatchingActions(RegretMatchingStrategies):
    def regret(self, opp_act):
        # expected payoff for current strategy
        actual_payoff = self.payoffs[self.action, opp_act]

        # get current iteration external regret
        pure_payoffs = self.payoffs[:, opp_act]
        trex = pure_payoffs - actual_payoff

        # get current iteration internal regret
        swap_payoffs = np.outer(np.ones(self.n), pure_payoffs)
        trint = swap_payoffs - actual_payoff

        return trex, trint

    def decision(self):
        self.action = np.random.choice(self.n, p=self.p)
        return self.action