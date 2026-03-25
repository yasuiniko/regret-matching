import numpy as np


class BimatrixGame:
    """
    Player A is the row player and player B is the column player.
    
    Please input the transposed matrix for player B, so that
    A's matrix is NxM and B's matrix is MxN.
    """

    # TODO: replace with generic class
    def __init__(self, name, PlayerClass, m1:np.ndarray, m2:np.ndarray):
        self.name = name
        self.a = PlayerClass(m1)
        self.b = PlayerClass(m2)

    def reset(self, a_strategy, b_strategy):
        self.a.reset(a_strategy)
        self.b.reset(b_strategy)

    def update(self):
        acta = self.a.decision().copy()
        actb = self.b.decision().copy()
        
        self.a.update_rule(actb)
        self.b.update_rule(acta)

    def selfplay(self, num_iterations=None):
        i = 0
        while num_iterations is None or i <= num_iterations:
            self.update()

            data_a = (
                self.a.p,
                np.max(self.a.ext_regret),
                np.max(self.a.int_regret),
                self.a.average_strategy,
            )
            data_b = (
                self.b.p,
                np.max(self.b.ext_regret),
                np.max(self.b.int_regret),
                self.b.average_strategy,
            )

            yield data_a, data_b

            i += 1


def chicken(cls):
    m1 = np.array([[7, 3], [9, 0]])
    m2 = m1.T
    return BimatrixGame("Chicken", cls, m1, m2)


def shapley(cls):
    m1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    m2 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T
    return BimatrixGame("Shapley", cls, m1, m2)


def matching_pennies(cls):
    m1 = np.array([[1, -1], [-1, 1]])
    m2 = m1.T * -1
    return BimatrixGame("Matching Pennies", cls, m1, m2)


def action_price(cls):
    m1 = np.array([[2, -2, -2], [1, 1, -2], [0, 0, 0]])
    m2 = np.array([[0, -2, -2], [0, 1, -2], [0, 1, 2]]).T
    return BimatrixGame("Action Price", cls, m1, m2)
