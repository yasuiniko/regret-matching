import numpy as np

class BimatrixGame:
    def __init__(self, PlayerClass, m1, m2):
        self.a = PlayerClass(m1)
        self.b = PlayerClass(m2)

    def reset(self, randomize=False):
        self.a.reset("randomize" if randomize else None)
        self.b.reset("randomize" if randomize else None)

    def update(self):
        acta = self.a.decision()
        actb = self.b.decision()
        self.a.update_rule(actb)
        self.b.update_rule(acta) 
    
    def selfplay(self):
        while True:
            self.update()
            yield self.a.p, np.max(self.a.ext_regret), np.max(self.a.int_regret), self.a.average_strategy
    

def chicken(cls):
    m = np.array([[7, 3], [9, 0]]) # chicken game
    return BimatrixGame(cls, m, m)

def shapley(cls):
    m1 = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
    m2 = np.array([[0, 0, 1],[1, 0, 0], [0,1,0]])
    return BimatrixGame(cls, m1, m2)

def matching_pennies(cls):
    m1 = np.array([[1, -1],[-1, 1]])
    m2 = m1 * -1
    return BimatrixGame(cls, m1, m2)