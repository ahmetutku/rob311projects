# mdp_agent.py
#
# --
# Artificial Intelligence
# ROB 311
# Programming Assignment 2

# Teaching Assistant:
# Zi Cong (Daniel) Guo
# zc.guo@mail.utoronto.ca

import numpy as np
from mdp_env import mdp_env

class mdp_agent:
    """
    Attributes -
        gamma:   (float) Discount factor, AIMA (3ed pg 649)
        utility: (np.array) The Utility matrix, AIMA (3ed pg 649)

    """
    def __init__(self, gamma: float):

        # Discount Factor
        assert 0.0 < gamma <= 1.0
        self.gamma = gamma

        # Utility Function
        self.utility = []
