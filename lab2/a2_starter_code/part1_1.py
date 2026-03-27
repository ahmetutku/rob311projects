# part1_1.py: Assignment 2 Part 1 script
#
# --
# Artificial Intelligence
# ROB 311
# Programming Assignment 2

# Teaching Assistant:
# Zi Cong (Daniel) Guo
# zc.guo@mail.utoronto.ca

import numpy as np
from mdp_cleaning_task import cleaning_env

## WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the method  get_transition_model which creates the
    transition probability matrix for the cleanign robot problem desribed in the
    project document.
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""

def get_transition_model(env: cleaning_env) -> np.ndarray:
    """
    get_transition_model method creates a table of size (SxSxA) that represents the
    probability of the agent going from s1 to s2 while taking action a
    e.g. P[s1,s2,a] = 0.5
    This is the method that will be used by the cleaning environment (described in the
    project document) for populating its transition probability table

    Inputs
    --------------
        env: The cleaning environment

    Outputs
    --------------
        P: Matrix of size (SxSxA) specifying all of the transition probabilities.
    """

    P = np.zeros([len(env.states), len(env.states), len(env.actions)])

    ## START: Student Code
    num_states = len(env.states)

    # Action 0 tries to move left, and action 1 tries to move right.
    intended_step = {
        0: -1,
        1: 1,
    }

    # Transition probabilities for the 1D cleaning robot.
    intended_prob = 0.80
    stay_prob = 0.15
    opposite_prob = 0.05

    for s in env.states:
        # Terminal states are absorbing only in the sense that no more actions
        # should be taken from them; the assignment requires zero outgoing mass.
        if s in env.terminal:
            continue

        for a in env.actions:
            move = intended_step[a]

            # Candidate next states before boundary handling.
            intended_state = s + move
            opposite_state = s - move

            # If a move leaves the chain, that probability mass stays put.
            if intended_state < 0 or intended_state >= num_states:
                intended_state = s
            if opposite_state < 0 or opposite_state >= num_states:
                opposite_state = s

            P[s, intended_state, a] += intended_prob
            P[s, s, a] += stay_prob
            P[s, opposite_state, a] += opposite_prob

    ## END: Student code
    return P
