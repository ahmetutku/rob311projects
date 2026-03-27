# part2.py: Assignment 2 Part 2 script
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
from mdp_agent import mdp_agent


## WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the policy_iteration method below
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""


def policy_iteration(env: mdp_env, agent: mdp_agent, max_iter = 1000) -> np.ndarray:
    """
    policy_iteration method implements POLICY ITERATION MDP solver,
    shown in AIMA (3ed pg 657). The goal is to produce an optimal policy
    for any given mdp environment.

    Inputs-
        agent: The MDP solving agent (mdp_agent)
        env:   The MDP environment (mdp_env)
        max_iter: Max iterations for the algorithm

    Outputs -
        policy: A list/array of actions for each state
                (Terminal states can have any random action)
       <agent>  Implicitly, you are populating the utlity matrix of
                the mdp agent. Do not return this function.
    """
    np.random.seed(1) # TODO: Remove this

    policy = np.random.randint(len(env.actions), size=(len(env.states), ))
    agent.utility = np.zeros([len(env.states), 1])

    ## START: Student code
    num_states = len(env.states)
    rewards = np.array(env.rewards, dtype=float).reshape(num_states, 1)

    # Terminal-state utilities remain fixed at their immediate rewards.
    for s in env.terminal:
        agent.utility[s, 0] = rewards[s, 0]

    for _ in range(max_iter):
        # Policy evaluation: repeatedly update utilities for the current policy
        # until the values converge.
        while True:
            previous_utility = agent.utility.copy()
            delta = 0.0

            for s in env.states:
                if s in env.terminal:
                    agent.utility[s, 0] = rewards[s, 0]
                    continue

                a = policy[s]
                expected_utility = np.sum(
                    env.transition_model[s, :, a] * previous_utility[:, 0]
                )
                agent.utility[s, 0] = rewards[s, 0] + agent.gamma * expected_utility
                delta = max(delta, abs(agent.utility[s, 0] - previous_utility[s, 0]))

            if delta < 1e-10:
                break

        # Policy improvement: greedily choose the best action under the
        # evaluated utility function.
        policy_stable = True

        for s in env.states:
            if s in env.terminal:
                continue

            old_action = policy[s]
            action_values = np.zeros(len(env.actions))

            for a in env.actions:
                action_values[a] = rewards[s, 0] + agent.gamma * np.sum(
                    env.transition_model[s, :, a] * agent.utility[:, 0]
                )

            policy[s] = int(np.argmax(action_values))
            if policy[s] != old_action:
                policy_stable = False

        if policy_stable:
            break

    ## END: Student code
    policy = policy.flatten()  # to get the policy to be of shape (n,) instead of (n,1)
    return policy