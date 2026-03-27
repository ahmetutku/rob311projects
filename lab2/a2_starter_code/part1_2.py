# part1_2.py: Assignment 2 Part 1 script
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
  - Complete the value_iteration method below
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""

def value_iteration(env: mdp_env, agent: mdp_agent, eps: float, max_iter = 1000) -> np.ndarray:
    """
    value_iteration method implements VALUE ITERATION MDP solver,
    shown in AIMA (3ed pg 653). The goal is to produce an optimal policy
    for any given mdp environment.

    Inputs
    ---------------
        agent: The MDP solving agent (mdp_agent)
        env:   The MDP environment (mdp_env)
        eps:   Max error allowed in the utility of a state
        max_iter: Max iterations for the algorithm

    Outputs
    ---------------
        policy: A list/array of actions for each state
                (Terminal states can have any random action)
       <agent>  Implicitly, you are populating the utlity matrix of
                the mdp agent. Do not return this function.
    """
    policy = np.empty_like(env.states)
    agent.utility = np.zeros([len(env.states), 1])

    ## START: Student code
    num_states = len(env.states)
    num_actions = len(env.actions)
    rewards = np.array(env.rewards, dtype=float).reshape(num_states, 1)

    # Terminal utilities should always stay equal to their immediate rewards.
    for s in env.terminal:
        agent.utility[s, 0] = rewards[s, 0]

    # AIMA stopping rule: stop when the Bellman residual is small enough to
    # guarantee an epsilon-optimal value function.
    if agent.gamma == 1.0:
        threshold = eps
    else:
        threshold = eps * (1.0 - agent.gamma) / agent.gamma

    for _ in range(max_iter):
        previous_utility = agent.utility.copy()
        delta = 0.0

        for s in env.states:
            if s in env.terminal:
                agent.utility[s, 0] = rewards[s, 0]
                continue

            action_values = np.zeros(num_actions)

            # Bellman backup: evaluate each action using the transition model.
            for a in env.actions:
                action_values[a] = np.sum(
                    env.transition_model[s, :, a] * previous_utility[:, 0]
                )

            agent.utility[s, 0] = rewards[s, 0] + agent.gamma * np.max(action_values)
            delta = max(delta, abs(agent.utility[s, 0] - previous_utility[s, 0]))

        if delta < threshold:
            break

    # Extract a greedy policy from the converged utilities.
    for s in env.states:
        if s in env.terminal:
            policy[s] = env.actions[0]
            continue

        action_values = np.zeros(num_actions)
        for a in env.actions:
            action_values[a] = np.sum(
                env.transition_model[s, :, a] * agent.utility[:, 0]
            )

        policy[s] = int(np.argmax(action_values))

    policy = policy.flatten()

    ## END Student code
    return policy