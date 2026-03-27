
# part3.py
#
# --
# Artificial Intelligence
# ROB 311
# Programming Assignment 2

# Teaching Assistant:
# Zi Cong (Daniel) Guo
# zc.guo@mail.utoronto.ca

"""
 We set up bandit arms with fixed probability distribution of success,
 and receive stochastic rewards from each arm of +1 for success,
 and 0 reward for failure.
"""
import numpy as np

class MAB_agent:
    """
        TODO:
        Implement the get_action and update_state function of an agent such that it
        is able to maximize the reward on the Multi-Armed Bandit (MAB) environment.
    """
    def __init__(self, num_arms=5):
        self.__num_arms = num_arms #private
        ## IMPLEMENTATION
        # Number of times each arm has been selected.
        self.counts = np.zeros(self.__num_arms, dtype=int)

        # Estimated value of each arm.
        self.values = np.zeros(self.__num_arms, dtype=float)

        # Total number of actions taken so far.
        self.t = 0

        # Exploration strength for UCB.
        self.c = 2.0

    def update_state(self, action, reward):
        """
            TODO:
            Based on your choice of algorithm, use the the current action and
            reward to update the state of the agent.
            Optinal function, only use if needed.
        """
        ## IMPLEMENTATION
        self.t += 1
        self.counts[action] += 1

        # Incremental mean update:
        # Q(a) <- Q(a) + (1 / N(a)) * (R - Q(a))
        step_size = 1.0 / self.counts[action]
        self.values[action] += step_size * (reward - self.values[action])

    def get_action(self) -> int:
        """
            TODO:
            Based on your choice of algorithm, generate the next action based on
            the current state of your agent.
            Return the index of the arm picked by the policy.
        """
        ## IMPLEMENTATION
        # Pull each arm at least once so every estimate is grounded in data.
        unexplored = np.where(self.counts == 0)[0]
        if len(unexplored) > 0:
            return int(unexplored[0])

        # Upper Confidence Bound (UCB) action selection balances exploration
        # and exploitation for the stationary bandit setting in this assignment.
        bonus = self.c * np.sqrt(np.log(self.t) / self.counts)
        ucb_values = self.values + bonus

        return int(np.argmax(ucb_values))
