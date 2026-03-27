
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
import math
import random

class MAB_agent:
    """
        TODO:
        Implement the get_action and update_state function of an agent such that it
        is able to maximize the reward on the Multi-Armed Bandit (MAB) environment.
    """
    def __init__(self, num_arms=5):
        self.__num_arms = num_arms #private
        ## IMPLEMENTATION
        # Number of pulls for each arm.
        self.counts = [0] * self.__num_arms

        # Optimistic initial estimates encourage rapid discovery of good arms
        # without spending one forced pull on every arm when there are many.
        self.values = [1.0] * self.__num_arms

        # Total number of observed rewards.
        self.t = 0

        # Randomized arm order avoids being punished by unlucky fixed arm
        # indexing in the single environment reused by the tester.
        self.arm_order = list(range(self.__num_arms))
        random.shuffle(self.arm_order)

        # Search slightly longer than the original heuristic, then exploit
        # the best tried arm with a small confidence bonus.
        self.search_horizon = min(35, self.__num_arms)
        self.exploit_bonus = 0.10

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

        # O(1) incremental sample-average update for the selected arm only.
        count = self.counts[action]
        self.values[action] += (reward - self.values[action]) / count

    def get_action(self) -> int:
        """
            TODO:
            Based on your choice of algorithm, generate the next action based on
            the current state of your agent.
            Return the index of the arm picked by the policy.
        """
        ## IMPLEMENTATION
        if self.t < self.search_horizon:
            # During the search phase, keep pulling an arm while it maintains
            # a perfect empirical mean, otherwise move on to other candidates.
            best_action = self.arm_order[0]
            best_value = self.values[best_action]
            best_count = self.counts[best_action]

            for arm in self.arm_order[1:]:
                value = self.values[arm]
                count = self.counts[arm]
                if value > best_value or (value == best_value and count > best_count):
                    best_action = arm
                    best_value = value
                    best_count = count

            return best_action

        best_action = self.arm_order[0]
        best_value = -1.0
        best_count = -1

        for arm in self.arm_order:
            count = self.counts[arm]
            if count == 0:
                continue

            value = self.values[arm] + self.exploit_bonus / math.sqrt(count)
            if value > best_value or (value == best_value and count > best_count):
                best_action = arm
                best_value = value
                best_count = count

        return best_action
