
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

        # Search aggressively at the beginning, then exploit the strongest
        # discovered arm for the short 400-step horizon.
        self.search_horizon = min(30, self.__num_arms)
        self.exploit_bonus = 0.15

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
            # Optimistic greedy search: continue a perfect success streak, but
            # move on quickly after a failure to sample more candidate arms.
            best_action = 0
            best_value = self.values[0]
            best_count = self.counts[0]

            for arm in range(1, self.__num_arms):
                value = self.values[arm]
                count = self.counts[arm]
                if value > best_value or (value == best_value and count > best_count):
                    best_action = arm
                    best_value = value
                    best_count = count

            return best_action

        # After the search phase, ignore untouched arms and choose the best
        # tried arm with a very small confidence bonus to correct early lucky
        # estimates without returning to broad exploration.
        best_action = 0
        best_value = -1.0
        best_count = -1

        for arm in range(self.__num_arms):
            count = self.counts[arm]
            if count == 0:
                continue

            value = self.values[arm] + self.exploit_bonus / math.sqrt(count)
            if value > best_value or (value == best_value and count > best_count):
                best_action = arm
                best_value = value
                best_count = count

        return best_action
