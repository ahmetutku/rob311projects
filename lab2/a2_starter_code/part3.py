
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
        self.__num_arms = num_arms

        # Shuffle the arm order once so the search phase stays unbiased.
        self.arm_order = list(range(self.__num_arms))
        random.shuffle(self.arm_order)

        # Search a modest randomized subset first.
        self.search_horizon = min(self.__num_arms, 20)

        # Track the next untouched arm so post-search correction can expand the
        # tried set slowly instead of revisiting the full action space at once.
        self.next_untried = self.search_horizon

        # Number of times each arm has been selected.
        self.counts = [0] * self.__num_arms

        # Optimistic initial values encourage early arm discovery.
        self.values = [1.0] * self.__num_arms

        # Total number of observed rewards.
        self.t = 0

    def update_state(self, action, reward):
        """
        Update the selected arm using the incremental sample average.
        """
        self.t += 1
        self.counts[action] += 1

        count = self.counts[action]
        estimate = self.values[action]
        self.values[action] = estimate + (reward - estimate) / count

    def get_action(self) -> int:
        """
            TODO:
            Based on your choice of algorithm, generate the next action based on
            the current state of your agent.
            Return the index of the arm picked by the policy.
        """
        ## IMPLEMENTATION
        if self.t < self.search_horizon:
            return self.arm_order[self.t]

        fresh_epsilon = max(0.0015, 0.015 / math.sqrt(self.t - self.search_horizon + 1.0))
        if self.next_untried < self.__num_arms and random.random() < fresh_epsilon:
            arm = self.arm_order[self.next_untried]
            self.next_untried += 1
            return arm

        tried_arms = self.arm_order[:self.next_untried]
        best_score = float("-inf")
        best_count = -1
        best_arms = []
        exploit_bonus = 0.10

        for arm in tried_arms:
            count = self.counts[arm]
            value = self.values[arm]

            # Every arm in tried_arms has been sampled at least once.
            score = value + exploit_bonus / math.sqrt(count)

            if score > best_score + 1e-12:
                best_score = score
                best_count = count
                best_arms = [arm]
            elif abs(score - best_score) <= 1e-12:
                if count > best_count:
                    best_score = score
                    best_count = count
                    best_arms = [arm]
                elif count == best_count:
                    best_arms.append(arm)

        return random.choice(best_arms)
