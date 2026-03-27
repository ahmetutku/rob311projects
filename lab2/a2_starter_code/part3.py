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


class MAB_agent:
    """
        TODO:
        Implement the get_action and update_state function of an agent such that it
        is able to maximize the reward on the Multi-Armed Bandit (MAB) environment.
    """
    def __init__(self, num_arms=5):
        self.__num_arms = num_arms

        # Deterministic arm order keeps the file independent of random.
        self.arm_order = list(range(self.__num_arms))

        # Search only a small subset at the start because the horizon is short.
        self.search_horizon = min(self.__num_arms, 20)

        # After the initial survey, probe a new arm on a sparse deterministic
        # schedule: 8, 24, 56, 120, ... episodes after the search phase.
        self.next_untried = self.search_horizon
        self.probe_gap = 8
        self.next_probe_step = self.search_horizon + self.probe_gap

        # Standard bandit state.
        self.counts = [0] * self.__num_arms
        self.values = [1.0] * self.__num_arms
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
        if self.t < self.search_horizon:
            return self.arm_order[self.t]

        if self.next_untried < self.__num_arms and self.t == self.next_probe_step:
            arm = self.arm_order[self.next_untried]
            self.next_untried += 1
            self.next_probe_step += self.probe_gap
            self.probe_gap *= 2
            return arm

        best_arm = 0
        best_score = float("-inf")
        best_count = -1
        exploit_bonus = 0.10

        for arm in self.arm_order[:self.next_untried]:
            count = self.counts[arm]
            value = self.values[arm]

            # Every arm in the tried set has already been sampled at least once.
            score = value + exploit_bonus / ((count + 1) ** 0.5)

            if score > best_score + 1e-12:
                best_score = score
                best_count = count
                best_arm = arm
            elif abs(score - best_score) <= 1e-12:
                if count > best_count:
                    best_count = count
                    best_arm = arm
                elif count == best_count and arm < best_arm:
                    best_arm = arm

        return best_arm
