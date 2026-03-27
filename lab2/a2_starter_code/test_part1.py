# test_part1.py
#
# --
# Artificial Intelligence
# ROB 311
# Programming Assignment 2

# Teaching Assistant:
# Zi Cong (Daniel) Guo
# zc.guo@mail.utoronto.ca

from part1_1 import get_transition_model
from part1_2 import value_iteration
from mdp_cleaning_task import cleaning_env
from mdp_grid_task import grid_env
from mdp_agent import mdp_agent

def print_policy(env: cleaning_env, policy: any):
    print("\n------------- Value Iteration (Policy)  -------------- ")
    print_string = ""

    for i in range(len(env.states)):
        print_string = print_string + "\t" + env.state_names[i]

    print_string = print_string + "\n"
    for i in range(len(env.states)):
        print_string = print_string + "\t"
        if env.states[i] in env.terminal:
            print_string = print_string + "T"
        else:
            print_string = print_string + env.action_names[policy[i]]
    print(print_string)

# Check transition model
def test_1():
    env = cleaning_env()
    env.init_stochatic_model(get_transition_model)
    env.print_env()
    env.print_transition_model()

# Check cleaning task
def test_2():
    env = cleaning_env()
    env.init_stochatic_model(get_transition_model)
    env.print_env()

    gamma = 0.8
    agent = mdp_agent(gamma)

    epsilon = 0.1
    policy = value_iteration(env, agent, epsilon)
    print_policy(env, policy)

if __name__ == '__main__':
    test_1()
    test_2()