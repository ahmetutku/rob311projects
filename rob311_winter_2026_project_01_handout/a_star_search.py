import queue
import numpy as np
from search_problems import Node, GridSearchProblem, get_random_grid_problem


def a_star_search(problem):
    """
    Uses the A* algorithm to solve an instance of GridSearchProblem. Use the methods of GridSearchProblem along with
    structures and functions from the allowed imports (see above) to implement A*.

    :param problem: an instance of GridSearchProblem to solve
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    # Do not add imports; autograder strips them.
    # Return path + stats tuple; path is list of states.
    # Return None/[] if no solution.
    # nodes_generated counts all successor generations, even if pruned.
    start_state = problem.init_state
    if problem.goal_test(start_state):
        return [start_state], 0, 1

    frontier = queue.PriorityQueue()
    tie_breaker = 0
    g_score = {start_state: 0}
    came_from = {start_state: None}
    closed = set()

    start_h = problem.heuristic(start_state) if hasattr(problem, 'heuristic') else 0
    frontier.put((start_h, tie_breaker, start_state))

    nodes_generated = 0
    max_frontier_size = 1

    while not frontier.empty():
        _, _, current_state = frontier.get()

        if current_state in closed:
            continue

        if problem.goal_test(current_state):
            path = []
            state = current_state
            while state is not None:
                path.append(state)
                state = came_from[state]
            path.reverse()
            return path, nodes_generated, max_frontier_size

        closed.add(current_state)

        for action in problem.get_actions(current_state):
            successor_state = problem.transition(current_state, action)
            nodes_generated += 1

            step_cost = problem.action_cost(current_state, action, successor_state)
            tentative_g = g_score[current_state] + step_cost
            current_best = g_score.get(successor_state, float('inf'))

            if tentative_g < current_best:
                g_score[successor_state] = tentative_g
                came_from[successor_state] = current_state
                if successor_state in closed:
                    closed.remove(successor_state)
                tie_breaker += 1
                heuristic_val = problem.heuristic(successor_state) if hasattr(problem, 'heuristic') else 0
                frontier.put((tentative_g + heuristic_val, tie_breaker, successor_state))
                if frontier.qsize() > max_frontier_size:
                    max_frontier_size = frontier.qsize()

    return None, nodes_generated, max_frontier_size


def search_phase_transition():
    """
    Simply fill in the prob. of occupancy values for the 'phase transition' and peak nodes expanded within 0.05. You do
    NOT need to submit your code that determines the values here: that should be computed on your own machine. Simply
    fill in the values!

    :return: tuple containing (transition_start_probability, transition_end_probability, peak_probability)
    """
    # TODO: Replace with your own experimentally measured values for N=500.
    transition_start_probability = 0.30
    transition_end_probability = 0.45
    peak_nodes_expanded_probability = 0.38
    return transition_start_probability, transition_end_probability, peak_nodes_expanded_probability


if __name__ == '__main__':
    # Test your code here!
    # Create a random instance of GridSearchProblem
    p_occ = 0.25
    M = 10
    N = 10
    problem = get_random_grid_problem(p_occ, M, N)
    # Solve it
    path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
    # Check the result
    correct = problem.check_solution(path)
    print("Solution is correct: {:}".format(correct))
    # Plot the result
    problem.plot_solution(path)

    # Experiment and compare with BFS