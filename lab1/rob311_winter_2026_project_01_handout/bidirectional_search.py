from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

def bidirectional_search(problem):
    """
        Implement a bidirectional search algorithm that takes instances of SimpleSearchProblem (or its derived
        classes) and provides a valid and optimal path from the initial state to the goal state.

        :param problem: instance of SimpleSearchProblem
        :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
                 num_nodes_expanded: number of nodes expanded by the search
                 max_frontier_size: maximum frontier size during search
        """
    # Do not add imports; autograder strips them.
    # Return path + stats tuple; path is list of states.
    # Return None/[] if no solution.
    start_state = problem.init_state
    goal_state = problem.goal_states[0]

    if problem.goal_test(start_state):
        return [start_state], 0, 1

    frontier_f = deque([start_state])
    frontier_b = deque([goal_state])

    discovered_f = {start_state}
    discovered_b = {goal_state}

    parent_f = {start_state: None}
    parent_b = {goal_state: None}

    depth_f = {start_state: 0}
    depth_b = {goal_state: 0}

    num_nodes_expanded = 0
    max_frontier_size = len(frontier_f) + len(frontier_b)

    meeting_state = None
    best_total = float('inf')

    next_depth_f = 0
    next_depth_b = 0

    def update_best(candidate_state):
        nonlocal meeting_state, best_total
        total_depth = depth_f[candidate_state] + depth_b[candidate_state]
        if total_depth < best_total:
            best_total = total_depth
            meeting_state = candidate_state

    def expand_layer(frontier, discovered_this, discovered_other,
                     parent_this, depth_this, depth_other, current_depth):
        nonlocal num_nodes_expanded, max_frontier_size
        layer_size = len(frontier)
        for _ in range(layer_size):
            state = frontier.popleft()
            num_nodes_expanded += 1

            if state in discovered_other:
                update_best(state)

            for action in problem.get_actions(state):
                child = problem.transition(state, action)
                if child in discovered_this:
                    continue

                discovered_this.add(child)
                parent_this[child] = state
                depth_this[child] = current_depth + 1
                frontier.append(child)

                if child in discovered_other:
                    update_best(child)

                if len(frontier_f) + len(frontier_b) > max_frontier_size:
                    max_frontier_size = len(frontier_f) + len(frontier_b)

    while frontier_f and frontier_b:
        if meeting_state is not None and (next_depth_f + next_depth_b) > best_total:
            break

        expand_layer(frontier_f, discovered_f, discovered_b,
                     parent_f, depth_f, depth_b, next_depth_f)
        next_depth_f += 1
        if len(frontier_f) + len(frontier_b) > max_frontier_size:
            max_frontier_size = len(frontier_f) + len(frontier_b)

        if not frontier_f or not frontier_b:
            break

        if meeting_state is not None and (next_depth_f + next_depth_b) > best_total:
            break

        expand_layer(frontier_b, discovered_b, discovered_f,
                     parent_b, depth_b, depth_f, next_depth_b)
        next_depth_b += 1
        if len(frontier_f) + len(frontier_b) > max_frontier_size:
            max_frontier_size = len(frontier_f) + len(frontier_b)

    if meeting_state is None:
        return [], num_nodes_expanded, max_frontier_size

    path_forward = []
    state = meeting_state
    while state is not None:
        path_forward.append(state)
        state = parent_f[state]
    path_forward.reverse()

    path_backward = []
    state = parent_b[meeting_state]
    while state is not None:
        path_backward.append(state)
        state = parent_b[state]

    return path_forward + path_backward, num_nodes_expanded, max_frontier_size


if __name__ == '__main__':
    # Simple example
    goal_states = [0]
    init_state = 9
    V = np.arange(0, 10)
    E = np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 8],
                  [8, 9],
                  [0, 6],
                  [1, 7],
                  [2, 5],
                  [9, 4]])
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('../datasets/stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Be sure to compare with breadth_first_search!