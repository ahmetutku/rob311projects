from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

def breadth_first_search(problem):
    """
    Implement a simple breadth-first search algorithm that takes instances of SimpleSearchProblem (or its derived
    classes) and provides a valid and optimal path from the initial state to the goal state. Useful for testing your
    bidirectional and A* search algorithms.

    :param problem: instance of SimpleSearchProblem
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by the search
             max_frontier_size: maximum frontier size during search
    """
    # Do not add imports; autograder strips them.
    # Return path + stats tuple; path is list of states.
    # Return None/[] if no solution.
    start_state = problem.init_state
    if problem.goal_test(start_state):
        return [start_state], 0, 1

    root = Node(None, start_state, None, 0)
    frontier = deque([root])
    frontier_states = {start_state}
    explored = set()

    num_nodes_expanded = 0
    max_frontier_size = 1

    while frontier:
        node = frontier.popleft()
        frontier_states.discard(node.state)

        if problem.goal_test(node.state):
            return problem.trace_path(node), num_nodes_expanded, max_frontier_size

        explored.add(node.state)
        num_nodes_expanded += 1

        for action in problem.get_actions(node.state):
            child = problem.get_child_node(node, action)
            if child.state in explored or child.state in frontier_states:
                continue
            frontier.append(child)
            frontier_states.add(child.state)

        if len(frontier) > max_frontier_size:
            max_frontier_size = len(frontier)

    return [], num_nodes_expanded, max_frontier_size


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
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('../datasets/stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)