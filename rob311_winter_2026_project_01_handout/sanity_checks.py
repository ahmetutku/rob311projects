import heapq
import numpy as np

from breadth_first_search import breadth_first_search
from bidirectional_search import bidirectional_search
from a_star_search import a_star_search
from search_problems import (
    GraphSearchProblem,
    GridSearchProblem,
    get_random_grid_problem,
)


def print_result(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    if detail:
        print(f"[{status}] {name}: {detail}")
    else:
        print(f"[{status}] {name}")


def bfs_reference_graph(problem):
    start = problem.init_state
    goal = problem.goal_states[0]
    if start == goal:
        return [start]

    queue = [start]
    parent = {start: None}
    head = 0

    while head < len(queue):
        state = queue[head]
        head += 1

        if state == goal:
            break

        for action in problem.get_actions(state):
            child = problem.transition(state, action)
            if child in parent:
                continue
            parent[child] = state
            queue.append(child)

    if goal not in parent:
        return None

    path = []
    s = goal
    while s is not None:
        path.append(s)
        s = parent[s]
    path.reverse()
    return path


def ucs_reference_grid(problem):
    start = problem.init_state
    goal = problem.goal_states[0]
    if start == goal:
        return [start]

    pq = [(0, start)]
    g_score = {start: 0}
    parent = {start: None}

    while pq:
        g, state = heapq.heappop(pq)
        if g != g_score.get(state, float("inf")):
            continue

        if state == goal:
            path = []
            s = state
            while s is not None:
                path.append(s)
                s = parent[s]
            path.reverse()
            return path

        for action in problem.get_actions(state):
            nxt = problem.transition(state, action)
            new_g = g + problem.action_cost(state, action, nxt)
            if new_g < g_score.get(nxt, float("inf")):
                g_score[nxt] = new_g
                parent[nxt] = state
                heapq.heappush(pq, (new_g, nxt))

    return None


def run_bfs_checks():
    print("\n=== BFS Checks ===")

    V = np.arange(0, 7)
    E = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 4],
        [4, 5],
        [5, 3],
        [3, 6],
    ])
    problem = GraphSearchProblem([6], 0, V, E)
    path, expanded, max_frontier = breadth_first_search(problem)
    ref = bfs_reference_graph(problem)

    valid = (path is not None) and problem.check_graph_solution(path)
    optimal = (path is not None and ref is not None and len(path) == len(ref))
    endpoints = (path is not None and path[0] == problem.init_state and path[-1] == problem.goal_states[0])

    print_result("BFS returns valid path", valid)
    print_result("BFS path endpoints", endpoints)
    print_result("BFS optimal length", optimal, f"len(path)={len(path) if path else None}, len(ref)={len(ref) if ref else None}")
    print_result("BFS max_frontier>=1", max_frontier >= 1, f"max_frontier={max_frontier}")
    print_result("BFS expanded nonnegative", expanded >= 0, f"expanded={expanded}")

    V2 = np.arange(0, 5)
    E2 = np.array([
        [0, 1],
        [1, 2],
        [3, 4],
    ])
    no_path_problem = GraphSearchProblem([4], 0, V2, E2)
    no_path, _, _ = breadth_first_search(no_path_problem)
    print_result("BFS no-path returns None/[]", (no_path is None) or (no_path == []))


def run_bidirectional_checks():
    print("\n=== Bidirectional Checks ===")

    V = np.arange(0, 9)
    E = np.array([
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 4],
        [2, 6], [4, 8],
    ])
    problem = GraphSearchProblem([8], 0, V, E)

    bfs_path, _, _ = breadth_first_search(problem)
    bidir_path, expanded, max_frontier = bidirectional_search(problem)

    valid = (bidir_path is not None) and problem.check_graph_solution(bidir_path)
    same_len = (bfs_path is not None and bidir_path is not None and len(bfs_path) == len(bidir_path))

    print_result("Bidirectional returns valid path", valid)
    print_result("Bidirectional matches BFS optimal length", same_len,
                 f"len(bidir)={len(bidir_path) if bidir_path else None}, len(bfs)={len(bfs_path) if bfs_path else None}")
    print_result("Bidirectional max_frontier>=1", max_frontier >= 1, f"max_frontier={max_frontier}")
    print_result("Bidirectional expanded nonnegative", expanded >= 0, f"expanded={expanded}")

    V2 = np.arange(0, 6)
    E2 = np.array([
        [0, 1], [1, 2],
        [3, 4], [4, 5],
    ])
    no_path_problem = GraphSearchProblem([5], 0, V2, E2)
    bfs_none, _, _ = breadth_first_search(no_path_problem)
    bidir_none, _, _ = bidirectional_search(no_path_problem)

    consistent_none = (((bfs_none is None) or (bfs_none == [])) and ((bidir_none is None) or (bidir_none == [])))
    print_result("Bidirectional no-path consistent with BFS", consistent_none)


def run_a_star_checks():
    print("\n=== A* Checks ===")

    seeds = [7, 11, 23, 42, 101]
    p_occ_values = [0.10, 0.20, 0.30]
    all_valid = True
    all_optimal_small = True
    all_max_frontier_ok = True

    for p_occ in p_occ_values:
        for seed in seeds:
            np.random.seed(seed)
            problem = get_random_grid_problem(p_occ, 12, 12)
            path, generated, max_frontier = a_star_search(problem)

            if path is None or path == []:
                ref = ucs_reference_grid(problem)
                case_ok = (ref is None)
                all_valid = all_valid and case_ok
                detail = f"seed={seed}, p={p_occ}, A*=None, ref={'None' if ref is None else 'Path'}"
                print_result("A* no-path behavior", case_ok, detail)
            else:
                valid = problem.check_solution(path)
                endpoints = (path[0] == problem.init_state and path[-1] == problem.goal_states[0])
                all_valid = all_valid and valid and endpoints
                detail = f"seed={seed}, p={p_occ}, len={len(path)}, generated={generated}"
                print_result("A* path validity", valid and endpoints, detail)

                ref = ucs_reference_grid(problem)
                optimal_small = (ref is not None and len(path) == len(ref))
                all_optimal_small = all_optimal_small and optimal_small
                print_result("A* optimal vs UCS (small grid)", optimal_small,
                             f"seed={seed}, p={p_occ}, len(A*)={len(path)}, len(UCS)={len(ref) if ref else None}")

            all_max_frontier_ok = all_max_frontier_ok and (max_frontier >= 1)

    print_result("A* aggregate validity", all_valid)
    print_result("A* aggregate optimality on tested cases", all_optimal_small)
    print_result("A* max_frontier>=1 in all tested cases", all_max_frontier_ok)

    grid_map = np.ones((3, 3), dtype=bool)
    no_solution_problem = GridSearchProblem([8], 0, 3, 3, grid_map)
    no_path, _, _ = a_star_search(no_solution_problem)
    print_result("A* explicit no-solution returns None/[]", (no_path is None) or (no_path == []))


def run_stats_sanity():
    print("\n=== Stats Sanity Checks ===")

    V_small = np.arange(0, 12)
    E_small = np.array([[i, i + 1] for i in range(11)])
    p_small = GraphSearchProblem([11], 0, V_small, E_small)
    _, expanded_small, frontier_small = breadth_first_search(p_small)

    V_large = np.arange(0, 40)
    E_large = np.array([[i, i + 1] for i in range(39)])
    p_large = GraphSearchProblem([39], 0, V_large, E_large)
    _, expanded_large, frontier_large = breadth_first_search(p_large)

    print_result("BFS expansions increase on larger line graph", expanded_large >= expanded_small,
                 f"small={expanded_small}, large={expanded_large}")
    print_result("BFS frontier sanity", frontier_small >= 1 and frontier_large >= 1,
                 f"small={frontier_small}, large={frontier_large}")

    seeds = [3, 5, 9, 17, 21]
    generated_10 = []
    generated_20 = []
    for seed in seeds:
        np.random.seed(seed)
        p10 = get_random_grid_problem(0.20, 10, 10)
        _, g10, _ = a_star_search(p10)
        generated_10.append(g10)

        np.random.seed(seed)
        p20 = get_random_grid_problem(0.20, 20, 20)
        _, g20, _ = a_star_search(p20)
        generated_20.append(g20)

    avg_10 = float(np.mean(generated_10))
    avg_20 = float(np.mean(generated_20))
    print_result("A* generated increases with larger grid on average", avg_20 >= avg_10,
                 f"avg10={avg_10:.2f}, avg20={avg_20:.2f}")


def main():
    run_bfs_checks()
    run_bidirectional_checks()
    run_a_star_checks()
    run_stats_sanity()


if __name__ == "__main__":
    main()
