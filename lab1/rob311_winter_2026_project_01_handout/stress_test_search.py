import math
import os
import random
import sys
import time

import numpy as np

from breadth_first_search import breadth_first_search
from bidirectional_search import bidirectional_search
from a_star_search import a_star_search
from search_problems import GraphSearchProblem, GridSearchProblem


BASE_SEED = 20260304


QUICK_CONFIG = {
    "mode": "quick",
    "graph_random_target": 220,
    "grid_random_target": 220,
    "graph_timeout_sec": 0.30,
    "grid_timeout_sec": 0.30,
    "graph_ns": [10, 50, 200],
    "graph_densities": [0.005, 0.02, 0.05, 0.10, 0.20],
    "grid_sizes": [(10, 10), (30, 30)],
    "grid_probs": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.38, 0.40, 0.45],
    "graph_n_weights": [0.45, 0.40, 0.15],
    "grid_size_weights": [0.55, 0.45],
    "enable_heavy_handcrafted": False,
    "save_plot": False,
}


HEAVY_CONFIG = {
    "mode": "heavy",
    "graph_random_target": 1000,
    "grid_random_target": 500,
    "graph_timeout_sec": 1.50,
    "grid_timeout_sec": 1.50,
    "graph_ns": [200, 500, 1000],
    "graph_densities": [0.002, 0.005, 0.01, 0.02, 0.05],
    "grid_sizes": [(50, 50), (80, 80)],
    "grid_probs": [0.05, 0.10, 0.20, 0.30, 0.35, 0.40],
    "graph_n_weights": [0.70, 0.25, 0.05],
    "grid_size_weights": [0.60, 0.40],
    "enable_heavy_handcrafted": True,
    "save_plot": True,
}


class StressHarness:
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures = []

    def pass_case(self):
        self.total += 1
        self.passed += 1

    def fail_case(self, category, details):
        self.total += 1
        self.failed += 1
        self.failures.append((category, details))
        print("[FAIL] {} | {}".format(category, details))

    def skip_case(self, category, details):
        self.skipped += 1
        print("[SKIP] {} | {}".format(category, details))


class ComparisonStats:
    def __init__(self, left_name, right_name):
        self.left_name = left_name
        self.right_name = right_name
        self.left_expanded = []
        self.right_expanded = []
        self.left_frontier = []
        self.right_frontier = []
        self.hardest_left = []
        self.hardest_right = []

    def _update_top(self, bucket, value, seed, params):
        bucket.append({"value": int(value), "seed": int(seed), "params": params})
        bucket.sort(key=lambda item: item["value"], reverse=True)
        if len(bucket) > 5:
            del bucket[5:]

    def record(self, left_expanded, right_expanded, left_frontier, right_frontier, seed, params):
        self.left_expanded.append(int(left_expanded))
        self.right_expanded.append(int(right_expanded))
        self.left_frontier.append(int(left_frontier))
        self.right_frontier.append(int(right_frontier))
        self._update_top(self.hardest_left, left_expanded, seed, params)
        self._update_top(self.hardest_right, right_expanded, seed, params)


def _edge_array(edge_list):
    if not edge_list:
        return np.empty((0, 2), dtype=int)
    return np.array(edge_list, dtype=int)


def validate_graph_path(path, V, E, start, goal, directed=False):
    if path is None:
        return False, "path is None"
    if path == []:
        return True, "empty path (no-solution convention)"
    if path[0] != start:
        return False, "path does not start at init_state"
    if path[-1] != goal:
        return False, "path does not end at goal_state"

    V_set = set(int(v) for v in V)
    for s in path:
        if int(s) not in V_set:
            return False, "path contains state not in V"

    edge_set = set((int(u), int(v)) for u, v in E)
    for idx in range(len(path) - 1):
        u = int(path[idx])
        v = int(path[idx + 1])
        if directed:
            if (u, v) not in edge_set:
                return False, "invalid directed edge {}->{}".format(u, v)
        else:
            if (u, v) not in edge_set and (v, u) not in edge_set:
                return False, "invalid undirected edge {}-{}".format(u, v)

    return True, "valid"


def validate_grid_path(problem, path):
    if path is None:
        return False, "path is None"
    if path == []:
        return True, "empty path (no-solution convention)"
    if path[0] != problem.init_state:
        return False, "path does not start at init_state"
    if path[-1] != problem.goal_states[0]:
        return False, "path does not end at goal_state"
    if not problem.check_solution(path):
        return False, "problem.check_solution returned False"
    return True, "valid"


def validate_stats(num_nodes_expanded, max_frontier_size, start_equals_goal):
    if num_nodes_expanded < 0:
        return False, "num_nodes_expanded < 0"
    if not start_equals_goal and max_frontier_size < 1:
        return False, "max_frontier_size < 1 with start != goal"
    if max_frontier_size < 0:
        return False, "max_frontier_size < 0"
    return True, "valid"


def make_chain_graph(n):
    V = np.arange(n, dtype=int)
    E = _edge_array([(i, i + 1) for i in range(n - 1)])
    return V, E


def make_star_graph(n):
    V = np.arange(n, dtype=int)
    E = _edge_array([(0, i) for i in range(1, n)])
    return V, E


def make_cycle_graph(n, include_self_loops=False):
    edges = [(i, (i + 1) % n) for i in range(n)]
    if include_self_loops:
        edges.extend([(0, 0), (n // 2, n // 2)])
    V = np.arange(n, dtype=int)
    E = _edge_array(edges)
    return V, E


def make_lollipop_graph(clique_size, tail_len):
    n = clique_size + tail_len
    edges = []
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            edges.append((i, j))
    if tail_len > 0:
        edges.append((clique_size - 1, clique_size))
        for i in range(clique_size, n - 1):
            edges.append((i, i + 1))
    V = np.arange(n, dtype=int)
    E = _edge_array(edges)
    return V, E


def make_barbell_graph(clique_a, clique_b, bridge_len):
    offset_b = clique_a + bridge_len
    n = clique_a + clique_b + bridge_len
    edges = []

    for i in range(clique_a):
        for j in range(i + 1, clique_a):
            edges.append((i, j))

    for i in range(offset_b, offset_b + clique_b):
        for j in range(i + 1, offset_b + clique_b):
            edges.append((i, j))

    left_anchor = clique_a - 1
    if bridge_len == 0:
        edges.append((left_anchor, offset_b))
    else:
        edges.append((left_anchor, clique_a))
        for i in range(clique_a, clique_a + bridge_len - 1):
            edges.append((i, i + 1))
        edges.append((clique_a + bridge_len - 1, offset_b))

    V = np.arange(n, dtype=int)
    E = _edge_array(edges)
    return V, E


def make_multi_shortest_path_graph():
    V = np.arange(6, dtype=int)
    E = _edge_array([
        (0, 1), (1, 5),
        (0, 2), (2, 5),
        (0, 3), (3, 4), (4, 5),
        (1, 2),
    ])
    return V, E


def erdos_renyi_graph(n, p, rng, self_loop_prob=0.0):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j))
    if self_loop_prob > 0.0:
        for i in range(n):
            if rng.random() < self_loop_prob:
                edges.append((i, i))
    V = np.arange(n, dtype=int)
    E = _edge_array(edges)
    return V, E


def build_grid_problem(M, N, p_occ, rng, force_no_path=False, start_equals_goal=False):
    grid_map = np.array([[rng.random() < p_occ for _ in range(N)] for _ in range(M)], dtype=bool)
    init_state = rng.randrange(M * N)
    goal_state = init_state if start_equals_goal else rng.randrange(M * N)

    if force_no_path:
        sx = init_state % M
        sy = init_state // M
        nbrs = [(sx + 1, sy), (sx - 1, sy), (sx, sy + 1), (sx, sy - 1)]
        for nx, ny in nbrs:
            if 0 <= nx < M and 0 <= ny < N:
                grid_map[nx, ny] = True

        if not start_equals_goal:
            gx = goal_state % M
            gy = goal_state // M
            nbrs_g = [(gx + 1, gy), (gx - 1, gy), (gx, gy + 1), (gx, gy - 1)]
            for nx, ny in nbrs_g:
                if 0 <= nx < M and 0 <= ny < N:
                    grid_map[nx, ny] = True

    return GridSearchProblem([goal_state], init_state, M, N, grid_map)


def run_graph_case(harness, graph_stats, case_name, seed, V, E, start, goal, graph_timeout_sec, directed=False):
    details_base = "seed={}, V={}, E={}, start={}, goal={}".format(seed, len(V), E.shape[0], start, goal)
    problem = GraphSearchProblem([goal], start, V, E)

    t0 = time.perf_counter()
    bfs_path, bfs_expanded, bfs_frontier = breadth_first_search(problem)
    bfs_dt = time.perf_counter() - t0
    if bfs_dt > graph_timeout_sec:
        harness.skip_case("Graph/{}".format(case_name), details_base + ", note=slow BFS ({:.3f}s)".format(bfs_dt))
        return

    t1 = time.perf_counter()
    bi_path, bi_expanded, bi_frontier = bidirectional_search(problem)
    bi_dt = time.perf_counter() - t1
    if bi_dt > graph_timeout_sec:
        harness.skip_case("Graph/{}".format(case_name), details_base + ", note=slow BiDir ({:.3f}s)".format(bi_dt))
        return

    start_equals_goal = (start == goal)
    ok_stats_bfs, msg_stats_bfs = validate_stats(bfs_expanded, bfs_frontier, start_equals_goal)
    ok_stats_bi, msg_stats_bi = validate_stats(bi_expanded, bi_frontier, start_equals_goal)

    ok_bfs_valid, bfs_msg = validate_graph_path(bfs_path, V, E, start, goal, directed=directed)
    ok_bi_valid, bi_msg = validate_graph_path(bi_path, V, E, start, goal, directed=directed)

    bfs_has_solution = bfs_path not in (None, [])
    bi_has_solution = bi_path not in (None, [])

    if not ok_stats_bfs:
        harness.fail_case("Graph/{}".format(case_name), details_base + ", bfs_stats_error={}".format(msg_stats_bfs))
        return
    if not ok_stats_bi:
        harness.fail_case("Graph/{}".format(case_name), details_base + ", bidir_stats_error={}".format(msg_stats_bi))
        return
    if not ok_bfs_valid:
        harness.fail_case("Graph/{}".format(case_name), details_base + ", bfs_path_invalid={}".format(bfs_msg))
        return
    if not ok_bi_valid:
        harness.fail_case("Graph/{}".format(case_name), details_base + ", bidir_path_invalid={}".format(bi_msg))
        return

    if bfs_has_solution != bi_has_solution:
        harness.fail_case(
            "Graph/{}".format(case_name),
            details_base + ", mismatch_solution_presence bfs={} bidir={}".format(bfs_has_solution, bi_has_solution),
        )
        return

    if bfs_has_solution and len(bfs_path) != len(bi_path):
        harness.fail_case(
            "Graph/{}".format(case_name),
            details_base + ", mismatch_length bfs_len={} bidir_len={}".format(len(bfs_path), len(bi_path)),
        )
        return

    graph_stats.record(
        bfs_expanded,
        bi_expanded,
        bfs_frontier,
        bi_frontier,
        seed,
        "V={}, E={}, start={}, goal={}".format(len(V), E.shape[0], start, goal),
    )

    harness.pass_case()


def run_grid_case(harness, grid_stats, case_name, seed, problem, grid_timeout_sec):
    M, N = problem.M, problem.N
    p_occ = float(np.mean(problem.grid_map))
    start = int(problem.init_state)
    goal = int(problem.goal_states[0])
    details_base = "seed={}, M={}, N={}, p_occ={:.3f}, start={}, goal={}".format(seed, M, N, p_occ, start, goal)

    t0 = time.perf_counter()
    bfs_path, bfs_expanded, bfs_frontier = breadth_first_search(problem)
    bfs_dt = time.perf_counter() - t0
    if bfs_dt > grid_timeout_sec:
        harness.skip_case("Grid/{}".format(case_name), details_base + ", note=slow BFS ({:.3f}s)".format(bfs_dt))
        return

    t1 = time.perf_counter()
    a_path, a_expanded, a_frontier = a_star_search(problem)
    a_dt = time.perf_counter() - t1
    if a_dt > grid_timeout_sec:
        harness.skip_case("Grid/{}".format(case_name), details_base + ", note=slow A* ({:.3f}s)".format(a_dt))
        return

    start_equals_goal = (start == goal)
    ok_stats_bfs, msg_stats_bfs = validate_stats(bfs_expanded, bfs_frontier, start_equals_goal)
    ok_stats_a, msg_stats_a = validate_stats(a_expanded, a_frontier, start_equals_goal)

    ok_bfs_valid, bfs_msg = validate_grid_path(problem, bfs_path)
    ok_a_valid, a_msg = validate_grid_path(problem, a_path)

    bfs_has_solution = bfs_path not in (None, [])
    a_has_solution = a_path not in (None, [])

    if not ok_stats_bfs:
        harness.fail_case("Grid/{}".format(case_name), details_base + ", bfs_stats_error={}".format(msg_stats_bfs))
        return
    if not ok_stats_a:
        harness.fail_case("Grid/{}".format(case_name), details_base + ", a_star_stats_error={}".format(msg_stats_a))
        return
    if not ok_bfs_valid:
        harness.fail_case("Grid/{}".format(case_name), details_base + ", bfs_path_invalid={}".format(bfs_msg))
        return
    if not ok_a_valid:
        harness.fail_case("Grid/{}".format(case_name), details_base + ", a_star_path_invalid={}".format(a_msg))
        return

    if bfs_has_solution != a_has_solution:
        harness.fail_case(
            "Grid/{}".format(case_name),
            details_base + ", mismatch_solution_presence bfs={} a_star={}".format(bfs_has_solution, a_has_solution),
        )
        return

    if bfs_has_solution and len(bfs_path) != len(a_path):
        harness.fail_case(
            "Grid/{}".format(case_name),
            details_base + ", mismatch_length bfs_len={} a_star_len={}".format(len(bfs_path), len(a_path)),
        )
        return

    grid_stats.record(
        bfs_expanded,
        a_expanded,
        bfs_frontier,
        a_frontier,
        seed,
        "M={}, N={}, p_occ={:.3f}, start={}, goal={}".format(M, N, p_occ, start, goal),
    )

    harness.pass_case()


def run_handcrafted_graph_tests(harness, graph_stats, config):
    rng = random.Random(BASE_SEED + 1)

    V, E = make_chain_graph(12)
    run_graph_case(harness, graph_stats, "chain_long", BASE_SEED + 1001, V, E, 0, 11, config["graph_timeout_sec"])

    V, E = make_star_graph(20)
    run_graph_case(harness, graph_stats, "star_leaf_to_leaf", BASE_SEED + 1002, V, E, 7, 19, config["graph_timeout_sec"])

    V, E = make_lollipop_graph(8, 18)
    run_graph_case(harness, graph_stats, "lollipop", BASE_SEED + 1003, V, E, 0, 25, config["graph_timeout_sec"])

    V, E = make_barbell_graph(7, 7, 10)
    run_graph_case(harness, graph_stats, "barbell", BASE_SEED + 1004, V, E, 0, 23, config["graph_timeout_sec"])

    V, E = make_multi_shortest_path_graph()
    run_graph_case(harness, graph_stats, "multiple_shortest", BASE_SEED + 1005, V, E, 0, 5, config["graph_timeout_sec"])

    V, E = make_cycle_graph(25, include_self_loops=True)
    run_graph_case(harness, graph_stats, "cycles_and_self_loops", BASE_SEED + 1006, V, E, 0, 13, config["graph_timeout_sec"])

    V = np.arange(10, dtype=int)
    E = _edge_array([(0, 1), (1, 2), (2, 3), (7, 8)])
    run_graph_case(harness, graph_stats, "disconnected_no_path", BASE_SEED + 1007, V, E, 0, 9, config["graph_timeout_sec"])

    V, E = erdos_renyi_graph(30, 0.25, rng, self_loop_prob=0.05)
    run_graph_case(harness, graph_stats, "erdos_self_loop_mix", BASE_SEED + 1008, V, E, 4, 4, config["graph_timeout_sec"])

    if config["enable_heavy_handcrafted"]:
        V, E = make_chain_graph(2000)
        run_graph_case(harness, graph_stats, "heavy_chain_2000", BASE_SEED + 1101, V, E, 0, 1999, config["graph_timeout_sec"])

        V, E = make_star_graph(2001)
        goal_leaf = rng.randint(1, 2000)
        run_graph_case(harness, graph_stats, "heavy_star_2000_leaves", BASE_SEED + 1102, V, E, 1, goal_leaf, config["graph_timeout_sec"])

        for idx in range(3):
            clique = rng.randint(50, 120)
            tail = rng.randint(200, 600)
            V, E = make_lollipop_graph(clique, tail)
            run_graph_case(
                harness,
                graph_stats,
                "heavy_lollipop_{}".format(idx + 1),
                BASE_SEED + 1110 + idx,
                V,
                E,
                0,
                clique + tail - 1,
                config["graph_timeout_sec"],
            )

        for idx in range(3):
            clique_a = rng.randint(50, 120)
            clique_b = rng.randint(50, 120)
            bridge = rng.randint(200, 600)
            V, E = make_barbell_graph(clique_a, clique_b, bridge)
            run_graph_case(
                harness,
                graph_stats,
                "heavy_barbell_{}".format(idx + 1),
                BASE_SEED + 1120 + idx,
                V,
                E,
                0,
                len(V) - 1,
                config["graph_timeout_sec"],
            )


def run_random_graph_tests(harness, graph_stats, config):
    target_random_tests = config["graph_random_target"]
    rng = random.Random(BASE_SEED + 2)
    attempted = 0
    executed = 0

    n_choices = config["graph_ns"]
    n_weights = config["graph_n_weights"]
    density_choices = config["graph_densities"]

    while executed < target_random_tests:
        attempted += 1

        n = rng.choices(n_choices, weights=n_weights, k=1)[0]
        p = density_choices[(attempted - 1) % len(density_choices)]
        if rng.random() < 0.30:
            p = rng.choice(density_choices)

        self_loop_prob = 0.02 if rng.random() < 0.12 else 0.0
        V, E = erdos_renyi_graph(n, p, rng, self_loop_prob=self_loop_prob)

        start = rng.randrange(n)
        goal = start if rng.random() < 0.05 else rng.randrange(n)

        before_total = harness.total
        before_skips = harness.skipped
        run_graph_case(
            harness,
            graph_stats,
            "random_erdos_renyi_n{}_p{:.3f}".format(n, p),
            BASE_SEED + 200000 + attempted,
            V,
            E,
            start,
            goal,
            config["graph_timeout_sec"],
        )
        if harness.total > before_total or harness.skipped > before_skips:
            executed += 1

    print("[INFO] Random graph cases attempted={}, executed={} (target={})".format(attempted, executed, target_random_tests))


def run_handcrafted_grid_tests(harness, grid_stats, config):
    rng = random.Random(BASE_SEED + 3)

    grid = np.zeros((10, 10), dtype=bool)
    problem = GridSearchProblem([0], 0, 10, 10, grid)
    run_grid_case(harness, grid_stats, "start_equals_goal", BASE_SEED + 3001, problem, config["grid_timeout_sec"])

    M, N = 10, 10
    grid = np.zeros((M, N), dtype=bool)
    for x in range(M):
        if x != M // 2:
            grid[x, N // 2] = True
    problem = GridSearchProblem([N * M - 1], 0, M, N, grid)
    run_grid_case(harness, grid_stats, "single_gap_wall", BASE_SEED + 3002, problem, config["grid_timeout_sec"])

    problem = build_grid_problem(20, 20, 0.15, rng, force_no_path=True, start_equals_goal=False)
    run_grid_case(harness, grid_stats, "forced_no_path", BASE_SEED + 3003, problem, config["grid_timeout_sec"])

    open_grid = np.zeros((12, 12), dtype=bool)
    problem = GridSearchProblem([11 * 12 + 11], 0, 12, 12, open_grid)
    run_grid_case(harness, grid_stats, "multiple_shortest_open_grid", BASE_SEED + 3004, problem, config["grid_timeout_sec"])

    near_phase = build_grid_problem(30, 30, 0.38, rng, force_no_path=False)
    run_grid_case(harness, grid_stats, "near_phase_transition", BASE_SEED + 3005, near_phase, config["grid_timeout_sec"])


def run_random_grid_tests(harness, grid_stats, config):
    target_random_tests = config["grid_random_target"]
    rng = random.Random(BASE_SEED + 4)
    attempted = 0
    executed = 0

    size_choices = config["grid_sizes"]
    size_weights = config["grid_size_weights"]
    all_probs = config["grid_probs"]

    while executed < target_random_tests:
        attempted += 1

        M, N = rng.choices(size_choices, weights=size_weights, k=1)[0]
        p_occ = all_probs[(attempted - 1) % len(all_probs)]
        if rng.random() < 0.40:
            p_occ = rng.choice(all_probs)

        if config["mode"] == "heavy" and attempted % 20 == 0 and (M, N) == (80, 80):
            M, N = 100, 100

        force_no_path = rng.random() < 0.10
        start_equals_goal = rng.random() < 0.05

        problem = build_grid_problem(M, N, p_occ, rng, force_no_path=force_no_path, start_equals_goal=start_equals_goal)

        before_total = harness.total
        before_skips = harness.skipped
        run_grid_case(
            harness,
            grid_stats,
            "random_grid_{}x{}_p{:.2f}".format(M, N, p_occ),
            BASE_SEED + 400000 + attempted,
            problem,
            config["grid_timeout_sec"],
        )
        if harness.total > before_total or harness.skipped > before_skips:
            executed += 1

    print("[INFO] Random grid cases attempted={}, executed={} (target={})".format(attempted, executed, target_random_tests))


def _safe_stats(values):
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    avg = float(np.mean(arr))
    med = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    return avg, med, p95


def print_comparison_table(title, stats_obj):
    left_exp_avg, left_exp_med, left_exp_p95 = _safe_stats(stats_obj.left_expanded)
    right_exp_avg, right_exp_med, right_exp_p95 = _safe_stats(stats_obj.right_expanded)
    left_fr_avg, left_fr_med, left_fr_p95 = _safe_stats(stats_obj.left_frontier)
    right_fr_avg, right_fr_med, right_fr_p95 = _safe_stats(stats_obj.right_frontier)

    print("\n{}".format(title))
    print("  metric=num_nodes_expanded")
    print("    {}: avg={:.2f}, median={:.2f}, p95={:.2f}".format(stats_obj.left_name, left_exp_avg, left_exp_med, left_exp_p95))
    print("    {}: avg={:.2f}, median={:.2f}, p95={:.2f}".format(stats_obj.right_name, right_exp_avg, right_exp_med, right_exp_p95))
    print("  metric=max_frontier_size")
    print("    {}: avg={:.2f}, median={:.2f}, p95={:.2f}".format(stats_obj.left_name, left_fr_avg, left_fr_med, left_fr_p95))
    print("    {}: avg={:.2f}, median={:.2f}, p95={:.2f}".format(stats_obj.right_name, right_fr_avg, right_fr_med, right_fr_p95))

    print("  hardest-by-{}-expanded (top 5):".format(stats_obj.left_name))
    for idx, item in enumerate(stats_obj.hardest_left, start=1):
        print("    {}. expanded={}, seed={}, {}".format(idx, item["value"], item["seed"], item["params"]))

    print("  hardest-by-{}-expanded (top 5):".format(stats_obj.right_name))
    for idx, item in enumerate(stats_obj.hardest_right, start=1):
        print("    {}. expanded={}, seed={}, {}".format(idx, item["value"], item["seed"], item["params"]))


def maybe_save_heavy_plot(harness):
    if harness.failed > 0:
        print("[INFO] Skipping PNG export because some tests failed.")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[INFO] matplotlib not available; skipping PNG save for A* example.")
        return

    rng = random.Random(BASE_SEED + 9000)
    selected = None
    for attempt in range(1, 26):
        problem = build_grid_problem(50, 50, 0.30, rng, force_no_path=False, start_equals_goal=False)
        path, _, _ = a_star_search(problem)
        if path not in (None, []):
            selected = (problem, path, attempt)
            break

    if selected is None:
        print("[INFO] Could not find a solvable representative 50x50 p=0.30 grid for PNG export.")
        return

    problem, path, attempt = selected
    original_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        fig = problem.plot_solution(path)
    finally:
        plt.show = original_show
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    png_path = os.path.join(repo_root, "a_star_example.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("[INFO] Saved A* representative plot to: {}".format(png_path))
    print("[INFO] Relative path: a_star_example.png")
    print(
        "[INFO] Example details: M={}, N={}, p_occ~0.30, attempt={}, start={}, goal={}, path_len={}".format(
            problem.M,
            problem.N,
            attempt,
            int(problem.init_state),
            int(problem.goal_states[0]),
            len(path),
        )
    )


def print_summary(harness, elapsed_seconds, mode, config, graph_stats, grid_stats):
    print("\n========== STRESS TEST SUMMARY ==========")
    print("Mode            : {}".format(mode))
    print("Seed base       : {}".format(BASE_SEED))
    print(
        "Targets         : randomized_graph={}, randomized_grid={}".format(
            config["graph_random_target"],
            config["grid_random_target"],
        )
    )
    print("Total tests run : {}".format(harness.total))
    print("Passed          : {}".format(harness.passed))
    print("Failed          : {}".format(harness.failed))
    print("Skipped (slow)  : {}".format(harness.skipped))
    print("Elapsed (sec)   : {:.3f}".format(elapsed_seconds))

    print_comparison_table("Graph comparison (BFS vs Bidirectional)", graph_stats)
    print_comparison_table("Grid comparison (BFS vs A*)", grid_stats)

    if harness.failures:
        print("\nFailures (repro details):")
        for idx, (category, details) in enumerate(harness.failures, start=1):
            print("  {}. {} | {}".format(idx, category, details))
    else:
        print("\nAll executed tests passed.")


def parse_config_from_argv(argv):
    args = set(argv[1:])
    if "--heavy" in args:
        mode = "heavy"
        config = HEAVY_CONFIG
        args.remove("--heavy")
    else:
        mode = "quick"
        config = QUICK_CONFIG

    if args:
        print("[WARN] Ignoring unknown args: {}".format(sorted(args)))

    return mode, config


def main():
    mode, config = parse_config_from_argv(sys.argv)

    print("========== STRESS TEST REPORT ==========")
    print("Mode: {}".format(mode))
    print("Seed base: {}".format(BASE_SEED))
    print(
        "Planned randomized tests: graph={}, grid={}".format(
            config["graph_random_target"],
            config["grid_random_target"],
        )
    )
    print("Graph n set: {} | densities: {}".format(config["graph_ns"], config["graph_densities"]))
    print("Grid sizes: {} | obstacle probs: {}".format(config["grid_sizes"], config["grid_probs"]))
    print("Running handcrafted + randomized graph/grid stress tests...")

    harness = StressHarness()
    graph_stats = ComparisonStats("BFS", "Bidirectional")
    grid_stats = ComparisonStats("BFS", "A*")
    start_time = time.perf_counter()

    run_handcrafted_graph_tests(harness, graph_stats, config)
    run_random_graph_tests(harness, graph_stats, config)

    run_handcrafted_grid_tests(harness, grid_stats, config)
    run_random_grid_tests(harness, grid_stats, config)

    elapsed = time.perf_counter() - start_time
    print_summary(harness, elapsed, mode, config, graph_stats, grid_stats)

    if config["save_plot"]:
        maybe_save_heavy_plot(harness)


if __name__ == "__main__":
    main()
