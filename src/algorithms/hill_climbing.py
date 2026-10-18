"""Hill-climbing: iterasi menghitung langkah yang diterima, mulai dari 0."""

import random
from dataclasses import asdict
from time import perf_counter

from core.evaluation import is_valid, objective_value
from core.operators import (
    generate_neighbors, random_initial_state, move_package, rotate_package,
    swap_packages, outside_position,
)
from core.state import FleetState, State


def _check_limit(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} harus berupa bilangan bulat non-negatif.")


def _start(initial_state: State) -> State:
    if not is_valid(initial_state):
        raise ValueError("Initial state harus valid.")
    return initial_state.copy()


def _history(initial: State, final: State, values: list, start: float, reason: str, objective: str) -> dict:
    return {
        "initial_state": asdict(initial),
        "final_state": asdict(final),
        "final_objective": values[-1],
        "objective": values,
        "objective_name": objective,
        "iteration": list(range(len(values))),
        "iterations": len(values) - 1,
        "duration_seconds": perf_counter() - start,
        "stop_reason": reason,
    }


def steepest_ascent(initial_state: State, objective: str = "value") -> tuple[State, dict]:
    start = perf_counter()
    current = _start(initial_state)
    values = [objective_value(current, objective)]
    while True:
        best, best_value = None, values[-1]
        for neighbor in generate_neighbors(current):
            value = objective_value(neighbor, objective)
            if value > best_value:
                best, best_value = neighbor, value
        if best is None:
            break
        current = best
        values.append(best_value)
    return current, _history(initial_state, current, values, start, "no_improvement", objective)


def random_restart(
    initial_state: State,
    max_restarts: int = 10,
    seed: int | None = None,
    max_attempts: int = 50,
    objective: str = "value",
) -> tuple[State, dict]:
    start = perf_counter()
    _check_limit("max_restarts", max_restarts)
    if type(max_attempts) is not int or max_attempts <= 0:
        raise ValueError("max_attempts harus berupa bilangan bulat positif.")
    current = _start(initial_state)
    rng = random.Random(seed)
    seed_base = rng.getrandbits(128)
    best, best_value, best_run = current.copy(), objective_value(current, objective), 0
    runs, values, iterations, run_indices = [], [], [], []
    accepted = 0
    for run_index in range(max_restarts + 1):
        restart_seed = None if run_index == 0 else seed_base + run_index
        if run_index:
            current = random_initial_state(
                [s.truck for s in initial_state.trucks] if isinstance(initial_state, FleetState) else initial_state.truck,
                initial_state.packages,
                seed=restart_seed, max_attempts=max_attempts,
            )
        final, history = steepest_ascent(current, objective)
        history.update(run_index=run_index, seed=restart_seed)
        runs.append(history)
        if run_index == 0 or history["final_objective"] > best_value:
            best, best_value, best_run = final.copy(), history["final_objective"], run_index
        values.extend(history["objective"])
        iterations.extend(accepted + i for i in history["iteration"])
        run_indices.extend([run_index] * len(history["objective"]))
        accepted += history["iterations"]
    history = _history(initial_state, best, values, start, "max_restarts", objective)
    history.update(
        final_objective=best_value, iterations=accepted, iteration=iterations,
        run_index=run_indices, runs=runs, restarts=max_restarts,
        iterations_per_run=[run["iterations"] for run in runs],
        best_run_index=best_run, seed=seed,
    )
    return best, history


def _random_successor(state: State, rng: random.Random) -> State | None:
    if isinstance(state, FleetState):
        neighbors = list(generate_neighbors(state))
        return rng.choice(neighbors) if neighbors else None
    index = rng.randrange(len(state.packages))
    operation = rng.choice(("move", "rotate", "swap"))
    if operation == "rotate":
        return rotate_package(state, index, rng.choice(("x", "y", "z")))
    if operation == "swap":
        if len(state.packages) < 2:
            return None
        other = rng.randrange(len(state.packages) - 1)
        if other >= index:
            other += 1
        return swap_packages(state, index, other)

    sizes = state.packages[index].oriented_dimensions()
    counts = [max(0, state.truck.dimensions[side] - size + 1)
              for side, size in zip(("w", "l", "h"), sizes)]
    inside_count = counts[0] * counts[1] * counts[2]
    target = rng.randrange(inside_count + 1)
    if target == inside_count:
        position = outside_position(state, index)
    else:
        x, remainder = divmod(target, counts[1] * counts[2])
        y, z = divmod(remainder, counts[2])
        position = (x, y, z)
    return move_package(state, index, position)


def stochastic(
    initial_state: State, seed: int | None = None, nmax: int = 100, objective: str = "value",
) -> tuple[State, dict]:
    start = perf_counter()
    _check_limit("nmax", nmax)
    current = _start(initial_state)
    rng = random.Random(seed)
    values = [objective_value(current, objective)]
    attempt_values = [values[0]]
    accepted_attempts = []
    invalid_proposals = 0
    reason = "nmax" if current.packages or nmax == 0 else "no_neighbors"
    for attempt in range(1, nmax + 1):
        if not current.packages:
            break
        selected = _random_successor(current, rng)
        if selected is None:
            invalid_proposals += 1
        else:
            value = objective_value(selected, objective)
            if value > values[-1]:
                current = selected
                values.append(value)
                accepted_attempts.append(attempt)
        attempt_values.append(values[-1])
    history = _history(initial_state, current, values, start, reason, objective)
    history.update(
        seed=seed, nmax=nmax, selection="random_operation_proposal",
        attempts=len(attempt_values) - 1, attempt_objective=attempt_values,
        accepted_attempts=accepted_attempts, invalid_proposals=invalid_proposals,
    )
    return current, history


def sideways(
    initial_state: State, max_sideways: int = 100, seed: int | None = None, objective: str = "value",
) -> tuple[State, dict]:
    start = perf_counter()
    _check_limit("max_sideways", max_sideways)
    current = _start(initial_state)
    rng = random.Random(seed)
    values = [objective_value(current, objective)]
    sideways_moves = 0
    while True:
        best, best_value, ties = None, values[-1], 0
        for neighbor in generate_neighbors(current):
            value = objective_value(neighbor, objective)
            if value > best_value:
                best, best_value, ties = neighbor, value, 1
            elif value == best_value:
                ties += 1
                if rng.randrange(ties) == 0:
                    best = neighbor
        if best is None:
            reason = "no_non_decreasing_neighbor"
            break
        equal = best_value == values[-1]
        if equal and sideways_moves == max_sideways:
            reason = "max_sideways"
            break
        current = best
        values.append(best_value)
        if equal:
            sideways_moves += 1
            if sideways_moves == max_sideways:
                reason = "max_sideways"
                break
    history = _history(initial_state, current, values, start, reason, objective)
    history.update(sideways_moves=sideways_moves, sideways_limit_mode="total", seed=seed)
    return current, history
