"""Genetic algorithm for the truck loading problem."""

from __future__ import annotations

import random
import time
from dataclasses import replace
from typing import Any

from core.evaluation import is_valid, objective_value
from core.operators import generate_neighbors, outside_position, random_initial_state
from core.state import ORIENTATIONS, FleetState, Package, State, Truck


def _fitness(state: State, objective: str = "value") -> float:
    return objective_value(state, objective)


def _randomized_initial_individual(
    truck: Truck | list[Truck],
    packages: list[Package],
    rng: random.Random,
    repair_attempts: int,
) -> State | FleetState:
    """Create a valid GA individual with randomized assignment and load density."""
    base = random_initial_state(truck, packages, seed=rng.randrange(2**32))
    states = base.trucks if isinstance(base, FleetState) else [base]
    loaded = [
        (truck_index, package.id)
        for truck_index, state in enumerate(states)
        for package in state.packages
        if state.is_inside(package)
    ]
    if not loaded:
        return base

    # Randomize how many of the initially placeable packages remain loaded.
    # The ordinary initializer is greedy, so without this step GA starts from
    # states that are already densely packed and often near the best objective.
    min_loaded = min(len(loaded), max(1, round(len(packages) * 0.30)))
    max_loaded = min(len(loaded), max(min_loaded, round(len(packages) * 0.70)))
    keep_count = rng.randint(min_loaded, max_loaded)
    kept = set(rng.sample(loaded, keep_count))

    randomized_states = []
    for truck_index, state in enumerate(states):
        genes = [
            (package.position, package.orientation.copy(),
             (truck_index, package.id) in kept)
            for package in state.packages
        ]
        randomized_states.append(
            _place_genes(state.truck, state.packages, genes, rng, repair_attempts)
        )

    return FleetState(randomized_states) if isinstance(base, FleetState) else randomized_states[0]


def _place_genes(
    truck: Truck,
    template: list[Package],
    genes: list[tuple[tuple[int, int, int], dict[str, str], bool]],
    rng: random.Random,
    repair_attempts: int,
) -> State:
    """Build a valid state from inherited genes, repairing placements as needed."""
    state = State(truck, [replace(p) for p in template])
    for i, package in enumerate(state.packages):
        package.position = outside_position(state, i)

    # Place valuable packages first so repair preserves the strongest genes.
    order = sorted(range(len(state.packages)), key=lambda i: state.packages[i].value, reverse=True)
    for i in order:
        position, orientation, wanted_inside = genes[i]
        package = state.packages[i]
        if not wanted_inside:
            package.orientation = orientation.copy()
            continue

        package.position = position
        package.orientation = orientation.copy()
        if is_valid(state):
            continue
        package.position = outside_position(state, i)

        # A parent's exact placement may no longer fit after crossover. Try a
        # bounded set of fresh placements before leaving this package outside.
        for _ in range(repair_attempts):
            package.orientation = rng.choice(ORIENTATIONS).copy()
            width, length, height = package.oriented_dimensions()
            max_x = truck.dimensions["w"] - width
            max_y = truck.dimensions["l"] - length
            max_z = truck.dimensions["h"] - height
            if min(max_x, max_y, max_z) < 0:
                continue
            levels = {0}
            for other in state.packages:
                if other.id != package.id and state.is_inside(other) and not other.isFragile:
                    top = state.bounds(other)[1][2]
                    if top <= max_z:
                        levels.add(top)
            package.position = (
                rng.randint(0, max_x),
                rng.randint(0, max_y),
                rng.choice(sorted(levels)),
            )
            if is_valid(state):
                break
            package.position = outside_position(state, i)
    return state


def _genes_from(state: State) -> list[tuple[tuple[int, int, int], dict[str, str], bool]]:
    return [
        (p.position, p.orientation.copy(), state.is_inside(p))
        for p in state.packages
    ]


def _crossover(
    first: State,
    second: State,
    rng: random.Random,
    repair_attempts: int,
) -> State:
    if isinstance(first, FleetState):
        trucks = [State(s.truck, []) for s in first.trucks]
        for package in first.packages:
            parent = first if rng.random() < 0.5 else second
            source, package_index = parent.locate(package.id)
            truck_index = next(i for i, candidate in enumerate(parent.trucks) if candidate is source)
            inherited = replace(source.packages[package_index])
            trucks[truck_index].packages.append(inherited)
        child = FleetState(trucks)
        return child if is_valid(child) else (first.copy() if rng.random() < 0.5 else second.copy())
    genes_a, genes_b = _genes_from(first), _genes_from(second)
    genes = [rng.choice((a, b)) for a, b in zip(genes_a, genes_b)]
    return _place_genes(first.truck, first.packages, genes, rng, repair_attempts)


def _mutate(state: State, rng: random.Random, repair_attempts: int) -> State:
    if isinstance(state, FleetState):
        neighbors = list(generate_neighbors(state))
        return rng.choice(neighbors) if neighbors else state.copy()
    genes = _genes_from(state)
    index = rng.randrange(len(genes))
    position, orientation, inside = genes[index]
    operation = rng.choice(("move", "rotate", "remove", "insert"))
    if operation == "remove":
        inside = False
    elif operation == "insert":
        inside = True
    elif operation == "rotate":
        orientation = rng.choice(ORIENTATIONS).copy()
    else:
        package = state.packages[index]
        width, length, height = package.oriented_dimensions()
        max_x = state.truck.dimensions["w"] - width
        max_y = state.truck.dimensions["l"] - length
        max_z = state.truck.dimensions["h"] - height
        if min(max_x, max_y, max_z) >= 0:
            position = (rng.randint(0, max_x), rng.randint(0, max_y), rng.randint(0, max_z))
            inside = True
    genes[index] = (position, orientation, inside)
    return _place_genes(state.truck, state.packages, genes, rng, repair_attempts)


def genetic_algorithm(
    truck: Truck | list[Truck],
    packages: list[Package],
    population_size: int = 40,
    iterations: int = 100,
    mutation_rate: float = 0.4,
    elite_size: int = 1,
    tournament_size: int = 2,
    repair_attempts: int = 40,
    seed: int | None = None,
    objective: str = "value",
) -> dict[str, Any]:
    """Run GA and return best states plus per-generation max/average fitness."""
    if type(population_size) is not int or population_size < 2:
        raise ValueError("population_size harus bilangan bulat minimal 2.")
    if type(iterations) is not int or iterations < 0:
        raise ValueError("iterations harus bilangan bulat nonnegatif.")
    if not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate harus berada pada rentang 0 sampai 1.")
    if type(elite_size) is not int or not 1 <= elite_size < population_size:
        raise ValueError("elite_size harus minimal 1 dan lebih kecil dari population_size.")
    if type(tournament_size) is not int or tournament_size < 1:
        raise ValueError("tournament_size harus bilangan bulat positif.")
    if type(repair_attempts) is not int or repair_attempts < 1:
        raise ValueError("repair_attempts harus bilangan bulat positif.")

    rng = random.Random(seed)
    started = time.perf_counter()
    population = [
        _randomized_initial_individual(truck, packages, rng, repair_attempts)
        for _ in range(population_size)
    ]
    if not all(is_valid(individual) for individual in population):
        raise RuntimeError("Inisialisasi menghasilkan state yang tidak valid.")
    initial_population = [individual.copy() for individual in population]
    initial_scores = [_fitness(individual, objective) for individual in initial_population]
    initial_best_score = max(initial_scores, default=0.0)
    non_best_initial = [
        individual for individual, score in zip(initial_population, initial_scores)
        if score < initial_best_score
    ]
    # Keep the displayed starting individual representative of a random
    # population member rather than accidentally showing the best individual.
    display_rng = random.Random(None if seed is None else seed ^ 0x5A17)
    displayed_initial_state = display_rng.choice(non_best_initial or initial_population).copy()
    max_history: list[float] = []
    average_history: list[float] = []

    def record() -> None:
        scores = [_fitness(individual, objective) for individual in population]
        max_history.append(max(scores, default=0.0))
        average_history.append(sum(scores) / len(scores) if scores else 0.0)

    record()
    for _ in range(iterations):
        ranked = sorted(population, key=lambda individual: _fitness(individual, objective), reverse=True)
        next_population = [individual.copy() for individual in ranked[:elite_size]]
        while len(next_population) < population_size:
            pool = rng.sample(population, min(tournament_size, len(population)))
            parent_a = max(pool, key=lambda individual: _fitness(individual, objective))
            pool = rng.sample(population, min(tournament_size, len(population)))
            parent_b = max(pool, key=lambda individual: _fitness(individual, objective))
            child = _crossover(parent_a, parent_b, rng, repair_attempts)
            if rng.random() < mutation_rate and child.packages:
                child = _mutate(child, rng, repair_attempts)
            next_population.append(child)
        population = next_population
        record()

    best = max(population, key=lambda individual: _fitness(individual, objective))
    return {
        "initial_population": initial_population,
        # A single representative initial state should be an actual random
        # individual, not the population's best member. Population-level best
        # and average are recorded separately in their history series.
        "initial_state": displayed_initial_state,
        "best_initial_state": max(
            initial_population, key=lambda individual: _fitness(individual, objective)
        ).copy(),
        "final_state": best.copy(),
        "final_value": _fitness(best, objective),
        "objective_name": objective,
        "max_history": max_history,
        "average_history": average_history,
        "population_size": population_size,
        "iterations": iterations,
        "duration_seconds": time.perf_counter() - started,
        "seed": seed,
    }


def run_genetic_experiments(
    truck: Truck,
    packages: list[Package],
    iteration_values: list[int],
    population_values: list[int],
    repetitions: int = 3,
    fixed_population_size: int = 40,
    fixed_iterations: int = 100,
    seed: int | None = None,
    **ga_options: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Run the two required one-parameter-at-a-time GA experiments."""
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions harus bilangan bulat positif.")
    rng = random.Random(seed)

    def run_configs(configs: list[tuple[int, int]]) -> list[dict[str, Any]]:
        records = []
        for pop_size, generations in configs:
            for run_index in range(repetitions):
                result = genetic_algorithm(
                    truck,
                    packages,
                    population_size=pop_size,
                    iterations=generations,
                    seed=rng.randrange(2**32),
                    **ga_options,
                )
                records.append({
                    "population_size": pop_size,
                    "iterations": generations,
                    "run": run_index + 1,
                    **result,
                })
        return records

    return {
        "vary_iterations": run_configs([
            (fixed_population_size, count) for count in iteration_values
        ]),
        "vary_population": run_configs([
            (count, fixed_iterations) for count in population_values
        ]),
    }
