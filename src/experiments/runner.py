import json
import random
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from core.evaluation import objective_value
from core.operators import random_initial_state
from core.state import Package, State, Truck
from experiments.output import result_data


def load_problem(path: Path) -> tuple[Truck, list[Package]]:
    with path.open(encoding="utf-8") as file:
        problem = json.load(file)

    truck_data = problem["truck"]
    truck = Truck(truck_data["dimensions"], truck_data["max_capacity"])
    packages = [
        Package(
            id=package["id"],
            dimensions=package["dimensions"],
            value=package["value"],
            weight=package["weight"],
            isFragile=package["is_fragile"],
        )
        for package in problem["packages"]
    ]
    return truck, packages


def run_experiment(
    algorithm_name: str,
    algorithm: Callable[..., tuple[State, dict]],
    config_path: Path,
    run_name: str,
    seed: int,
    parameters: dict,
) -> dict:
    truck, packages = load_problem(config_path)
    initial_state = random_initial_state(truck, packages, seed=seed)

    random.seed(seed)
    start = perf_counter()
    final_state, history = algorithm(initial_state, **parameters)
    duration = perf_counter() - start

    return result_data(
        algorithm=algorithm_name,
        run_name=run_name,
        initial_state=initial_state,
        final_state=final_state,
        final_objective=objective_value(final_state),
        duration_seconds=duration,
        parameters={"seed": seed, **parameters},
        history=history,
    )
