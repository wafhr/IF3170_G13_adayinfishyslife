import json
import random
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from algorithms.genetic import genetic_algorithm
from algorithms.hill_climbing import random_restart, sideways, steepest_ascent, stochastic
from algorithms.simulated_annealing import simulated_annealing
from core.evaluation import objective_value
from core.operators import random_initial_state
from core.state import Package, State, Truck
from experiments.output import result_data


ALGORITHMS = {
    "simulated_annealing": simulated_annealing,
    "random_restart": random_restart,
    "steepest_ascent": steepest_ascent,
    "stochastic": stochastic,
    "sideways": sideways,
}
ALGORITHM_NAMES = (*ALGORITHMS, "genetic_algorithm")


def load_problem(path: Path) -> tuple[Truck, list[Package]]:
    try:
        with path.open(encoding="utf-8") as file:
            problem = json.load(file)
    except OSError as error:
        raise OSError(f"Tidak dapat membaca file konfigurasi: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON konfigurasi tidak valid pada baris {error.lineno}, kolom {error.colno}.") from error

    if not isinstance(problem, dict):
        raise ValueError("Root konfigurasi harus berupa object JSON.")

    truck_data = problem.get("truck")
    package_data = problem.get("packages")
    if not isinstance(truck_data, dict) or not isinstance(package_data, list):
        raise ValueError("Konfigurasi harus berisi object `truck` dan list `packages`.")
    try:
        truck = Truck(truck_data["dimensions"], truck_data["max_capacity"])
    except (KeyError, TypeError) as error:
        raise ValueError("Konfigurasi truck harus memiliki `dimensions` dan `max_capacity` yang valid.") from error

    packages = []
    for index, package in enumerate(package_data, start=1):
        if not isinstance(package, dict):
            raise ValueError(f"Data paket ke-{index} harus berupa object JSON.")
        try:
            packages.append(Package(
                id=package["id"],
                dimensions=package["dimensions"],
                value=package["value"],
                weight=package["weight"],
                isFragile=package.get("is_fragile", False),
                eta=package.get("eta", 0),
            ))
        except (KeyError, TypeError) as error:
            raise ValueError(f"Data paket ke-{index} tidak lengkap atau formatnya salah.") from error
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


def run_genetic_experiment(
    config_path: Path,
    run_name: str,
    seed: int,
    population_size: int,
    iterations: int,
) -> dict:
    truck, packages = load_problem(config_path)
    elite_size = min(2, population_size - 1)
    result = genetic_algorithm(
        truck,
        packages,
        population_size=population_size,
        iterations=iterations,
        elite_size=elite_size,
        seed=seed,
    )
    history = {
        "max_objective": result["max_history"],
        "average_objective": result["average_history"],
        "iterations": iterations,
    }
    output = result_data(
        algorithm="genetic_algorithm",
        run_name=run_name,
        initial_state=result["best_initial_state"],
        final_state=result["final_state"],
        final_objective=result["final_value"],
        duration_seconds=result["duration_seconds"],
        parameters={"seed": seed, "population_size": population_size, "iterations": iterations, "elite_size": elite_size},
        history=history,
    )
    output["population_size"] = population_size
    return output


def run_selected_experiment(
    algorithm_name: str,
    config_path: Path,
    run_name: str,
    seed: int,
    parameters: dict,
) -> dict:
    if algorithm_name == "genetic_algorithm":
        return run_genetic_experiment(
            config_path=config_path,
            run_name=run_name,
            seed=seed,
            population_size=parameters["population_size"],
            iterations=parameters["iterations"],
        )

    algorithm = ALGORITHMS.get(algorithm_name)
    if algorithm is None:
        raise ValueError(f"Algoritma tidak dikenal: {algorithm_name}")

    algorithm_parameters = dict(parameters)
    if algorithm_name in {"random_restart", "stochastic", "sideways"}:
        algorithm_parameters["seed"] = seed
    return run_experiment(
        algorithm_name=algorithm_name,
        algorithm=algorithm,
        config_path=config_path,
        run_name=run_name,
        seed=seed,
        parameters=algorithm_parameters,
    )
