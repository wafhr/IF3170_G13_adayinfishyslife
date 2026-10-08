import argparse
from pathlib import Path

from algorithms.hill_climbing import random_restart, sideways, steepest_ascent, stochastic
from algorithms.simulated_annealing import simulated_annealing
from experiments.output import write_result
from experiments.runner import run_experiment


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["simulated_annealing", "random_restart", "steepest_ascent", "stochastic", "sideways"], default="simulated_annealing")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "problem.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=100.0)
    parser.add_argument("--cooling-rate", type=float, default=0.95)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--max-restarts", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--max-sideways", type=int, default=100)
    parser.add_argument("--nmax", type=int, default=100)
    args = parser.parse_args()
    if args.output is None:
        args.output = PROJECT_ROOT / "output" / args.algorithm / "run_1.json"
    algorithm = simulated_annealing
    parameters = {
        "initial_temperature": args.temperature,
        "cooling_rate": args.cooling_rate,
        "max_iterations": args.iterations,
    }
    if args.algorithm == "random_restart":
        algorithm = random_restart
        parameters = {"max_restarts": args.max_restarts, "max_attempts": args.max_attempts, "seed": args.seed}
    elif args.algorithm == "steepest_ascent":
        algorithm, parameters = steepest_ascent, {}
    elif args.algorithm == "stochastic":
        algorithm, parameters = stochastic, {"seed": args.seed, "nmax": args.nmax}
    elif args.algorithm == "sideways":
        algorithm, parameters = sideways, {"max_sideways": args.max_sideways, "seed": args.seed}

    result = run_experiment(
        algorithm_name=args.algorithm,
        algorithm=algorithm,
        config_path=args.config,
        run_name=args.output.stem,
        seed=args.seed,
        parameters=parameters,
    )
    write_result(args.output, result)
    print(f"Hasil disimpan di {args.output}")


if __name__ == "__main__":
    main()
