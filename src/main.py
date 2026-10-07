import argparse
from pathlib import Path

from algorithms.simulated_annealing import simulated_annealing
from experiments.output import write_result
from experiments.runner import run_experiment


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["simulated_annealing"], default="simulated_annealing")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "problem.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output" / "simulated_annealing" / "run_1.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=100.0)
    parser.add_argument("--cooling-rate", type=float, default=0.95)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    result = run_experiment(
        algorithm_name=args.algorithm,
        algorithm=simulated_annealing,
        config_path=args.config,
        run_name=args.output.stem,
        seed=args.seed,
        parameters={
            "initial_temperature": args.temperature,
            "cooling_rate": args.cooling_rate,
            "max_iterations": args.iterations,
        },
    )
    write_result(args.output, result)
    print(f"Hasil disimpan di {args.output}")


if __name__ == "__main__":
    main()
