import json
import math
from pathlib import Path

from core.state import State


def state_data(state: State) -> dict:
    return {
        "packages": [
            {
                "id": package.id,
                "eta": package.eta,
                "position": list(package.position),
                "orientation": package.orientation,
            }
            for package in state.packages
        ]
    }


def result_data(
    algorithm: str,
    run_name: str,
    initial_state: State,
    final_state: State,
    final_objective: float,
    duration_seconds: float,
    parameters: dict,
    history: dict,
) -> dict:
    objectives = history.get("objective", history.get("max_objective", []))
    iterations = history.get("iterations", len(objectives) if isinstance(objectives, list) else 0)
    return {
        "algorithm": algorithm,
        "run_name": run_name,
        "initial_state": state_data(initial_state),
        "final_state": state_data(final_state),
        "final_objective": final_objective,
        "duration_seconds": duration_seconds,
        "iterations": iterations,
        "parameters": parameters,
        "history": history,
    }


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(result), file, indent=2)
