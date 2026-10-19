import json
import math
import os
import tempfile
from pathlib import Path

from core.state import FleetState, State


def state_data(state: State | FleetState) -> dict:
    if isinstance(state, FleetState):
        return {
            "truck_count": len(state.trucks),
            "trucks": [
                {"truck_index": index + 1, "packages": [
                    {"id": p.id, "eta": p.eta, "position": list(p.position),
                     "orientation": p.orientation, "inside": truck_state.is_inside(p)}
                    for p in truck_state.packages
                ]}
                for index, truck_state in enumerate(state.trucks)
            ],
            "packages": [
                {"id": p.id, "eta": p.eta, "position": list(p.position),
                 "orientation": p.orientation, "truck_index": index + 1,
                 "inside": truck_state.is_inside(p)}
                for index, truck_state in enumerate(state.trucks) for p in truck_state.packages
            ],
        }
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
    initial_state: State | FleetState,
    final_state: State | FleetState,
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
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(json_safe(result), file, indent=2)
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
