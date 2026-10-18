import math
import random
from typing import Any

from core.evaluation import objective_value
from core.operators import generate_neighbors
from core.state import State


def simulated_annealing(
    initial_state: State,
    initial_temperature: float,
    cooling_rate: float,
    max_iterations: int,
    objective: str = "value",
) -> tuple[State, dict[str, Any]]:
    if initial_temperature < 0:
        raise ValueError("Initial temperature tidak boleh negatif.")
    if not 0 < cooling_rate <= 1:
        raise ValueError("Cooling rate harus berada pada rentang (0, 1].")
    if type(max_iterations) is not int or max_iterations < 0:
        raise ValueError("Max iterations harus berupa bilangan bulat non-negatif.")

    current = initial_state.copy()
    best = current.copy()
    history = {
        "objective": [],
        "objective_name": objective,
        "exp_delta_e_over_t": [],
        "temperature": [],
        "stuck": [],
        "stuck_frequency": 0,
    }

    for iteration in range(1, max_iterations + 1):
        temperature = initial_temperature * cooling_rate ** (iteration - 1)
        if temperature <= 0:
            break

        neighbors = list(generate_neighbors(current))
        if not neighbors:
            break

        current_value = objective_value(current, objective)
        is_stuck = not any(objective_value(neighbor, objective) > current_value for neighbor in neighbors)
        next_state = random.choice(neighbors)
        delta_e = objective_value(next_state, objective) - current_value
        try:
            exp_delta_e_over_t = math.exp(delta_e / temperature)
        except OverflowError:
            exp_delta_e_over_t = math.inf

        if min(1.0, exp_delta_e_over_t) > random.random():
            current = next_state

        if objective_value(current, objective) > objective_value(best, objective):
            best = current.copy()

        history["objective"].append(objective_value(current, objective))
        history["exp_delta_e_over_t"].append(exp_delta_e_over_t)
        history["temperature"].append(temperature)
        history["stuck"].append(is_stuck)
        history["stuck_frequency"] += is_stuck

    return best, history
