from core.state import State


def is_valid(state: State) -> bool:
    loaded = []
    total_weight = 0

    for package in state.packages:
        if state.is_inside(package):
            loaded.append(package)
            total_weight += package.weight
            if total_weight > state.truck.max_capacity:
                return False
        elif not state.is_outside(package):
            return False

    for i, package in enumerate(loaded):
        start, end = state.bounds(package)
        for other in loaded[i + 1:]:
            other_start, other_end = state.bounds(other)
            # Paket overlap
            if all(start[axis] < other_end[axis] and other_start[axis] < end[axis] for axis in range(3)):
                return False

        supported = start[2] == 0
        for other in loaded:
            if other.id == package.id:
                continue
            other_start, other_end = state.bounds(other)
            # Permukaan atas harus tepat di bawah alas dan beririsan pada x-y
            touches_base = (
                other_end[2] == start[2]
                and start[0] < other_end[0]
                and other_start[0] < end[0]
                and start[1] < other_end[1]
                and other_start[1] < end[1]
            )
            if touches_base:
                supported = True
                # Paket pecah-belah
                if other.isFragile:
                    return False

        if not supported:
            return False

    return True

OBJECTIVE_LABELS = {
    "value": "Total value paket",
    "eta": "Prioritas ETA",
}


def objective_value(state: State, objective: str = "value") -> float:
    """Nilai objective untuk paket yang seluruh bagiannya berada di dalam truk."""
    if objective == "value":
        return sum(package.value for package in state.packages if state.is_inside(package))
    if objective == "eta":
        return sum(1 / (package.eta + 1) for package in state.packages if state.is_inside(package))
    raise ValueError(f"Objective tidak dikenal: {objective}")
