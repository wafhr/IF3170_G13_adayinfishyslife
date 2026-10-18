import random
from dataclasses import replace
from collections.abc import Iterator

from core.evaluation import is_valid
from core.state import ORIENTATIONS, FleetState, Package, State, Truck

# Helper yg mengembalikan koordinat di luar truk
def outside_position(state: State, package_index: int) -> tuple[int, int, int]:
    x = state.truck.dimensions["w"]
    for package in state.packages[:package_index]:
        x += max(package.dimensions.values())
    return x, 0, 0

def random_initial_state(truck: Truck | list[Truck], packages: list[Package], seed: int | None = None,max_attempts: int = 50) -> State | FleetState:
    if type(max_attempts) is not int or max_attempts <= 0:
        raise ValueError("Max attempts harus berupa bilangan bulat positif.")

    if isinstance(truck, list):
        if not truck:
            raise ValueError("Armada harus memiliki minimal satu truk.")
        # Start jointly: all packages are candidates for any truck; neighborhoods
        # include reassignment moves, so the search optimizes the fleet globally.
        first = random_initial_state(truck[0], packages, seed=seed, max_attempts=max_attempts)
        return FleetState([first, *(State(t, []) for t in truck[1:])])
    rng = random.Random(seed)
    state = State(truck, packages).copy()
    for i, package in enumerate(state.packages):
        package.position = outside_position(state, i)
        package.orientation = rng.choice(ORIENTATIONS).copy()

    order = list(range(len(state.packages)))
    rng.shuffle(order)
    for i in order:
        package = state.packages[i]
        for _ in range(max_attempts):
            package.orientation = rng.choice(ORIENTATIONS).copy()
            width, length, height = package.oriented_dimensions()
            max_x = state.truck.dimensions["w"] - width
            max_y = state.truck.dimensions["l"] - length
            max_z = state.truck.dimensions["h"] - height
            if min(max_x, max_y, max_z) < 0:
                continue

            # Paket harus ada supportnya
            levels = {0}
            for other in state.packages:
                if other.id != package.id and state.is_inside(other) and not other.isFragile:
                    top = state.bounds(other)[1][2]
                    if top <= max_z:
                        levels.add(top)
            package.position = (rng.randint(0, max_x), rng.randint(0, max_y), rng.choice(sorted(levels)))
            if is_valid(state):
                break
            package.position = outside_position(state, i)
    return state

def move_package(state: State, package_index: int, position: tuple[int, int, int]) -> State | None:
    if state.packages[package_index].position == position:
        return None
    neighbor = state.copy()
    neighbor.packages[package_index] = replace(neighbor.packages[package_index], position=position)
    return neighbor if is_valid(neighbor) else None


def swap_packages(state: State, first: int, second: int) -> State | None:
    if first == second or state.packages[first].position == state.packages[second].position:
        return None
    neighbor = state.copy()
    a, b = neighbor.packages[first], neighbor.packages[second]
    a.position, b.position = b.position, a.position
    return neighbor if is_valid(neighbor) else None

# Rotasi 90 derajat terhadap axis x/y/z
def rotate_package(state: State, package_index: int, axis: str) -> State | None:
    if axis not in ("x", "y", "z"):
        raise ValueError("Sumbu rotasi harus x, y, atau z.")
    first, second = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[axis]
    neighbor = state.copy()
    package = neighbor.packages[package_index]
    for side, current_axis in package.orientation.items():
        if current_axis == first:
            package.orientation[side] = second
        elif current_axis == second:
            package.orientation[side] = first
    return neighbor if is_valid(neighbor) else None

# Hasilkan neighbor valid satu per satu
def generate_neighbors(state: State) -> Iterator[State]:
    if isinstance(state, FleetState):
        # Geometric moves inside each vehicle.
        for truck_index, truck_state in enumerate(state.trucks):
            for neighbor in generate_neighbors(truck_state):
                candidate = state.copy()
                candidate.trucks[truck_index] = neighbor
                yield candidate
        # Reassign a package to another truck, testing feasible placements there.
        for source_index, source in enumerate(state.trucks):
            for package_index, package in enumerate(source.packages):
                for target_index, target in enumerate(state.trucks):
                    if source_index == target_index:
                        continue
                    base = state.copy()
                    moving = base.trucks[source_index].packages.pop(package_index)
                    moving.position = outside_position(base.trucks[source_index], package_index)
                    base.trucks[target_index].packages.append(moving)
                    destination = base.trucks[target_index]
                    i = len(destination.packages) - 1
                    width, length, height = moving.oriented_dimensions()
                    dims = destination.truck.dimensions
                    for x in range(max(0, dims['w'] - width + 1)):
                        for y in range(max(0, dims['l'] - length + 1)):
                            for z in range(max(0, dims['h'] - height + 1)):
                                trial = base.copy()
                                trial.trucks[target_index].packages[i].position = (x, y, z)
                                if is_valid(trial):
                                    yield trial
        return
    for i, package in enumerate(state.packages):
        width, length, height = package.oriented_dimensions()
        for x in range(state.truck.dimensions["w"] - width + 1):
            for y in range(state.truck.dimensions["l"] - length + 1):
                for z in range(state.truck.dimensions["h"] - height + 1):
                    neighbor = move_package(state, i, (x, y, z))
                    if neighbor is not None:
                        yield neighbor

        if state.is_inside(package):
            neighbor = move_package(state, i, outside_position(state, i))
            if neighbor is not None:
                yield neighbor

        for axis in ("x", "y", "z"):
            neighbor = rotate_package(state, i, axis)
            if neighbor is not None:
                yield neighbor

        for j in range(i + 1, len(state.packages)):
            neighbor = swap_packages(state, i, j)
            if neighbor is not None:
                yield neighbor
