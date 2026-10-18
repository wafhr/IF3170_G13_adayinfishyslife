from dataclasses import dataclass, field, replace

ORIENTATIONS = (
    {"w": "x", "l": "y", "h": "z"},
    {"w": "x", "l": "z", "h": "y"},
    {"w": "y", "l": "x", "h": "z"},
    {"w": "y", "l": "z", "h": "x"},
    {"w": "z", "l": "x", "h": "y"},
    {"w": "z", "l": "y", "h": "x"},
)

def check_dimensions(width: int, length: int, height: int) -> None:
    if any(type(size) is not int or size <= 0 for size in (width, length, height)):
        raise ValueError("Dimensi harus berupa bilangan bulat positif.")


@dataclass
class Package:
    id: str
    dimensions: dict[str, int]
    value: float
    weight: float
    isFragile: bool = False
    position: tuple[int, int, int] = (0, 0, 0)
    orientation: dict[str, str] = field(default_factory=lambda: {"w": "x", "l": "y", "h": "z"})
    eta: int = 0  # Sisa hari; default 0 untuk pemanggilan lama.

    def __post_init__(self):
        if not isinstance(self.dimensions, dict) or set(self.dimensions) != {"w", "l", "h"}:
            raise ValueError("Dimensions harus berisi w, l, h berupa bilangan bulat positif.")
        check_dimensions(*(self.dimensions[side] for side in ("w", "l", "h")))
        self.dimensions = self.dimensions.copy()
        if type(self.eta) is not int or self.eta < 0:
            raise ValueError("ETA harus berupa bilangan bulat non-negatif (sisa hari).")
        if self.value < 0 or self.weight < 0:
            raise ValueError("Value dan weight paket tidak boleh negatif.")
        if (not isinstance(self.position, tuple) or len(self.position) != 3 or any(type(p) is not int for p in self.position)):
            raise ValueError("Posisi harus berupa tuple tiga bilangan bulat")
        if not isinstance(self.orientation, dict) or self.orientation not in ORIENTATIONS:
            raise ValueError("Orientasi harus merupakan permutasi x, y, dan z.")
        self.orientation = self.orientation.copy()

    def oriented_dimensions(self) -> tuple[int, int, int]:
        axis_sizes = {self.orientation[side]: size for side, size in self.dimensions.items()}
        return axis_sizes["x"], axis_sizes["y"], axis_sizes["z"]
    
@dataclass
class Truck:
    dimensions: dict[str, int]
    max_capacity: float
    
    def __post_init__(self):
        if not isinstance(self.dimensions, dict) or set(self.dimensions) != {"w", "l", "h"}:
            raise ValueError("Dimensions harus berisi w, l, h berupa bilangan bulat positif.")
        check_dimensions(*(self.dimensions[side] for side in ("w", "l", "h")))
        self.dimensions = self.dimensions.copy()
        if self.max_capacity < 0:
            raise ValueError("Kapasitas truk tidak boleh negatif.")
        
@dataclass
class State:
    truck: Truck
    packages: list[Package]

    def __post_init__(self):
        ids = [package.id for package in self.packages]
        if len(set(ids)) != len(ids):
            raise ValueError("ID paket harus unik.")

    def copy(self) -> "State":
        return State(replace(self.truck), [replace(package) for package in self.packages])

    def bounds(self, package: Package):
        x, y, z = package.position
        width, length, height = package.oriented_dimensions()
        return (x, y, z), (x + width, y + length, z + height)

    def is_inside(self, package: Package) -> bool:
        start, end = self.bounds(package)
        limits = tuple(self.truck.dimensions[side] for side in ("w", "l", "h"))
        return all(0 <= start[i] and end[i] <= limits[i] for i in range(3))

    def is_outside(self, package: Package) -> bool:
        start, end = self.bounds(package)
        limits = tuple(self.truck.dimensions[side] for side in ("w", "l", "h"))
        return any(end[i] <= 0 or start[i] >= limits[i] for i in range(3))

    # Petakan semua sel unit truk ke ID paket, atau None jika kosong
    def space_map(self) -> dict[tuple[int, int, int], str | None]:
        cells = {(x, y, z): None for x in range(self.truck.dimensions["w"]) for y in range(self.truck.dimensions["l"]) for z in range(self.truck.dimensions["h"])}
        for package in self.packages:
            if self.is_outside(package):
                continue
            if not self.is_inside(package):
                raise ValueError(f"Paket {package.id} menembus dinding truk.")
            (x1, y1, z1), (x2, y2, z2) = self.bounds(package)
            for x in range(x1, x2):
                for y in range(y1, y2):
                    for z in range(z1, z2):
                        if cells[x, y, z] is not None:
                            raise ValueError(f"Paket {package.id} overlap dengan paket lain.")
                        cells[x, y, z] = package.id
        return cells
