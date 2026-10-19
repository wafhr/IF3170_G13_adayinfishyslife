from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence
from typing import Any

import plotly.graph_objects as go


PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _dimensions(data: Any) -> tuple[float, float, float] | None:
    if not isinstance(data, Mapping):
        return None

    dimensions = tuple(_number(data.get(axis)) for axis in ("w", "l", "h"))
    if any(value is None or value <= 0 for value in dimensions):
        return None

    return dimensions  


def _position(data: Any) -> tuple[float, float, float] | None:
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)) or len(data) != 3:
        return None

    position = tuple(_number(value) for value in data)
    if any(value is None for value in position):
        return None

    return position  


def _oriented_dimensions(package: Mapping[str, Any]) -> tuple[float, float, float] | None:
    dimensions = _dimensions(package.get("dimensions"))
    if dimensions is None:
        return None

    orientation = package.get("orientation")
    if not isinstance(orientation, Mapping):
        return dimensions

    if set(orientation) != {"w", "l", "h"} or set(orientation.values()) != {"x", "y", "z"}:
        return dimensions

    original = dict(zip(("w", "l", "h"), dimensions))
    axis_sizes = {orientation[side]: original[side] for side in ("w", "l", "h")}
    return axis_sizes["x"], axis_sizes["y"], axis_sizes["z"]


def _status(
    position: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    truck_dimensions: tuple[float, float, float],
) -> str:
    end = tuple(position[axis] + dimensions[axis] for axis in range(3))

    if all(0 <= position[axis] and end[axis] <= truck_dimensions[axis] for axis in range(3)):
        return "Di dalam truk"

    if any(end[axis] <= 0 or position[axis] >= truck_dimensions[axis] for axis in range(3)):
        return "Di luar truk"

    return "Melewati dinding (state tidak valid)"


def _add_truck_wireframe(
    figure: go.Figure,
    truck_dimensions: tuple[float, float, float],
    color: str,
    offset_x: float = 0,
    name: str = "Container truk",
) -> None:
    width, length, height = truck_dimensions
    corners = [
        (offset_x, 0, 0),
        (offset_x + width, 0, 0),
        (offset_x + width, length, 0),
        (offset_x, length, 0),
        (offset_x, 0, height),
        (offset_x + width, 0, height),
        (offset_x + width, length, height),
        (offset_x, length, height),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    x_values: list[float | None] = []
    y_values: list[float | None] = []
    z_values: list[float | None] = []

    for start, end in edges:
        x_values.extend([corners[start][0], corners[end][0], None])
        y_values.extend([corners[start][1], corners[end][1], None])
        z_values.extend([corners[start][2], corners[end][2], None])

    figure.add_trace(
        go.Scatter3d(
            x=x_values,
            y=y_values,
            z=z_values,
            mode="lines",
            line={"color": color, "width": 5},
            name=name,
            hoverinfo="skip",
        )
    )


def _add_package_box(
    figure: go.Figure,
    position: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    package_id: str,
    color: str,
    opacity: float,
    hover_text: str,
    selected: bool = False,
    offset_x: float = 0,
) -> None:
    x0, y0, z0 = position
    x0 += offset_x
    width, length, height = dimensions
    x1, y1, z1 = x0 + width, y0 + length, z0 + height

    x = [x0, x1, x1, x0, x0, x1, x1, x0]
    y = [y0, y0, y1, y1, y0, y0, y1, y1]
    z = [z0, z0, z0, z0, z1, z1, z1, z1]

    faces = [
        (0, 1, 2), (0, 2, 3),
        (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 4), (3, 4, 0),
    ]

    figure.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=[face[0] for face in faces],
            j=[face[1] for face in faces],
            k=[face[2] for face in faces],
            color=color,
            opacity=opacity,
            flatshading=True,
            name=package_id,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    for start, end in edges:
        edge_x.extend([x[start], x[end], None])
        edge_y.extend([y[start], y[end], None])
        edge_z.extend([z[start], z[end], None])

    figure.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line={"color": "#111827" if selected else "#263244", "width": 7 if selected else 4},
            showlegend=False,
            hoverinfo="skip",
        )
    )


def _state_for_view(problem: Any, state: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(problem, Mapping):
        return None, "Problem config tidak tersedia atau formatnya bukan object JSON."
    if not isinstance(state, Mapping):
        return None, "State tidak tersedia atau formatnya bukan object JSON."

    truck = problem.get("truck")
    packages = problem.get("packages")
    placements = state.get("packages")
    if not isinstance(truck, Mapping) or not isinstance(packages, list):
        return None, "Problem config harus memiliki truck dan packages."
    if not isinstance(placements, list):
        return None, "State harus memiliki daftar posisi packages."

    placement_by_id = {
        placement.get("id"): placement
        for placement in placements
        if isinstance(placement, Mapping) and isinstance(placement.get("id"), str)
    }
    merged_packages = []
    missing_ids = []
    for package in packages:
        if not isinstance(package, Mapping) or not isinstance(package.get("id"), str):
            continue
        placement = placement_by_id.get(package["id"])
        if placement is None:
            missing_ids.append(package["id"])
        merged = dict(package)
        merged["position"] = placement.get("position") if placement is not None else None
        merged["orientation"] = placement.get("orientation") if placement is not None else None
        merged["isFragile"] = package.get("is_fragile", False)
        merged["truck_index"] = placement.get("truck_index", 1) if placement is not None else 1
        merged_packages.append(merged)

    warning = None
    if missing_ids:
        warning = f"Posisi tidak tersedia untuk: {', '.join(missing_ids)}."
    return {"truck": truck, "packages": merged_packages}, warning


def state_package_rows(problem: Any, state: Any) -> tuple[list[dict[str, Any]] | None, str | None]:
    state_data, warning = _state_for_view(problem, state)
    if state_data is None:
        return None, warning

    truck = state_data.get("truck")
    truck_dimensions = _dimensions(truck.get("dimensions")) if isinstance(truck, Mapping) else None
    if truck_dimensions is None:
        return None, "Dimensi truk tidak lengkap atau tidak valid."

    packages = state_data.get("packages")
    if not isinstance(packages, list):
        return None, "Field `packages` pada state tidak tersedia atau bukan list."

    rows = []
    invalid_count = 0
    for index, package in enumerate(packages):
        if not isinstance(package, Mapping):
            invalid_count += 1
            continue
        position = _position(package.get("position"))
        dimensions = _oriented_dimensions(package)
        if position is None or dimensions is None:
            status = "Data tidak valid"
            invalid_count += 1
        else:
            status = _status(position, dimensions, truck_dimensions)
        rows.append({
            "id": str(package.get("id", f"Paket {index + 1}")),
            "position": position,
            "dimensions": package.get("dimensions"),
            "oriented_dimensions": dimensions,
            "orientation": package.get("orientation"),
            "weight": package.get("weight"),
            "value": package.get("value"),
            "is_fragile": package.get("isFragile", package.get("is_fragile", False)),
            "eta": package.get("eta"),
            "status": status,
            "index": index,
            "truck_index": package.get("truck_index", 1),
        })

    if invalid_count:
        invalid_warning = f"{invalid_count} paket memiliki posisi atau dimensi yang tidak valid."
        warning = f"{warning} {invalid_warning}" if warning else invalid_warning
    return rows, warning


def create_state_figure(
    problem: Any,
    state: Any,
    dark: bool = False,
    show_outside: bool = True,
    package_filter: str = "Semua",
    selected_package_id: str | None = None,
    height: int = 440,
    truck_index: int | None = None,
) -> tuple[go.Figure | None, str | None]:
    rows, warning = state_package_rows(problem, state)
    if rows is None:
        return None, warning

    truck = problem.get("truck") if isinstance(problem, Mapping) else None
    if not isinstance(truck, Mapping):
        return None, "Field `truck` pada state tidak tersedia."

    truck_dimensions = _dimensions(truck.get("dimensions"))
    if truck_dimensions is None:
        return None, "Dimensi truk tidak lengkap atau tidak valid."

    figure = go.Figure()
    text_color = "#f5f5f5" if dark else "#111111"
    truck_color = "#f5f5f5" if dark else "#374151"
    truck_count = state.get("truck_count", 1) if isinstance(state, Mapping) else 1
    if truck_index is None and truck_count > 1:
        height = max(height, min(900, 360 + 90 * truck_count))
    displayed_trucks = range(1, truck_count + 1) if truck_index is None else (truck_index,)
    gap = truck_dimensions[0] * 0.25
    truck_offsets = {index: (index - 1) * (truck_dimensions[0] + gap) for index in displayed_trucks}
    for index, offset in truck_offsets.items():
        _add_truck_wireframe(figure, truck_dimensions, truck_color, offset, f"Truk {index}")

    extent_x = [min(truck_offsets.values()), max(truck_offsets.values()) + truck_dimensions[0]]
    extent_y = [0.0, truck_dimensions[1]]
    extent_z = [0.0, truck_dimensions[2]]
    package_labels = []
    hidden_outside = 0
    outside_rows = [row for row in rows if row["status"] != "Di dalam truk"]
    outside_width = max(
        (row["oriented_dimensions"][0] for row in outside_rows if row["oriented_dimensions"] is not None),
        default=1.0,
    )
    outside_start_x = max(truck_offsets.values()) + truck_dimensions[0] + 2 * gap
    outside_slot = 0
    for row in rows:
        row_truck_index = row.get("truck_index", 1)
        if row_truck_index not in truck_offsets:
            continue
        offset_x = truck_offsets[row_truck_index]
        source_position = row["position"]
        dimensions = row["oriented_dimensions"]
        if source_position is None or dimensions is None:
            continue

        status = row["status"]
        if package_filter == "Di dalam truk" and status != "Di dalam truk":
            continue
        if package_filter == "Di luar truk" and status == "Di dalam truk":
            continue
        if status == "Di luar truk" and not show_outside:
            hidden_outside += 1
            continue
        package_id = row["id"]
        # Invalid/outside coordinates are local to a truck and can overlap a
        # neighboring truck after fleet offsets are applied. Show these packages
        # in a dedicated staging lane beyond the last truck instead.
        if status != "Di dalam truk":
            position = (outside_start_x + outside_slot * (outside_width + gap), 0.0, 0.0)
            offset_x = 0.0
            outside_slot += 1
        else:
            position = source_position

        hover_text = (
            f"<b>{html.escape(package_id)}</b><br>"
            f"Posisi lokal: {html.escape(str(source_position))}<br>"
            f"Truk tujuan: {row_truck_index}<br>"
            f"Berat: {html.escape(str(row['weight']))}<br>"
            f"Value: {html.escape(str(row['value']))}"
        )

        is_selected = package_id == selected_package_id
        other_selected = selected_package_id is not None and not is_selected
        _add_package_box(
            figure=figure,
            position=position,
            dimensions=dimensions,
            package_id=package_id,
            color="#ff00c8" if is_selected else ("#9ca3af" if other_selected else PALETTE[row["index"] % len(PALETTE)]),
            opacity=0.98 if is_selected else (0.22 if other_selected else (0.82 if status == "Di dalam truk" else 0.45)),
            hover_text=hover_text,
            selected=is_selected,
            offset_x=offset_x,
        )
        package_labels.append({
            "x": offset_x + position[0] + dimensions[0] / 2,
            "y": position[1] + dimensions[1] / 2,
            "z": position[2] + dimensions[2] / 2,
            "text": html.escape(package_id),
            "showarrow": False,
            "bgcolor": "#d500a8" if is_selected else "rgba(255,255,255,0.96)",
            "bordercolor": "#111827" if is_selected else "#263244",
            "borderwidth": 2 if is_selected else 1,
            "borderpad": 4,
            "font": {
                "family": "Arial Black" if is_selected else "Arial",
                "size": 13 if is_selected else 11,
                "color": "#ffffff" if is_selected else "#111827",
            },
            "captureevents": False,
        })

        extent_x.extend([offset_x + position[0], offset_x + position[0] + dimensions[0]])
        extent_y.extend([position[1], position[1] + dimensions[1]])
        extent_z.extend([position[2], position[2] + dimensions[2]])

    padding = max(*truck_dimensions, 1.0) * 0.08
    truck_labels = [
        {"x": offset + truck_dimensions[0] / 2, "y": truck_dimensions[1] / 2,
         "z": truck_dimensions[2] + padding * 2, "text": f"Truk {index}",
         "showarrow": False, "font": {"size": 14, "color": text_color}}
        for index, offset in truck_offsets.items()
    ]
    if outside_slot:
        truck_labels.append({
            "x": outside_start_x + max(0, outside_slot - 1) * (outside_width + gap) / 2,
            "y": 0.0,
            "z": truck_dimensions[2] + padding * 2,
            "text": "Paket belum termuat",
            "showarrow": False,
            "font": {"size": 14, "color": text_color},
        })
    figure.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        legend={
            "bgcolor": "rgba(20,20,20,0.75)" if dark else "rgba(255,255,255,0.92)",
            "bordercolor": truck_color,
            "borderwidth": 1,
            "font": {"color": text_color},
        },
        scene={
            "bgcolor": "rgba(0,0,0,0)",
            "annotations": [*truck_labels, *package_labels],
            "xaxis": {"title": "X / Width", "range": [min(extent_x) - padding, max(extent_x) + padding]},
            "yaxis": {"title": "Y / Length", "range": [min(extent_y) - padding, max(extent_y) + padding]},
            "zaxis": {"title": "Z / Height", "range": [min(extent_z) - padding, max(extent_z) + padding]},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.6, "y": 1.6, "z": 1.2}},
        },
    )

    if hidden_outside:
        outside_warning = f"{hidden_outside} paket di luar truk disembunyikan pada mode fokus container."
        warning = f"{warning} {outside_warning}" if warning else outside_warning

    return figure, warning
