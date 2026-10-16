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
) -> None:
    width, length, height = truck_dimensions
    corners = [
        (0, 0, 0),
        (width, 0, 0),
        (width, length, 0),
        (0, length, 0),
        (0, 0, height),
        (width, 0, height),
        (width, length, height),
        (0, length, height),
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
            name="Container truk",
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
) -> None:
    x0, y0, z0 = position
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
            continue
        merged = dict(package)
        merged["position"] = placement.get("position")
        merged["orientation"] = placement.get("orientation")
        merged["isFragile"] = package.get("is_fragile", False)
        merged_packages.append(merged)

    warning = None
    if missing_ids:
        warning = f"Posisi tidak tersedia untuk: {', '.join(missing_ids)}."
    return {"truck": truck, "packages": merged_packages}, warning


def create_state_figure(
    problem: Any,
    state: Any,
    dark: bool = False,
    show_outside: bool = True,
) -> tuple[go.Figure | None, str | None]:
    state, state_warning = _state_for_view(problem, state)
    if state is None:
        return None, state_warning

    truck = state.get("truck")
    if not isinstance(truck, Mapping):
        return None, "Field `truck` pada state tidak tersedia."

    truck_dimensions = _dimensions(truck.get("dimensions"))
    if truck_dimensions is None:
        return None, "Dimensi truk tidak lengkap atau tidak valid."

    packages = state.get("packages")
    if not isinstance(packages, list):
        return None, "Field `packages` pada state tidak tersedia atau bukan list."

    figure = go.Figure()
    text_color = "#f5f5f5" if dark else "#111111"
    truck_color = "#f5f5f5" if dark else "#374151"
    _add_truck_wireframe(figure, truck_dimensions, truck_color)

    extent_x = [0.0, truck_dimensions[0]]
    extent_y = [0.0, truck_dimensions[1]]
    extent_z = [0.0, truck_dimensions[2]]
    skipped = 0
    hidden_outside = 0
    inside_count = 0
    outside_count = 0
    state_value = 0.0

    for index, package in enumerate(packages):
        if not isinstance(package, Mapping):
            skipped += 1
            continue

        position = _position(package.get("position"))
        dimensions = _oriented_dimensions(package)
        if position is None or dimensions is None:
            skipped += 1
            continue

        status = _status(position, dimensions, truck_dimensions)
        if status == "Di dalam truk":
            inside_count += 1
            value = _number(package.get("value"))
            if value is not None:
                state_value += value
        elif status == "Di luar truk":
            outside_count += 1
        if status == "Di luar truk" and not show_outside:
            hidden_outside += 1
            continue
        package_id = str(package.get("id", f"Paket {index + 1}"))
        original_dimensions = package.get("dimensions")
        orientation = package.get("orientation", {"w": "x", "l": "y", "h": "z"})

        hover_text = (
            f"<b>{html.escape(package_id)}</b><br>"
            f"Status: {html.escape(status)}<br>"
            f"Posisi: {tuple(position)}<br>"
            f"Dimensi asli: {html.escape(str(original_dimensions))}<br>"
            f"Dimensi terorientasi: {tuple(dimensions)}<br>"
            f"Orientasi: {html.escape(str(orientation))}<br>"
            f"Value: {html.escape(str(package.get('value', 'Tidak tersedia')))}<br>"
            f"Weight: {html.escape(str(package.get('weight', 'Tidak tersedia')))}<br>"
            f"Fragile: {html.escape(str(package.get('isFragile', False)))}"
        )

        _add_package_box(
            figure=figure,
            position=position,
            dimensions=dimensions,
            package_id=package_id,
            color=PALETTE[index % len(PALETTE)],
            opacity=0.82 if status == "Di dalam truk" else 0.45,
            hover_text=hover_text,
        )
        figure.add_trace(
            go.Scatter3d(
                x=[position[0] + dimensions[0] / 2],
                y=[position[1] + dimensions[1] / 2],
                z=[position[2] + dimensions[2] / 2],
                mode="text",
                text=[package_id],
                textfont={"size": 11, "color": text_color},
                showlegend=False,
                hoverinfo="skip",
            )
        )

        extent_x.extend([position[0], position[0] + dimensions[0]])
        extent_y.extend([position[1], position[1] + dimensions[1]])
        extent_z.extend([position[2], position[2] + dimensions[2]])

    padding = max(*truck_dimensions, 1.0) * 0.08
    figure.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        height=820,
        title={
            "text": (
                f"Nilai state: {state_value:g}"
                f" &nbsp;|&nbsp; Di dalam container: {inside_count} paket"
                f" &nbsp;|&nbsp; Di luar container: {outside_count} paket"
            ),
            "x": 0.02,
            "xanchor": "left",
            "font": {"color": text_color, "size": 16},
        },
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        legend={
            "bgcolor": "rgba(20,20,20,0.75)" if dark else "rgba(255,255,255,0.92)",
            "bordercolor": truck_color,
            "borderwidth": 1,
            "font": {"color": text_color},
        },
        scene={
            "bgcolor": "rgba(0,0,0,0)",
            "xaxis": {"title": "X / Width", "range": [min(extent_x) - padding, max(extent_x) + padding]},
            "yaxis": {"title": "Y / Length", "range": [min(extent_y) - padding, max(extent_y) + padding]},
            "zaxis": {"title": "Z / Height", "range": [min(extent_z) - padding, max(extent_z) + padding]},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.6, "y": 1.6, "z": 1.2}},
        },
    )

    warning = state_warning
    if skipped:
        skipped_warning = f"{skipped} paket tidak divisualisasikan karena data posisi/dimensinya tidak valid."
        warning = f"{warning} {skipped_warning}" if warning else skipped_warning
    if hidden_outside:
        outside_warning = f"{hidden_outside} paket di luar truk disembunyikan pada mode fokus container."
        warning = f"{warning} {outside_warning}" if warning else outside_warning

    return figure, warning
