from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import plotly.graph_objects as go


def _numeric_series(values: Any) -> list[float | None] | None:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return None

    series: list[float | None] = []
    for value in values:
        if isinstance(value, bool):
            series.append(None)
        elif isinstance(value, (int, float)) and math.isfinite(value):
            series.append(float(value))
        else:
            series.append(None)

    return series if any(value is not None for value in series) else None


def _line_figure(
    series: list[float | None],
    title: str,
    y_title: str,
    name: str,
    color: str,
    dark: bool,
) -> go.Figure:
    figure = go.Figure(
        go.Scatter(
            x=list(range(1, len(series) + 1)),
            y=series,
            mode="lines+markers",
            name=name,
            line={"color": color},
            connectgaps=False,
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="Iterasi",
        yaxis_title=y_title,
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        margin={"l": 40, "r": 20, "t": 55, "b": 40},
    )
    return figure


def objective_figure(history: Mapping[str, Any], dark: bool = False) -> go.Figure | None:
    series = _numeric_series(history.get("objective"))
    if series is None:
        return None

    return _line_figure(
        series,
        "Objective Value terhadap Iterasi",
        "Objective Value",
        "Objective",
        "#1f77b4",
        dark,
    )


def annealing_probability_figure(history: Mapping[str, Any], dark: bool = False) -> go.Figure | None:
    values = history.get("exp_delta_e_over_t")
    series = _numeric_series(values)
    if series is None:
        return None

    return _line_figure(
        series,
        "e^(ΔE / T) terhadap Iterasi",
        "e^(ΔE / T)",
        "e^(ΔE / T)",
        "#d62728",
        dark,
    )


def genetic_objective_figure(history: Mapping[str, Any], dark: bool = False) -> go.Figure | None:
    maximum = _numeric_series(history.get("max_objective"))
    average = _numeric_series(history.get("average_objective"))

    if maximum is None and average is None:
        return None

    figure = go.Figure()

    if maximum is not None:
        figure.add_trace(
            go.Scatter(
                x=list(range(1, len(maximum) + 1)),
                y=maximum,
                mode="lines+markers",
                name="Maximum Objective",
                line={"color": "#2ca02c"},
                connectgaps=False,
            )
        )

    if average is not None:
        figure.add_trace(
            go.Scatter(
                x=list(range(1, len(average) + 1)),
                y=average,
                mode="lines+markers",
                name="Average Objective",
                line={"color": "#ff7f0e"},
                connectgaps=False,
            )
        )

    figure.update_layout(
        title="Objective Populasi terhadap Iterasi",
        xaxis_title="Iterasi",
        yaxis_title="Objective Value",
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        margin={"l": 40, "r": 20, "t": 55, "b": 40},
    )
    return figure
