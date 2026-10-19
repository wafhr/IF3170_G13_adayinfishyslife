from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from html import escape
from typing import Any

import streamlit as st

from core.evaluation import OBJECTIVE_LABELS
from visualization.controls import display_algorithm
from visualization.plots import annealing_probability_figure, genetic_objective_figure, objective_figure
from visualization.state_view import create_state_figure, state_package_rows


def history_from(data: Mapping[str, Any]) -> Mapping[str, Any]:
    history = data.get("history")
    return history if isinstance(history, Mapping) else {}


def is_simulated_annealing(algorithm: str) -> bool:
    normalized = algorithm.casefold().replace("_", " ").replace("-", " ")
    return "simulated" in normalized or normalized.strip() == "sa"


def is_genetic_algorithm(algorithm: str) -> bool:
    normalized = algorithm.casefold().replace("_", " ").replace("-", " ")
    return "genetic" in normalized or normalized.strip() == "ga"


def format_number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "Tidak tersedia"
    return f"{value:g}"


def format_position(value: Any) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return "Tidak tersedia"
    return "(" + ", ".join(format_number(part) for part in value) + ")"


def display_metric(column: Any, label: str, value: Any, suffix: str = "") -> None:
    column.metric(label, "Tidak tersedia" if value is None else f"{value}{suffix}")


def state_metrics(container: Any, rows: list[dict[str, Any]], capacity: Any) -> None:
    inside = [row for row in rows if row["status"] == "Di dalam truk"]
    total_value = sum(
        value for row in inside
        if isinstance((value := row["value"]), (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    )
    loaded_weight = sum(
        weight for row in inside
        if isinstance((weight := row["weight"]), (int, float)) and not isinstance(weight, bool)
        and math.isfinite(weight)
    )
    with container:
        columns = st.columns(3)
        columns[0].metric("Total value", format_number(total_value))
        columns[1].metric("Paket di dalam / total", f"{len(inside)} / {len(rows)}")
        columns[2].metric("Berat / kapasitas", f"{format_number(loaded_weight)}/{format_number(capacity)}")


def table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "ID": row["id"],
            "Truk": row.get("truck_index", 1),
            "Posisi (x, y, z)": format_position(row["position"]),
            "Berat": row["weight"],
            "Value": row["value"],
            "Fragile": "Ya" if row["is_fragile"] else "Tidak",
            "ETA": row["eta"],
            "Status": row["status"],
        }
        for row in rows
    ]


def render_package_details(container: Any, package_id: str | None, rows: list[dict[str, Any]], state_label: str) -> None:
    with container:
        st.markdown('<div class="package-inspector-heading">Detail paket</div>', unsafe_allow_html=True)
        row = next((item for item in rows if item["id"] == package_id), None)
        if row is None:
            body = '<div class="package-inspector__hint">Pilih baris paket pada tabel untuk melihat detailnya.</div>'
        else:
            title = escape(f"{row['id']} · {state_label}")
            details = (
                ("Posisi", format_position(row["position"])),
                ("Truk", str(row.get("truck_index", 1))),
                ("Berat", format_number(row["weight"])),
                ("Value", format_number(row["value"])),
                ("Fragile", "Ya" if row["is_fragile"] else "Tidak"),
                ("ETA", str(row["eta"])),
                ("Status", str(row["status"])),
            )
            rows_html = "".join(
                f'<div class="package-inspector__row"><span>{escape(label)}</span>'
                f'<strong>{escape(value)}</strong></div>'
                for label, value in details
            )
            body = (
                f'<div class="package-inspector__title">{title}</div>'
                f'<div class="package-inspector__grid">{rows_html}</div>'
            )

        st.markdown(f'<div class="package-inspector-card">{body}</div>', unsafe_allow_html=True)


def visible_package_rows(rows: list[dict[str, Any]], package_filter: str) -> list[dict[str, Any]]:
    if package_filter == "Di dalam truk":
        return [row for row in rows if row["status"] == "Di dalam truk"]
    if package_filter == "Di luar truk":
        return [row for row in rows if row["status"] != "Di dalam truk"]
    return rows


def render_state_chart(
    container: Any,
    title: str,
    key: str,
    problem: Mapping[str, Any],
    state: Any,
    package_filter: str,
    selected_package_id: str | None,
    truck_index: int | None,
) -> None:
    with container:
        st.subheader(title)
        figure, warning = create_state_figure(
            problem,
            state,
            package_filter=package_filter,
            selected_package_id=selected_package_id,
            truck_index=truck_index,
        )
        if figure is None:
            st.warning(warning or "State tidak dapat divisualisasikan.")
            return
        st.plotly_chart(figure, width="stretch", theme=None, key=key)
        if warning:
            st.warning(warning)


def render_state_and_packages(
    result: Mapping[str, Any],
    problem: Mapping[str, Any],
    run_key: str,
) -> None:
    if is_genetic_algorithm(str(result.get("algorithm", ""))):
        st.caption(
            "Untuk GA, Initial State menampilkan satu individu acak dari populasi awal. "
            "Grafik menampilkan nilai maksimum dan rata-rata seluruh populasi."
        )
    initial_rows, initial_warning = state_package_rows(problem, result.get("initial_state"))
    final_rows, final_warning = state_package_rows(problem, result.get("final_state"))
    if initial_rows is None or final_rows is None:
        st.error(initial_warning or final_warning or "State hasil tidak dapat dibaca.")
        return

    controls = st.columns([2, 1])
    mode = controls[0].segmented_control(
        "State", ["Initial", "Final"], default="Final", key="state_view_mode", width="stretch"
    )
    package_filter = controls[1].selectbox(
        "Filter paket", ["Semua", "Di dalam truk", "Di luar truk"], key="package_filter",
        help="Filter berlaku pada grafik 3D dan tabel paket.",
    )
    capacity = problem.get("truck", {}).get("max_capacity") if isinstance(problem.get("truck"), Mapping) else None
    active_state = result.get("initial_state") if mode == "Initial" else result.get("final_state")
    truck_count = active_state.get("truck_count", 1) if isinstance(active_state, Mapping) else 1
    truck_index = None if truck_count > 1 else 1
    selected_id = st.session_state.get("selected_package_id")

    if mode == "Initial":
        state_rows = initial_rows
    else:
        state_rows = final_rows

    selected_truck_rows = (
        state_rows if truck_index is None
        else [row for row in state_rows if row.get("truck_index", 1) == truck_index]
    )
    displayed_rows = visible_package_rows(selected_truck_rows, package_filter)
    visible_ids = {row["id"] for row in displayed_rows}
    if selected_id not in visible_ids:
        selected_id = None
        st.session_state.pop("selected_package_id", None)

    state_label = "Initial" if mode == "Initial" else "Final"
    state_value = result.get("initial_state") if mode == "Initial" else result.get("final_state")
    fleet_capacity = capacity * truck_count if isinstance(capacity, (int, float)) else capacity
    state_metrics(st.container(), state_rows, fleet_capacity)
    chart_column, detail_column = st.columns([3, 1])
    render_state_chart(
        chart_column, f"{state_label} State", f"{run_key}_{state_label.lower()}_state", problem,
        state_value, package_filter, selected_id, truck_index,
    )
    render_package_details(detail_column, selected_id, state_rows, state_label)

    if mode == "Initial":
        warnings = (("Initial state", initial_warning),)
    else:
        warnings = (("Final state", final_warning),)
    for label, warning in warnings:
        if warning:
            st.warning(f"{label}: {warning}")

    st.subheader("Daftar paket")
    table_key = f"packages_{run_key}_{mode}_{package_filter}"
    table = st.dataframe(
        table_rows(displayed_rows),
        column_order=("ID", "Truk", "Posisi (x, y, z)", "Berat", "Value", "Fragile", "ETA", "Status"),
        hide_index=True,
        height=220,
        width="stretch",
        key=table_key,
        on_select="rerun",
        selection_mode="single-row",
    )
    selected_indices = table.selection.rows
    if selected_indices:
        new_selected_id = displayed_rows[selected_indices[0]]["id"]
        if new_selected_id != st.session_state.get("selected_package_id"):
            st.session_state["selected_package_id"] = new_selected_id
            st.rerun()


def render_graphs(history: Mapping[str, Any], algorithm: str, run_key: str) -> None:
    if not history:
        st.warning("Field `history` tidak tersedia atau bukan object. Grafik tidak dapat dibuat.")
        return

    objective_plot = objective_figure(history)
    if objective_plot is None and not is_genetic_algorithm(algorithm):
        st.warning("Data `history.objective` tidak tersedia atau kosong.")
    elif objective_plot is not None:
        st.plotly_chart(objective_plot, width="stretch", theme=None, key=f"{run_key}_objective")

    if is_simulated_annealing(algorithm):
        probability_plot = annealing_probability_figure(history)
        if probability_plot is None:
            st.warning("Data `history.exp_delta_e_over_t` tidak tersedia untuk grafik probabilitas penerimaan.")
        else:
            st.plotly_chart(probability_plot, width="stretch", theme=None, key=f"{run_key}_annealing")

    if is_genetic_algorithm(algorithm):
        st.caption(
            "Garis maksimum pada iterasi 0 adalah individu terbaik dari populasi acak awal. "
            "Jika garis rata-rata cepat menyusulnya, populasi telah konvergen; generasi berikutnya "
            "bisa tetap datar walaupun proses masih berjalan."
        )
        genetic_plot = genetic_objective_figure(history)
        if genetic_plot is None:
            st.warning("Riwayat objective maksimum/rata-rata populasi GA tidak tersedia.")
        else:
            st.plotly_chart(genetic_plot, width="stretch", theme=None, key=f"{run_key}_genetic")


def render_algorithm_statistics(
    result: Mapping[str, Any], history: Mapping[str, Any], algorithm: str,
) -> None:
    if is_simulated_annealing(algorithm):
        st.metric(
            "Frekuensi stuck", format_number(history.get("stuck_frequency")),
            help="Jumlah iterasi saat tidak ada neighbor dengan objective lebih tinggi.",
        )
    elif is_genetic_algorithm(algorithm):
        parameters = result.get("parameters")
        population_size = result.get("population_size")
        if population_size is None and isinstance(parameters, Mapping):
            population_size = parameters.get("population_size")
        st.metric("Ukuran populasi", format_number(population_size))
    elif algorithm.casefold().replace("-", "_").replace(" ", "_") == "random_restart":
        st.metric("Jumlah restart", format_number(history.get("restarts")))
        iterations_per_run = history.get("iterations_per_run")
        if isinstance(iterations_per_run, list) and iterations_per_run:
            st.caption("Iterasi per run · Awal = sebelum restart pertama")
            st.dataframe(
                [
                    {
                        "Run": "Awal" if index == 0 else f"Restart {index}",
                        "Jumlah iterasi": iterations,
                    }
                    for index, iterations in enumerate(iterations_per_run)
                ],
                hide_index=True,
                width=360,
                height=min(32 + 28 * len(iterations_per_run) + 3, 260),
                row_height=28,
                column_config={
                    "Run": st.column_config.TextColumn("Run", width=180),
                    "Jumlah iterasi": st.column_config.NumberColumn(
                        "Jumlah iterasi", width=160, alignment="center", format="%d",
                    ),
                },
                alt="Jumlah iterasi run awal dan setiap restart",
            )
        else:
            st.info("Data iterasi per run tidak tersedia.")


def render_result(result: Mapping[str, Any], problem: Mapping[str, Any]) -> None:
    algorithm = str(result.get("algorithm", "Tidak diketahui"))
    history = history_from(result)
    run_name = str(result.get("run_name", "Run terbaru"))
    run_key = f"{algorithm}_{run_name}"
    iterations = result.get("iterations")
    if iterations is None and isinstance(history.get("objective"), Sequence):
        iterations = len(history["objective"])
    st.header(display_algorithm(algorithm))

    columns = st.columns(3)
    objective_name = result.get("parameters", {}).get("objective", "value")
    objective_label = OBJECTIVE_LABELS.get(objective_name, "Objective")
    display_metric(columns[0], f"Nilai objective · {objective_label}", result.get("final_objective"))
    display_metric(columns[1], "Durasi", result.get("duration_seconds", result.get("duration")), " s")
    display_metric(columns[2], "Jumlah iterasi", iterations)
    render_algorithm_statistics(result, history, algorithm)

    state_tab, graph_tab = st.tabs(["State & Paket", "Grafik"])
    with state_tab:
        render_state_and_packages(result, problem, run_key)
    with graph_tab:
        render_graphs(history, algorithm, run_key)
