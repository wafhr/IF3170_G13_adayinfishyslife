from __future__ import annotations

import json
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = PROJECT_ROOT / "src"
CONFIG_DIRECTORY = PROJECT_ROOT / "config"
OUTPUT_DIRECTORY = PROJECT_ROOT / "output"
DEFAULT_PROBLEM_PATH = CONFIG_DIRECTORY / "problem.json"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from core.evaluation import OBJECTIVE_LABELS
from experiments.output import write_result
from experiments.runner import ALGORITHM_NAMES, run_selected_experiment
from visualization.plots import annealing_probability_figure, genetic_objective_figure, objective_figure
from visualization.state_view import create_state_figure, state_package_rows


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
SEED_DEFAULT = 42
PARAMETER_DEFAULTS = {
    "simulated_annealing": {
        "initial_temperature": 100.0,
        "cooling_rate": 0.95,
        "max_iterations": 50,
    },
    "random_restart": {"max_restarts": 10, "max_attempts": 50},
    "stochastic": {"nmax": 100},
    "sideways": {"max_sideways": 100},
    "genetic_algorithm": {"population_size": 40, "iterations": 100},
}


def load_problem(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open(encoding="utf-8") as file:
            problem = json.load(file)
    except OSError:
        return None, f"Problem config tidak dapat dibaca: {path}"
    except UnicodeDecodeError:
        return None, f"Problem config bukan file teks UTF-8 yang valid: {path}"
    except json.JSONDecodeError as error:
        return None, f"JSON problem config tidak valid pada baris {error.lineno}, kolom {error.colno}."

    if not isinstance(problem, dict):
        return None, "Problem config harus berupa object JSON."
    return problem, None


def choose_file(*, save: bool, initial_directory: Path, initial_file: str = "") -> str | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if save:
            return filedialog.asksaveasfilename(
                title="Pilih lokasi dan nama file output",
                initialdir=str(initial_directory),
                initialfile=initial_file,
                defaultextension=".json",
                filetypes=(("JSON", "*.json"),),
            ) or None
        return filedialog.askopenfilename(
            title="Pilih file problem",
            initialdir=str(initial_directory),
            filetypes=(("JSON", "*.json"),),
        ) or None
    finally:
        root.destroy()


def display_algorithm(algorithm: str) -> str:
    if algorithm == "steepest_ascent":
        return "Hill Climbing (Steepest Ascent)"
    return algorithm.replace("_", " ").replace("-", " ").title()


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


def clear_last_run() -> None:
    st.session_state.pop("last_run", None)
    st.session_state.pop("last_problem", None)
    st.session_state.pop("selected_package_id", None)


def render_selected_file(path_text: str) -> None:
    path = Path(path_text)
    try:
        folder = path.parent.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        parent_parts = path.parent.parts
        folder = "…/" + "/".join(parent_parts[-2:])
    st.sidebar.markdown(
        f'<div class="selected-file" title="{escape(path_text)}">'
        f'<strong>{escape(path.name)}</strong><span>{escape(folder)}</span></div>',
        unsafe_allow_html=True,
    )


def reset_settings() -> None:
    st.session_state["problem_path"] = str(DEFAULT_PROBLEM_PATH)
    st.session_state["output_path"] = str(OUTPUT_DIRECTORY / f"hasil_{datetime.now():%Y%m%d_%H%M%S_%f}.json")
    st.session_state["run_algorithm"] = "steepest_ascent"
    st.session_state["objective_function"] = "value"
    st.session_state["run_seed"] = SEED_DEFAULT
    for algorithm, defaults in PARAMETER_DEFAULTS.items():
        for parameter, value in defaults.items():
            st.session_state[f"parameter_{algorithm}_{parameter}"] = value
    clear_last_run()


def parameter_input(
    algorithm: str,
    name: str,
    label: str,
    default: int | float,
    help_text: str,
    *,
    minimum: int | float,
    step: int | float,
    maximum: int | float | None = None,
) -> int | float:
    key = f"parameter_{algorithm}_{name}"
    st.session_state.setdefault(key, default)
    arguments: dict[str, Any] = {
        "min_value": minimum,
        "step": step,
        "key": key,
        "help": help_text,
    }
    if maximum is not None:
        arguments["max_value"] = maximum
    return st.sidebar.number_input(label, **arguments)


def run_controls() -> tuple[str, str, str, int, dict[str, Any], bool]:
    st.session_state.setdefault("problem_path", str(DEFAULT_PROBLEM_PATH))
    st.session_state.setdefault(
        "output_path",
        str(OUTPUT_DIRECTORY / f"hasil_{datetime.now():%Y%m%d_%H%M%S_%f}.json"),
    )

    st.sidebar.subheader("Problem")
    if st.sidebar.button("Pilih file problem JSON", use_container_width=True):
        try:
            selected = choose_file(save=False, initial_directory=CONFIG_DIRECTORY)
        except Exception as error:
            st.sidebar.error(f"File manager tidak dapat dibuka: {error}")
        else:
            if selected:
                st.session_state["problem_path"] = selected
                clear_last_run()
    problem_path = st.session_state["problem_path"]
    render_selected_file(problem_path)

    st.sidebar.subheader("Algoritma")
    algorithm = st.sidebar.selectbox(
        "Algoritma",
        ALGORITHM_NAMES,
        index=ALGORITHM_NAMES.index("steepest_ascent"),
        format_func=display_algorithm,
        key="run_algorithm",
        on_change=clear_last_run,
    )

    st.session_state.setdefault("run_seed", SEED_DEFAULT)
    parameters: dict[str, Any] = {}
    parameters["objective"] = st.sidebar.selectbox(
        "Fungsi objective",
        tuple(OBJECTIVE_LABELS),
        format_func=OBJECTIVE_LABELS.get,
        key="objective_function",
        help="Pilih fungsi objective yang akan dipakai untuk menilai dan mengoptimalkan state.",
    )
    if parameters["objective"] == "eta":
        st.sidebar.latex(r"f(S) = \sum_{i \in P_m} \frac{1}{\mathrm{ETA}_i + 1}")
    else:
        st.sidebar.latex(r"f(S) = \sum_{i \in P_m} \mathrm{Value}_i")
    seed = st.sidebar.number_input(
        "Seed", min_value=0, step=1, key="run_seed",
        help="Seed membuat eksperimen acak dapat diulang dengan hasil yang sama.",
    )
    if algorithm == "simulated_annealing":
        parameters["initial_temperature"] = parameter_input(
            algorithm, "initial_temperature", "Temperatur awal", 100.0,
            "Suhu awal yang mengatur peluang menerima langkah yang lebih buruk.", minimum=0.0, step=10.0,
        )
        parameters["cooling_rate"] = parameter_input(
            algorithm, "cooling_rate", "Cooling rate", 0.95,
            "Faktor pendinginan tiap iterasi; nilai dekat 1 mendingin lebih lambat.",
            minimum=0.001, maximum=1.0, step=0.01,
        )
        parameters["max_iterations"] = parameter_input(
            algorithm, "max_iterations", "Jumlah iterasi", 50,
            "Batas maksimum langkah Simulated Annealing.", minimum=0, step=1,
        )
    elif algorithm == "random_restart":
        parameters["max_restarts"] = parameter_input(
            algorithm, "max_restarts", "Maksimum restart", 10,
            "Jumlah restart setelah pencarian awal.", minimum=0, step=1,
        )
        parameters["max_attempts"] = parameter_input(
            algorithm, "max_attempts", "Maksimum percobaan state", 50,
            "Batas percobaan untuk membentuk setiap state awal baru.", minimum=1, step=1,
        )
    elif algorithm == "stochastic":
        parameters["nmax"] = parameter_input(
            algorithm, "nmax", "Maksimum percobaan", 100,
            "Jumlah maksimum proposal langkah acak.", minimum=0, step=1,
        )
    elif algorithm == "sideways":
        parameters["max_sideways"] = parameter_input(
            algorithm, "max_sideways", "Maksimum sideways move", 100,
            "Batas langkah dengan objective yang sama.", minimum=0, step=1,
        )
    elif algorithm == "genetic_algorithm":
        parameters["population_size"] = parameter_input(
            algorithm, "population_size", "Ukuran populasi", 40,
            "Jumlah kandidat state pada setiap generasi.", minimum=2, step=1,
        )
        parameters["iterations"] = parameter_input(
            algorithm, "iterations", "Jumlah generasi", 100,
            "Banyak generasi yang dijalankan.", minimum=0, step=1,
        )

    st.sidebar.subheader("Output")
    if st.sidebar.button("Pilih lokasi dan nama output", use_container_width=True):
        current_output = Path(st.session_state["output_path"])
        try:
            selected = choose_file(
                save=True,
                initial_directory=current_output.parent,
                initial_file=current_output.name,
            )
        except Exception as error:
            st.sidebar.error(f"File manager tidak dapat dibuka: {error}")
        else:
            if selected:
                st.session_state["output_path"] = selected
    output_path = st.session_state["output_path"]
    render_selected_file(output_path)

    st.sidebar.button("Reset pengaturan", on_click=reset_settings, use_container_width=True)
    submitted = st.sidebar.button("Jalankan", type="primary", use_container_width=True)
    return problem_path, output_path, algorithm, int(seed), parameters, submitted


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
) -> None:
    with container:
        st.subheader(title)
        figure, warning = create_state_figure(
            problem,
            state,
            package_filter=package_filter,
            selected_package_id=selected_package_id,
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
    selected_id = st.session_state.get("selected_package_id")

    if mode == "Initial":
        state_rows = initial_rows
    else:
        state_rows = final_rows

    displayed_rows = visible_package_rows(state_rows, package_filter)
    visible_ids = {row["id"] for row in displayed_rows}
    if selected_id not in visible_ids:
        selected_id = None
        st.session_state.pop("selected_package_id", None)

    state_label = "Initial" if mode == "Initial" else "Final"
    state_value = result.get("initial_state") if mode == "Initial" else result.get("final_state")
    state_metrics(st.container(), state_rows, capacity)
    chart_column, detail_column = st.columns([3, 1])
    render_state_chart(
        chart_column, f"{state_label} State", f"{run_key}_{state_label.lower()}_state", problem,
        state_value, package_filter, selected_id,
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
        column_order=("ID", "Posisi (x, y, z)", "Berat", "Value", "Fragile", "ETA", "Status"),
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
        genetic_plot = genetic_objective_figure(history)
        if genetic_plot is None:
            st.warning("Riwayat objective maksimum/rata-rata populasi GA tidak tersedia.")
        else:
            st.plotly_chart(genetic_plot, width="stretch", theme=None, key=f"{run_key}_genetic")


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

    state_tab, graph_tab = st.tabs(["State & Paket", "Grafik"])
    with state_tab:
        render_state_and_packages(result, problem, run_key)
    with graph_tab:
        render_graphs(history, algorithm, run_key)


def main() -> None:
    st.set_page_config(page_title="Eksperimen Pemuatan Truk", layout="wide")
    st.markdown(
        """<style>
        .block-container { max-width: 1600px; padding-top: 1.25rem; padding-bottom: 2rem; }
        .stApp { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
        h1 { font-size: 1.8rem; line-height: 1.25; margin-bottom: 0.35rem; }
        h2 { font-size: 1.3rem; line-height: 1.35; }
        h3 { font-size: 1.05rem; line-height: 1.4; }
        [data-testid="stMetricLabel"] { font-size: 0.85rem; }
        [data-testid="stMetricValue"] { font-size: 1.45rem; }
        .selected-file { display: flex; flex-direction: column; gap: 0.1rem; margin: 0.4rem 0 0.8rem; padding: 0.55rem 0.7rem; border: 1px solid rgba(128, 128, 128, 0.24); border-radius: 0.55rem; overflow: hidden; }
        .selected-file strong { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 0.86rem; font-weight: 600; }
        .selected-file span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; opacity: 0.65; font-size: 0.74rem; }
        .package-inspector-heading { font-size: 1.05rem; font-weight: 600; margin: 0 0 0.55rem; text-align: center; }
        .package-inspector-card { box-sizing: border-box; width: 100%; border: 1px solid #d1d5db; border-radius: 0.5rem; padding: 0.35rem 0.6rem 0.7rem; font-size: 0.82rem; line-height: 1.35; }
        .package-inspector__title { font-size: 0.95rem; font-weight: 700; text-align: center; margin: 0 0 0.35rem; }
        .package-inspector__grid { display: grid; grid-template-columns: minmax(0, 0.8fr) minmax(0, 1.2fr); gap: 0.3rem 0.65rem; width: 100%; max-width: 13rem; margin: 0 auto; }
        .package-inspector__row { display: contents; }
        .package-inspector__row span { opacity: 0.72; text-align: right; }
        .package-inspector__row strong { text-align: left; overflow-wrap: anywhere; }
        .package-inspector__hint { max-width: 13rem; margin: 0 auto; padding: 0.2rem 0 0.4rem; opacity: 0.72; text-align: center; }
        </style>""",
        unsafe_allow_html=True,
    )
    st.title("Eksperimen Pemuatan Truk")

    with st.sidebar:
        st.header("Eksperimen")

    config_text, output_text, algorithm, seed, parameters, submitted = run_controls()

    if submitted:
        clear_last_run()
        try:
            if not config_text.strip():
                raise ValueError("Path file problem wajib diisi.")
            config_path = Path(config_text)
            problem, problem_error = load_problem(config_path)
            if problem is None:
                raise ValueError(problem_error or "Problem config tidak tersedia.")

            output_path = Path(output_text)
            suffix = output_path.suffix.casefold()
            if suffix and suffix != ".json":
                raise ValueError("File output harus menggunakan ekstensi .json.")
            if not suffix:
                output_path = output_path.with_suffix(".json")
            if output_path.exists():
                raise ValueError(f"File output sudah ada: {output_path}")
            run_name = output_path.stem

            LOGGER.info(
                "Memulai eksperimen algorithm=%s config=%s seed=%s parameters=%s",
                algorithm, config_path, seed, parameters,
            )
            with st.spinner(f"Menjalankan {display_algorithm(algorithm)}..."):
                result = run_selected_experiment(
                    algorithm_name=algorithm,
                    config_path=config_path,
                    run_name=run_name,
                    seed=seed,
                    parameters=parameters,
                )
            result["problem_file"] = str(config_path)
            result["output_file"] = str(output_path)
            write_result(output_path, result)
        except (OSError, ValueError, KeyError, TypeError) as error:
            LOGGER.warning("Eksperimen %s gagal: %s", algorithm, error)
            st.error(f"Eksperimen gagal: {error}")
            return
        except Exception:
            LOGGER.exception("Kesalahan tak terduga saat menjalankan %s", algorithm)
            st.error("Terjadi kesalahan tak terduga. Detail tersedia pada log aplikasi.")
            return

        st.session_state["last_run"] = result
        st.session_state["last_problem"] = problem
        LOGGER.info("Eksperimen selesai algorithm=%s output=%s", algorithm, output_path)

    result = st.session_state.get("last_run")
    if result is None:
        st.info("Atur problem, algoritma, parameter, dan output di sidebar, lalu klik **Jalankan**.")
        return

    render_result(result, st.session_state["last_problem"])


if __name__ == "__main__":
    main()
