from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from experiments.output import unique_result_path, write_result
from experiments.runner import ALGORITHM_NAMES, run_selected_experiment
from visualization.plots import annealing_probability_figure, genetic_objective_figure, objective_figure
from visualization.state_view import create_state_figure


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
OUTPUT_DIRECTORY = PROJECT_ROOT / "output"
PROBLEM_FILE = PROJECT_ROOT / "config" / "problem.json"


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


def display_algorithm(algorithm: str) -> str:
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


def display_metric(column: Any, label: str, value: Any, suffix: str = "") -> None:
    if value is None:
        column.metric(label, "Tidak tersedia")
    else:
        column.metric(label, f"{value}{suffix}")


def render_state(title: str, key: str, problem: Mapping[str, Any], state: Any, show_outside: bool) -> None:
    st.subheader(title)
    figure, warning = create_state_figure(problem, state, show_outside=show_outside)
    if figure is None:
        st.warning(warning or "State tidak dapat divisualisasikan.")
        return

    st.plotly_chart(figure, width="stretch", theme=None, key=key)
    if warning:
        st.warning(warning)


def clear_last_run() -> None:
    st.session_state.pop("last_run", None)


def run_controls() -> tuple[str | None, int | None, dict[str, Any], bool]:
    algorithm = st.sidebar.selectbox(
        "Algoritma",
        ALGORITHM_NAMES,
        index=None,
        placeholder="Pilih algoritma",
        format_func=display_algorithm,
        key="run_algorithm",
        on_change=clear_last_run,
    )
    if algorithm is None:
        return None, None, {}, False

    parameters: dict[str, Any] = {}
    with st.sidebar.form("run_experiment", enter_to_submit=False):
        seed = st.number_input("Seed", min_value=0, value=None, step=1, placeholder="Masukkan seed")
        if algorithm == "simulated_annealing":
            parameters["initial_temperature"] = st.number_input(
                "Temperatur awal", min_value=0.0, value=None, step=10.0, placeholder="Masukkan temperatur awal"
            )
            parameters["cooling_rate"] = st.number_input(
                "Cooling rate", min_value=0.001, max_value=1.0, value=None, step=0.01,
                placeholder="Masukkan cooling rate",
            )
            parameters["max_iterations"] = st.number_input(
                "Jumlah iterasi", min_value=0, value=None, step=1, placeholder="Masukkan jumlah iterasi"
            )
        elif algorithm == "random_restart":
            parameters["max_restarts"] = st.number_input(
                "Maksimum restart", min_value=0, value=None, step=1, placeholder="Masukkan maksimum restart"
            )
            parameters["max_attempts"] = st.number_input(
                "Maksimum percobaan state", min_value=1, value=None, step=1,
                placeholder="Masukkan maksimum percobaan state",
            )
        elif algorithm == "stochastic":
            parameters["nmax"] = st.number_input(
                "Maksimum percobaan", min_value=0, value=None, step=1, placeholder="Masukkan maksimum percobaan"
            )
        elif algorithm == "sideways":
            parameters["max_sideways"] = st.number_input(
                "Maksimum sideways move", min_value=0, value=None, step=1,
                placeholder="Masukkan maksimum sideways move",
            )
        elif algorithm == "genetic_algorithm":
            parameters["population_size"] = st.number_input(
                "Ukuran populasi", min_value=2, value=None, step=1, placeholder="Masukkan ukuran populasi"
            )
            parameters["iterations"] = st.number_input(
                "Jumlah iterasi", min_value=0, value=None, step=1, placeholder="Masukkan jumlah iterasi"
            )
        submitted = st.form_submit_button("Jalankan", type="primary", use_container_width=True)

    return algorithm, seed, parameters, submitted


def render_result(result: Mapping[str, Any], problem: Mapping[str, Any], show_outside: bool) -> None:
    algorithm = str(result.get("algorithm", "Tidak diketahui"))
    history = history_from(result)
    run_name = str(result.get("run_name", "Run terbaru"))
    chart_prefix = run_name

    st.header(display_algorithm(algorithm))
    st.caption(f"Run: {run_name}")

    final_objective = result.get("final_objective")
    duration = result.get("duration_seconds", result.get("duration"))
    iterations = result.get("iterations")
    if iterations is None and isinstance(history.get("objective"), Sequence):
        iterations = len(history["objective"])

    summary_columns = st.columns(3)
    display_metric(summary_columns[0], "Final Objective", final_objective)
    display_metric(summary_columns[1], "Durasi", duration, " s")
    display_metric(summary_columns[2], "Jumlah Iterasi", iterations)

    if not history:
        st.warning("Field `history` tidak tersedia atau bukan object. Grafik tidak dapat dibuat.")
    elif not is_genetic_algorithm(algorithm):
        objective_plot = objective_figure(history)
        if objective_plot is None:
            st.warning("Data `history.objective` tidak tersedia atau kosong.")
        else:
            st.plotly_chart(objective_plot, width="stretch", theme=None, key=f"{chart_prefix}_objective")

    st.divider()
    render_state("Initial State", f"{chart_prefix}_initial_state", problem, result.get("initial_state"), show_outside)
    render_state("Final State", f"{chart_prefix}_final_state", problem, result.get("final_state"), show_outside)

    if is_simulated_annealing(algorithm):
        st.header("Simulated Annealing")
        probability_plot = annealing_probability_figure(history)
        if probability_plot is None:
            st.warning("Data `history.exp_delta_e_over_t` tidak tersedia untuk grafik probabilitas penerimaan.")
        else:
            st.plotly_chart(
                probability_plot, width="stretch", theme=None, key=f"{chart_prefix}_exp_delta_e_over_t"
            )
        display_metric(st.columns(1)[0], "Frekuensi Stuck di Local Optima", history.get("stuck_frequency"))

    if is_genetic_algorithm(algorithm):
        st.header("Genetic Algorithm")
        result_parameters = result.get("parameters")
        population_size = result.get("population_size")
        if population_size is None and isinstance(result_parameters, Mapping):
            population_size = result_parameters.get("population_size")
        display_metric(st.columns(1)[0], "Ukuran Populasi", population_size)
        genetic_plot = genetic_objective_figure(history)
        if genetic_plot is None:
            st.warning("Riwayat objective maksimum/rata-rata populasi GA tidak tersedia.")
        else:
            st.plotly_chart(genetic_plot, width="stretch", theme=None, key=f"{chart_prefix}_genetic_objective")


def main() -> None:
    st.set_page_config(page_title="Eksperimen Pemuatan Truk", layout="wide")
    st.markdown("<style>.block-container { max-width: 1600px; padding-top: 2rem; }</style>", unsafe_allow_html=True)
    st.title("Eksperimen Pemuatan Truk")
    st.caption("Pilih algoritma, isi parameter, lalu jalankan untuk melihat visualisasi hasilnya.")

    with st.sidebar:
        st.header("Eksperimen")
        show_outside = st.checkbox("Tampilkan paket di luar truk", value=False)
        st.caption("Mode fokus menyesuaikan viewport ke container.")

    problem, problem_error = load_problem(PROBLEM_FILE)
    if problem is None:
        st.error(problem_error or "Problem config tidak tersedia.")
        return

    algorithm, seed, parameters, submitted = run_controls()
    if algorithm is None:
        st.info("Pilih algoritma untuk mengisi parameter dan memulai eksperimen.")
        return

    if submitted:
        if seed is None or any(value is None for value in parameters.values()):
            st.error("Isi seed dan semua parameter sebelum menjalankan algoritma.")
            return

        clear_last_run()
        output_path = unique_result_path(OUTPUT_DIRECTORY, algorithm)
        LOGGER.info("Memulai eksperimen algorithm=%s seed=%s parameters=%s", algorithm, seed, parameters)
        try:
            with st.spinner(f"Menjalankan {display_algorithm(algorithm)}..."):
                result = run_selected_experiment(
                    algorithm_name=algorithm,
                    config_path=PROBLEM_FILE,
                    run_name=output_path.stem,
                    seed=int(seed),
                    parameters=parameters,
                )
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
        LOGGER.info("Eksperimen selesai algorithm=%s output=%s", algorithm, output_path)

    result = st.session_state.get("last_run")
    if result is None:
        st.info("Atur parameter lalu tekan **Jalankan** untuk melihat hasil.")
        return

    render_result(result, problem, show_outside)


if __name__ == "__main__":
    main()
