from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import streamlit as st

if __package__:
    from .plots import annealing_probability_figure, genetic_objective_figure, objective_figure
    from .state_view import create_state_figure
else:
    from plots import annealing_probability_figure, genetic_objective_figure, objective_figure
    from state_view import create_state_figure


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIRECTORY = PROJECT_ROOT / "output"
PROBLEM_FILE = PROJECT_ROOT / "config" / "problem.json"


@st.cache_data(show_spinner=False)
def load_problem(path: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with Path(path).open(encoding="utf-8") as file:
            problem = json.load(file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, f"Problem config tidak dapat dibaca: {path}"

    if not isinstance(problem, dict):
        return None, "Problem config harus berupa object JSON."
    return problem, None


@st.cache_data(show_spinner=False)
def load_experiments(output_directory: str) -> tuple[list[dict[str, Any]], list[str]]:
    directory = Path(output_directory)
    if not directory.exists():
        return [], [f"Folder output tidak ditemukan: {directory}"]

    result_files = sorted(directory.rglob("*.json"))
    if not result_files:
        return [], [f"Belum ada file JSON hasil eksperimen di: {directory}"]

    experiments: list[dict[str, Any]] = []
    warnings: list[str] = []

    for path in result_files:
        try:
            with path.open(encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            warnings.append(f"Melewati `{path.relative_to(directory)}` karena JSON tidak valid atau tidak dapat dibaca.")
            continue

        if not isinstance(data, dict):
            warnings.append(f"Melewati `{path.relative_to(directory)}` karena root JSON harus berupa object.")
            continue

        data["_source_file"] = str(path.relative_to(directory))
        data["_algorithm_label"] = infer_algorithm(data, path, directory)
        experiments.append(data)

    return experiments, warnings


def infer_algorithm(data: Mapping[str, Any], path: Path, output_directory: Path) -> str:
    algorithm = data.get("algorithm")
    if isinstance(algorithm, str) and algorithm.strip():
        return algorithm.strip()

    relative_parent = path.relative_to(output_directory).parent
    if str(relative_parent) != ".":
        return relative_parent.name.replace("_", " ")

    return "Tidak diketahui"


def run_label(data: Mapping[str, Any]) -> str:
    configured_name = data.get("run_name", data.get("run"))
    source = str(data.get("_source_file", "hasil.json"))

    if configured_name is None:
        return Path(source).stem.replace("_", " ").title()

    return str(configured_name).replace("_", " ").title()


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


def render_state(
    title: str,
    key: str,
    problem: Mapping[str, Any],
    state: Any,
    dark: bool,
    show_outside: bool,
) -> None:
    st.subheader(title)
    figure, warning = create_state_figure(problem, state, dark=dark, show_outside=show_outside)

    if figure is None:
        st.warning(warning or "State tidak dapat divisualisasikan.")
        return

    st.plotly_chart(figure, width="stretch", key=key)
    if warning:
        st.warning(warning)


def main() -> None:
    st.set_page_config(page_title="Visualisasi Pemuatan Truk", layout="wide")
    st.markdown("<style>.block-container { max-width: 1600px; padding-top: 2rem; }</style>", unsafe_allow_html=True)
    st.title("Visualisasi Eksperimen Pemuatan Truk")
    st.caption("Data dibaca dari config/problem.json dan output/*.json.")

    with st.sidebar:
        st.header("Kontrol eksperimen")
        st.caption("Pilih hasil yang ingin diperiksa.")

        if st.button("Muat ulang hasil"):
            load_experiments.clear()
            load_problem.clear()

    problem, problem_error = load_problem(str(PROBLEM_FILE))
    if problem is None:
        st.error(problem_error or "Problem config tidak tersedia.")
        return

    experiments, loading_warnings = load_experiments(str(OUTPUT_DIRECTORY))

    for warning in loading_warnings:
        st.warning(warning)

    if not experiments:
        st.info(
            "Belum ada hasil yang dapat divisualisasikan. "
            "Simpan hasil eksperimen sebagai file JSON di folder `output/`."
        )
        return

    algorithms = sorted({str(experiment["_algorithm_label"]) for experiment in experiments})
    algorithm_labels = {algorithm: display_algorithm(algorithm) for algorithm in algorithms}
    selected_algorithm = st.sidebar.selectbox(
        "Algoritma",
        algorithms,
        format_func=algorithm_labels.get,
    )

    selected_runs = [
        experiment
        for experiment in experiments
        if str(experiment["_algorithm_label"]) == selected_algorithm
    ]
    labels = [run_label(experiment) for experiment in selected_runs]
    selected_label = st.sidebar.selectbox("Run", labels)
    selected = selected_runs[labels.index(selected_label)]

    with st.sidebar:
        st.divider()
        st.subheader("Tampilan")
        appearance = st.selectbox("Tema grafik", ["Ikuti Streamlit", "Light", "Dark"])
        streamlit_theme = st.get_option("theme.base")
        dark = appearance == "Dark" or (appearance == "Ikuti Streamlit" and streamlit_theme == "dark")
        show_outside = st.checkbox("Tampilkan paket di luar truk", value=False)
        st.caption("Mode fokus menyesuaikan viewport ke container. Aktifkan opsi di atas untuk melihat seluruh state.")

    history = history_from(selected)

    chart_prefix = f"{selected_algorithm}_{selected.get('_source_file', selected_label)}"
    st.header(display_algorithm(selected_algorithm))
    st.caption(selected_label)

    final_objective = selected.get("final_objective")
    duration = selected.get("duration_seconds", selected.get("duration"))
    iterations = selected.get("iterations")
    if iterations is None and isinstance(history.get("objective"), Sequence):
        iterations = len(history["objective"])

    summary_columns = st.columns(3)
    display_metric(summary_columns[0], "Final Objective", final_objective)
    display_metric(summary_columns[1], "Duration", duration, " s")
    display_metric(summary_columns[2], "Iterations", iterations)

    if not history:
        st.warning("Field `history` tidak tersedia atau bukan object. Grafik tidak dapat dibuat.")
    else:
        objective_plot = objective_figure(history, dark=dark)
        if objective_plot is None:
            st.warning("Data `history.objective` tidak tersedia atau kosong.")
        else:
            st.plotly_chart(objective_plot, width="stretch", key=f"{chart_prefix}_objective")

    st.divider()
    render_state(
        "Initial State",
        f"{chart_prefix}_initial_state",
        problem,
        selected.get("initial_state"),
        dark,
        show_outside,
    )
    render_state(
        "Final State",
        f"{chart_prefix}_final_state",
        problem,
        selected.get("final_state"),
        dark,
        show_outside,
    )

    if is_simulated_annealing(selected_algorithm):
        st.header("Simulated Annealing")

        probability_plot = annealing_probability_figure(history, dark=dark)
        if probability_plot is None:
            st.warning(
                "Data nilai mentah `e^(ΔE/T)` tidak tersedia. "
                "Dashboard membutuhkan `history.exp_delta_e_over_t`."
            )
        else:
            st.plotly_chart(probability_plot, width="stretch", key=f"{chart_prefix}_exp_delta_e_over_t")

        stuck_frequency = history.get("stuck_frequency", selected.get("stuck_frequency"))
        display_metric(st.columns(1)[0], "Frekuensi Stuck di Local Optima", stuck_frequency)

    if is_genetic_algorithm(selected_algorithm):
        st.header("Genetic Algorithm")

        ga_columns = st.columns(2)
        display_metric(ga_columns[0], "Population Size", selected.get("population_size"))
        display_metric(ga_columns[1], "Jumlah Iterasi", selected.get("iterations"))

        genetic_plot = genetic_objective_figure(history, dark=dark)
        if genetic_plot is None:
            st.warning(
                "Data GA `history.max_objective` dan/atau "
                "`history.average_objective` tidak tersedia."
            )
        else:
            st.plotly_chart(genetic_plot, width="stretch", key=f"{chart_prefix}_genetic_objective")


if __name__ == "__main__":
    main()
