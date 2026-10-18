from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from core.evaluation import OBJECTIVE_LABELS
from experiments.runner import ALGORITHM_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRECTORY = PROJECT_ROOT / "config"
OUTPUT_DIRECTORY = PROJECT_ROOT / "output"
DEFAULT_PROBLEM_PATH = CONFIG_DIRECTORY / "problem.json"
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
