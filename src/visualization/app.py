from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import streamlit as st

SRC_DIRECTORY = Path(__file__).resolve().parents[1]
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from experiments.output import write_result
from experiments.runner import run_selected_experiment
from visualization.controls import (
    clear_last_run,
    display_algorithm,
    run_controls,
)
from visualization.results import render_result


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


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
