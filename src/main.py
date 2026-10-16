import importlib.util
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_FILE = PROJECT_ROOT / "src" / "visualization" / "app.py"


def main() -> int:
    if importlib.util.find_spec("streamlit") is None:
        print("Streamlit belum terpasang. Jalankan: python -m pip install -r requirements.txt", file=sys.stderr)
        return 1
    if not APP_FILE.is_file():
        print(f"File GUI tidak ditemukan: {APP_FILE}", file=sys.stderr)
        return 1

    print("Menjalankan GUI eksperimen...", flush=True)
    try:
        exit_code = subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(APP_FILE)],
            cwd=PROJECT_ROOT,
        )
    except OSError as error:
        print(f"GUI tidak dapat dijalankan: {error}", file=sys.stderr)
        return 1

    print(f"GUI berhenti dengan kode {exit_code}.", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
