from pathlib import Path
import shutil
import os
import glob

# ------------------------------------------------------------------------------
# Project root resolution
# ------------------------------------------------------------------------------

# This file lives at:
#   src/DNAI/utils/paths.py
# So:
#   parents[0] -> utils
#   parents[1] -> DNAI
#   parents[2] -> src
#   parents[3] -> project root

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ------------------------------------------------------------------------------
# Core directories
# ------------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

SRC_DIR = PROJECT_ROOT / "src"
PACKAGE_DIR = SRC_DIR / "DNAI"


# ------------------------------------------------------------------------------
# structured data subfolders
# ------------------------------------------------------------------------------

DATA_RAW_DIR = DATA_DIR / "raw"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"


# ------------------------------------------------------------------------------
# results subfolders
# ------------------------------------------------------------------------------

RESULTS_TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_MODELS_DIR = RESULTS_DIR / "models"


# ------------------------------------------------------------------------------
# Safety checks (fail fast)
# ------------------------------------------------------------------------------

_REQUIRED_DIRS = [
    DATA_DIR,
    RESULTS_DIR,
]

for _dir in _REQUIRED_DIRS:
    if not _dir.exists():
        raise RuntimeError(
            f"Required directory does not exist: {_dir}\n"
            "Did you run the code from the project root?"
        )


def find_rscript_exe() -> str:
    """
    Return a usable path to Rscript.exe.
    Search order:
      1) PATH (shutil.which)
      2) Environment variable RSCRIPT_EXE
      3) Common Windows install directories under Program Files
    """
    # 1) PATH
    exe = shutil.which("Rscript")
    if exe:
        return exe

    # 2) Explicit environment variable
    env = os.environ.get("RSCRIPT_EXE")
    if env and Path(env).exists():
        return env

    # 3) Common Windows locations (covers most installs)
    candidates = []
    for base in [
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
    ]:
        # R installs typically like: C:\Program Files\R\R-4.4.1\bin\Rscript.exe or bin\x64\Rscript.exe
        candidates += glob.glob(str(Path(base) / "R" / "R-*" / "bin" / "Rscript.exe"))
        candidates += glob.glob(str(Path(base) / "R" / "R-*" / "bin" / "x64" / "Rscript.exe"))

    # Pick the newest version if multiple found
    if candidates:
        candidates = sorted(candidates)  # lexicographic works for R-4.x.y reasonably well
        return candidates[-1]

    raise FileNotFoundError(
        "Could not find Rscript.exe.\n"
        "Fix options:\n"
        "  - Add R to PATH so `Rscript` works in a terminal, OR\n"
        "  - Set an environment variable RSCRIPT_EXE to the full path of Rscript.exe.\n"
        "Example (Windows): setx RSCRIPT_EXE \"C:\\Program Files\\R\\R-4.4.1\\bin\\Rscript.exe\""
    )