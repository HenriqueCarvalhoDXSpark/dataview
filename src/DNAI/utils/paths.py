from pathlib import Path

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
