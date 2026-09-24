import os
from pathlib import Path


# Locate source data and workflow outputs.
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent


# Load .env without overriding variables already set by the debugger or shell.
def _load_dotenv(env_path):
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key and value and key not in os.environ:
            os.environ[key] = value


def _env_path(name, default):
    value = os.environ.get(name, "").strip()
    return Path(value) if value else default


_load_dotenv(CODE_DIR / ".env")

OSM_FILE = _env_path(
    "OSM_FILE",
    PROJECT_ROOT / "Initial Input Files" / "OSM_Short_Level_CSV"
    / "Complete_OSM_Short_Level_Link_List.csv",
)
TXDOT_FILE = _env_path(
    "TXDOT_FILE",
    PROJECT_ROOT / "Initial Input Files" / "TxDOT"
    / "link_list_with_speed_NOL.csv",
)
CV_FILE = _env_path(
    "CV_FILE",
    PROJECT_ROOT / "Output" / "Final" / "9_20_2025"
    / "9_20_2025_hr=19.csv",
)

OUTPUT_DIR = _env_path("OUTPUT_DIR", CODE_DIR / "Results" / "Network_MOE")
FIGURE_DIR = OUTPUT_DIR / "Corridor_Figures"

TXDOT_REFERENCE_FILE = OUTPUT_DIR / "txdot_speed_reference.parquet"
OBSERVED_LINKS_FILE = OUTPUT_DIR / "observed_link_keys.parquet"
PARENT_FILE = OUTPUT_DIR / "parent_links.parquet"
SEGMENT_FILE = OUTPUT_DIR / "moe_segments_200ft.parquet"
MOE_FILE = OUTPUT_DIR / "segment_moe_5min.parquet"
CORRIDOR_FILE = OUTPUT_DIR / "corridors.parquet"
CORRIDOR_SEGMENT_FILE = OUTPUT_DIR / "corridor_segments.parquet"
JOURNEY_PARTITION_DIR = OUTPUT_DIR / "CV_Journey_Partitions"
MOE_PARTIAL_DIR = OUTPUT_DIR / "MOE_Parallel_Parts"
TRANSITION_PARTIAL_DIR = OUTPUT_DIR / "Transition_Parallel_Parts"

# Use one projected coordinate system and one unit conversion throughout.
SOURCE_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:3083"
M_TO_FT = 1 / 0.3048

# Define MOE and matching settings.
SEGMENT_LENGTH_FT = 200.0
TIME_BIN_MINUTES = 15
SLOW_SPEED_THRESHOLD_MPH = 5.0
MAX_WAYPOINT_GAP_SECONDS = 120
MAX_TXDOT_MATCH_DISTANCE_FT = 100.0
MAX_TXDOT_HEADING_DIFFERENCE_DEG = 35.0

# Balance large-file throughput and memory use.
CSV_CHUNK_SIZE = 500_000
PARQUET_COMPRESSION = "snappy"
REPORT_INTERVAL_SECONDS = 5

# Read a positive integer from .env. Blank means use the computed default.
def _env_int(name):
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be at least 1")
    return number


# Leave PARALLEL_WORKERS blank to use every core except two.
# Journey partitions must be at least the worker count, or those cores sit idle.
_CPU_COUNT = os.cpu_count() or 1
PARALLEL_WORKERS = _env_int("PARALLEL_WORKERS") or max(1, _CPU_COUNT - 2)
JOURNEY_PARTITIONS = max(
    _env_int("JOURNEY_PARTITIONS") or 32,
    PARALLEL_WORKERS,
)
FINAL_SEGMENT_BATCH_SIZE = 100_000

# Configure automatic corridor selection for visualization only.
CORRIDOR_START_LINKS = []
NUMBER_OF_CORRIDORS = 5
MAX_CORRIDOR_PARENT_LINKS = 40
MAX_CORRIDOR_LENGTH_FT = 26_400.0


# Check source files and create output folders.
def check_inputs():
    for path in (OSM_FILE, TXDOT_FILE, CV_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
