import argparse
import csv
import io
import math
import re
import zipfile
from collections import defaultdict
from contextlib import ExitStack
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Literal, Optional, TextIO

from datetime import datetime, timezone

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # allow `--help` without optional runtime deps
    mdates = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]

CSV_FILE_PATH = "/mnt/c/Users/DSavkovic/Downloads/SC/20260103105034.csv"

SENSORS = {
    "LEFT": "DF:25:4D:3D:35:6A",
    "RIGHT": "CC:4F:71:B6:CE:8F"
}

# `time` is the axis we use for all plots. Support multiple formats so microseconds remain optional.
TIME_COLUMN = "time"
TIME_FORMATS = (
    "%m/%d/%Y %H:%M:%S.%f",
    "%m/%d/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
)

PACKAGE_NAME = "sc"

PlotBackend = Literal["mp", "pl"]


def _version_from_pyproject() -> Optional[str]:
    try:
        import tomllib
    except ModuleNotFoundError:
        return None
    try:
        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with pyproject_path.open("rb") as fp:
            data = tomllib.load(fp)
        return data["project"]["version"]
    except (FileNotFoundError, KeyError):
        return None


try:
    PACKAGE_VERSION = version(PACKAGE_NAME)
except PackageNotFoundError:
    PACKAGE_VERSION = _version_from_pyproject() or "0.0.0"


def is_header_row(row: Dict[str, str]) -> bool:
    time_value = row.get(TIME_COLUMN, "")
    return isinstance(time_value, str) and time_value.strip().lower() == TIME_COLUMN.lower()


COLUMNS = {
    "ang": ("AngleX(°)", "AngleY(°)", "AngleZ(°)"),  # orientation angles
    "acc": ("AccX(g)", "AccY(g)", "AccZ(g)"),  # accelerometer
    "as": ("AsX(°/s)", "AsY(°/s)", "AsZ(°/s)"),  # gyroscope (angular speed)
    "h": ("HX(uT)", "HY(uT)", "HZ(uT)"),  # magnetometer (magnetic field)
    "q": ("Q0()", "Q1()", "Q2()", "Q3()"),  # quaternion components (sensor orientation)
    "edge": ("Edge(°)",),  # edging/roll angle around the boot forward axis (derived)
    "edge2": ("Edge(°)", "Pitch(°)", "Yaw(°)"),  # edging plus pitch/yaw from same fusion
    "alt": ("Altitude(m)",),  # GNSS altitude (from *gnss.ride inside a .zip)
    "tilt": ("Pitch(°)", "Roll(°)", "Yaw(°)"),  # pitch/roll from accelerometer + yaw from magnetometer
    "sf0": ("Roll(°)",),  # IMU-only complementary filter (acc + gyro) for roll
    "mad": ("Pitch(°)", "Roll(°)", "Yaw(°)"),  # Madgwick filter
    "mah": ("Pitch(°)", "Roll(°)", "Yaw(°)"),  # Mahony filter
    "ekf": ("Pitch(°)", "Roll(°)", "Yaw(°)"),  # Extended Kalman Filter
}

ANGLE_COLORS = {
    "AngleX(°)": "blue",
    "AngleY(°)": "green",
    "AngleZ(°)": "magenta",
}


def has_quaternion_values(rows_by_device: Dict[str, List[Dict[str, str]]]) -> bool:
    """Return True if the dataset contains any numeric quaternion (Q0..Q3) samples.

    Some input formats do not include quaternion columns at all. In those cases,
    requesting the `q` group should not crash the CLI.
    """

    for rows in rows_by_device.values():
        for row in rows:
            if extract_quaternion_wxyz(row) is not None:
                return True
    return False


def has_quaternion_columns(rows: List[Dict[str, str]]) -> bool:
    """Return True if the per-row dicts expose all Q0..Q3 columns.

    Some file formats may use different quaternion column naming conventions
    (e.g. q0..q3, Q0..Q3, qw/qx/qy/qz). Treat them as present when at least one
    row yields a full quaternion.
    """

    for row in rows:
        if extract_quaternion_wxyz(row) is not None:
            return True
    return False


def extract_quaternion_wxyz(row: Dict[str, str]) -> Optional[tuple[float, float, float, float]]:
    """Extract a quaternion from a row dict.

    Supports case-insensitive keys and multiple naming conventions:
      - q0/q1/q2/q3 (or Q0/Q1/Q2/Q3) interpreted as (w, x, y, z)
      - qw/qx/qy/qz interpreted as (w, x, y, z)
      - qx/qy/qz/qw interpreted as (x, y, z, w) and reordered

    The plotting code uses canonical column names "Q0()".."Q3()"; this helper
    allows inputs like `.ride` exports to feed those plots without requiring the
    file format to match exactly.
    """

    normalized: Dict[str, str] = {key.strip().lower(): key for key in row}

    def lookup(*candidates: str) -> Optional[str]:
        for candidate in candidates:
            key = candidate.strip().lower()
            original = normalized.get(key)
            if original is None:
                continue
            value = row.get(original)
            if value is None:
                continue
            stripped = value.strip()
            if not stripped or stripped.lower() == "null":
                return None
            return stripped
        return None

    def parse_component(value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    # Prefer q0..q3 conventions.
    w = parse_component(lookup("q0", "q0()"))
    x = parse_component(lookup("q1", "q1()"))
    y = parse_component(lookup("q2", "q2()"))
    z = parse_component(lookup("q3", "q3()"))
    if w is not None and x is not None and y is not None and z is not None:
        return (w, x, y, z)

    # Next try qw/qx/qy/qz.
    w = parse_component(lookup("qw", "q_w"))
    x = parse_component(lookup("qx", "q_x"))
    y = parse_component(lookup("qy", "q_y"))
    z = parse_component(lookup("qz", "q_z"))
    if w is not None and x is not None and y is not None and z is not None:
        return (w, x, y, z)

    # Finally, try qx/qy/qz/qw ordering.
    x = parse_component(lookup("qx", "q_x"))
    y = parse_component(lookup("qy", "q_y"))
    z = parse_component(lookup("qz", "q_z"))
    w = parse_component(lookup("qw", "q_w"))
    if w is not None and x is not None and y is not None and z is not None:
        return (w, x, y, z)

    return None


def quaternion_to_edge_deg(q: tuple[float, float, float, float]) -> float:
    """Compute the boot edging angle (degrees) from an orientation quaternion.

    This follows the approach described in `quart.md`: rotate the world-down
    gravity vector into the body frame and compute roll about the boot forward
    axis (assumed to be body +X).

    Assumptions:
      - Quaternion order is (w, x, y, z)
      - The quaternion represents body->world rotation
      - Body axes are: +X forward (along ski), +Y left, +Z up

    Returns:
        Edging/roll angle in degrees, where positive means rolling toward +Y.
    """

    qw, qx, qy, qz = q

    # Compute g_body = q*conj applied to world-down vector (0, 0, -1).
    # We implement: g_body = q_conj * (0, g_world) * q, returning vector part.
    # This matches the pseudocode from `quart.md`.
    cw, cx, cy, cz = (qw, -qx, -qy, -qz)

    # vq = (0, 0, 0, -1)
    vw, vx, vy, vz = (0.0, 0.0, 0.0, -1.0)

    # First multiply: a = conj(q) * vq
    aw = cw * vw - cx * vx - cy * vy - cz * vz
    ax = cw * vx + cx * vw + cy * vz - cz * vy
    ay = cw * vy - cx * vz + cy * vw + cz * vx
    az = cw * vz + cx * vy - cy * vx + cz * vw

    # Then multiply: b = a * q
    by = aw * qy - ax * qz + ay * qw + az * qx
    bz = aw * qz + ax * qy - ay * qx + az * qw

    # Gravity vector in body frame.
    gy, gz = by, bz
    return math.degrees(math.atan2(gy, gz))


def quaternion_to_euler_deg(q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Return roll, pitch, yaw in degrees for quaternion ordered as (w, x, y, z).

    The conversion follows the intrinsic Tait-Bryan ZYX convention used by
    common AHRS filters and matches the outputs of Madgwick/Mahony/EKF above.
    """

    w, x, y, z = q

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.degrees(math.copysign(math.pi / 2.0, sinp))
    else:
        pitch = math.degrees(math.asin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))

    return roll, pitch, yaw


def parse_timestamp(value: str, row_index: int) -> datetime:
    last_error: Exception | None = None
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError as exc:
            last_error = exc
    raise ValueError(
        f"Cannot parse timestamp {value!r} at row {row_index}: expected formats {TIME_FORMATS}"
    ) from last_error


def unwrap_degrees(values: List[float]) -> List[float]:
    if not values:
        return []

    unwrapped = [values[0]]
    prev_adjusted = values[0]
    for value in values[1:]:
        adjusted = value
        delta = adjusted - prev_adjusted
        if delta > 180:
            adjusted -= 360
        elif delta < -180:
            adjusted += 360
        unwrapped.append(unwrapped[-1] + (adjusted - prev_adjusted))
        prev_adjusted = adjusted
    return unwrapped


def parse_csv_file(csv_path: Path | str) -> Dict[str, List[Dict[str, str]]]:
    """Return rows grouped by DeviceName so callers can access data per device, sorted by time."""

    return _parse_tabular_file(csv_path, delimiter=",")


def parse_tsv_file(tsv_path: Path | str) -> Dict[str, List[Dict[str, str]]]:
    """Return rows grouped by DeviceName so callers can access data per device, sorted by time."""

    return _parse_tabular_file(tsv_path, delimiter="\t")


def parse_ride_pair(ride_path: Path | str) -> Dict[str, List[Dict[str, str]]]:
    """Parse a .ride file and its LEFT/RIGHT counterpart.

    The .ride export format is a CSV-like file that does not include a `DeviceName`
    column. Instead, the device side is encoded in the filename (e.g.
    `*_left.ride` / `*_right.ride`). When the user provides one side, this loader
    automatically locates and loads the other side from the same directory.

    Returns:
        A mapping of device label ("LEFT"/"RIGHT") to a list of normalized rows.
    """

    input_path = Path(ride_path).expanduser()
    left_path, right_path = _resolve_ride_pair_paths(input_path)
    return {
        "LEFT": parse_ride_file(left_path, device_label="LEFT"),
        "RIGHT": parse_ride_file(right_path, device_label="RIGHT"),
    }


def _resolve_ride_pair_paths(path: Path) -> tuple[Path, Path]:
    """Return (left_path, right_path) for a .ride input path."""

    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() != ".ride":
        raise ValueError(f"Expected a .ride file, got: {path}")

    match = re.match(
        r"^(?P<prefix>.*?)(?P<sep>[_-])(?P<side>left|right)(?P<suffix>\.ride)$",
        path.name,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(
            "Cannot infer sensor side from .ride filename. Expected suffix like '_left.ride' or '_right.ride'. "
            f"Got: {path.name}"
        )

    prefix = match.group("prefix")
    sep = match.group("sep")
    side = match.group("side").lower()
    suffix = match.group("suffix")

    if side == "left":
        left_path = path
        right_path = path.with_name(f"{prefix}{sep}right{suffix}")
    else:
        right_path = path
        left_path = path.with_name(f"{prefix}{sep}left{suffix}")

    if not left_path.is_file():
        raise FileNotFoundError(f"Matching LEFT .ride file not found: {left_path}")
    if not right_path.is_file():
        raise FileNotFoundError(f"Matching RIGHT .ride file not found: {right_path}")
    return left_path, right_path


def parse_ride_file(ride_path: Path | str, *, device_label: str) -> List[Dict[str, str]]:
    """Parse a single .ride file into normalized rows.

    The .ride columns are equivalent to the TSV export, but use short names:
      - AccX/AccY/AccZ -> AccX(g)/AccY(g)/AccZ(g)
      - AsX/AsY/AsZ    -> AsX(°/s)/AsY(°/s)/AsZ(°/s)
      - AngX/AngY/AngZ -> AngleX(°)/AngleY(°)/AngleZ(°)
      - HX/HY/HZ       -> HX(uT)/HY(uT)/HZ(uT)

    The time axis is derived from `timestamp_ms` (Unix epoch milliseconds) and
    written into the shared `time` column used by the plotting pipeline.
    """

    path = Path(ride_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as fp:
        return parse_ride_stream(fp, device_label=device_label)


def parse_ride_stream(fp: TextIO, *, device_label: str) -> List[Dict[str, str]]:
    """Parse a file-like stream containing `.ride` CSV content.

    This is used both for regular `.ride` files on disk and `.ride` members
    extracted from a `.zip` archive.
    """

    normalized_rows: List[Dict[str, str]] = []
    reader = csv.DictReader(fp, delimiter=",")
    for idx, row in enumerate(reader, start=1):
        normalized_keys = {key.strip().lower(): key for key in row}

        def ride_value(*candidates: str) -> str:
            for candidate in candidates:
                original = normalized_keys.get(candidate.strip().lower())
                if original is None:
                    continue
                return row.get(original, "")
            return ""

        ts_ms = row.get("timestamp_ms")
        if ts_ms is None:
            raise KeyError("Ride file is missing the required 'timestamp_ms' column")
        try:
            ts = datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)
        except ValueError as exc:
            raise ValueError(f"Cannot parse timestamp_ms {ts_ms!r} at row {idx}") from exc

        normalized_row = {
                "DeviceName": device_label,
                TIME_COLUMN: ts.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "AccX(g)": row.get("AccX", ""),
                "AccY(g)": row.get("AccY", ""),
                "AccZ(g)": row.get("AccZ", ""),
                "AsX(°/s)": row.get("AsX", ""),
                "AsY(°/s)": row.get("AsY", ""),
                "AsZ(°/s)": row.get("AsZ", ""),
                "AngleX(°)": row.get("AngX", ""),
                "AngleY(°)": row.get("AngY", ""),
                "AngleZ(°)": row.get("AngZ", ""),
                "HX(uT)": row.get("HX", ""),
                "HY(uT)": row.get("HY", ""),
                "HZ(uT)": row.get("HZ", ""),
            }

        q0 = ride_value("q0", "q0()")
        q1 = ride_value("q1", "q1()")
        q2 = ride_value("q2", "q2()")
        q3 = ride_value("q3", "q3()")
        if any((q0, q1, q2, q3)):
            normalized_row["Q0()"] = q0
            normalized_row["Q1()"] = q1
            normalized_row["Q2()"] = q2
            normalized_row["Q3()"] = q3

        normalized_rows.append(normalized_row)

    timestamped: List[tuple[datetime, Dict[str, str]]] = []
    for idx, row in enumerate(normalized_rows, start=1):
        time_value = row.get(TIME_COLUMN)
        if time_value is None:
            raise KeyError(
                f"Ride rows are missing the {TIME_COLUMN!r} column required for sorting"
            )
        timestamped.append((parse_timestamp(time_value, idx), row))
    timestamped.sort(key=lambda pair: pair[0])
    return [row for _, row in timestamped]


def parse_ride_zip(zip_path: Path | str) -> Dict[str, List[Dict[str, str]]]:
    """Parse a `.zip` archive that contains a LEFT/RIGHT `.ride` pair.

    The archive is expected to include two `.ride` files whose basenames match
    the usual naming convention, such as `*_left.ride` and `*_right.ride`.
    """

    path = Path(zip_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    with zipfile.ZipFile(path) as zf:
        left_member, right_member = _resolve_ride_pair_members(zf)
        with ExitStack() as stack:
            left_raw = stack.enter_context(zf.open(left_member))
            right_raw = stack.enter_context(zf.open(right_member))
            left_text = stack.enter_context(
                io.TextIOWrapper(left_raw, encoding="utf-8-sig", newline="")
            )
            right_text = stack.enter_context(
                io.TextIOWrapper(right_raw, encoding="utf-8-sig", newline="")
            )
            return {
                "LEFT": parse_ride_stream(left_text, device_label="LEFT"),
                "RIGHT": parse_ride_stream(right_text, device_label="RIGHT"),
            }


def parse_gnss_ride_stream(fp: TextIO) -> tuple[List[datetime], List[float]]:
    """Parse a GNSS `.ride` CSV stream and return timestamps + altitude.

    Expected columns:
      - timestamp_ms: Unix epoch milliseconds
      - altitude: altitude in meters
    """

    timestamps: List[datetime] = []
    altitudes: List[float] = []

    reader = csv.DictReader(fp, delimiter=",")
    for idx, row in enumerate(reader, start=1):
        ts_ms_raw = row.get("timestamp_ms")
        alt_raw = row.get("altitude")
        if ts_ms_raw is None or alt_raw is None:
            continue

        ts_ms = ts_ms_raw.strip()
        altitude_str = alt_raw.strip()
        if not ts_ms or not altitude_str:
            continue
        if ts_ms.lower() == "null" or altitude_str.lower() == "null":
            continue

        try:
            timestamp = datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)
        except ValueError as exc:
            raise ValueError(
                f"Cannot parse GNSS timestamp_ms {ts_ms_raw!r} at row {idx}"
            ) from exc

        try:
            altitude_m = float(altitude_str)
        except ValueError as exc:
            raise ValueError(
                f"Cannot parse GNSS altitude {alt_raw!r} at row {idx}"
            ) from exc

        timestamps.append(timestamp)
        altitudes.append(altitude_m)

    if timestamps:
        combined = sorted(zip(timestamps, altitudes), key=lambda pair: pair[0])
        timestamps = [ts for ts, _ in combined]
        altitudes = [alt for _, alt in combined]
    return timestamps, altitudes


def parse_gnss_ride_zip(zip_path: Path | str) -> tuple[List[datetime], List[float]]:
    """Parse a `.zip` archive member matching `*gnss.ride` and return altitude."""

    path = Path(zip_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    with zipfile.ZipFile(path) as zf:
        member = _resolve_gnss_member(zf)
        with zf.open(member) as raw:
            with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
                return parse_gnss_ride_stream(text)


def _resolve_gnss_member(zf: zipfile.ZipFile) -> str:
    """Return the GNSS ride member name (basename ends with 'gnss.ride')."""

    candidates = [
        name
        for name in zf.namelist()
        if not name.endswith("/")
        and PurePosixPath(name).name.lower().endswith("gnss.ride")
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            "Zip archive does not contain a GNSS ride file matching '*gnss.ride'."
        )
    formatted = ", ".join(PurePosixPath(name).name for name in candidates)
    raise ValueError(
        "Zip archive contains multiple GNSS ride files; unable to choose one. "
        f"Candidates: {formatted}"
    )


def _resolve_ride_pair_members(zf: zipfile.ZipFile) -> tuple[str, str]:
    """Return (left_member, right_member) for a `.zip` archive."""

    ride_members = [
        name
        for name in zf.namelist()
        if not name.endswith("/") and name.lower().endswith(".ride")
    ]

    strict_pattern = re.compile(
        r"^(?P<prefix>.*?)(?P<sep>[_-])(?P<side>left|right)(?P<suffix>\.ride)$",
        flags=re.IGNORECASE,
    )
    grouped: dict[tuple[str, str, str], dict[str, str]] = {}
    for member in ride_members:
        basename = PurePosixPath(member).name
        match = strict_pattern.match(basename)
        if not match:
            continue
        key = (
            match.group("prefix"),
            match.group("sep"),
            match.group("suffix").lower(),
        )
        side = match.group("side").lower()
        grouped.setdefault(key, {})[side] = member

    pairs = [
        (sides["left"], sides["right"]) for sides in grouped.values() if "left" in sides and "right" in sides
    ]
    if len(pairs) == 1:
        return pairs[0]

    if not pairs:
        lefts = [m for m in ride_members if "left" in PurePosixPath(m).name.lower()]
        rights = [m for m in ride_members if "right" in PurePosixPath(m).name.lower()]
        if len(lefts) == 1 and len(rights) == 1:
            return lefts[0], rights[0]
        raise ValueError(
            "Zip archive must contain a LEFT/RIGHT .ride pair (e.g. '*_left.ride' and '*_right.ride'). "
            f"Found {len(ride_members)} .ride files."
        )

    formatted = ", ".join(
        f"({PurePosixPath(left).name}, {PurePosixPath(right).name})" for left, right in pairs
    )
    raise ValueError(
        "Zip archive contains multiple LEFT/RIGHT .ride pairs; unable to choose one. "
        f"Pairs: {formatted}"
    )


def _parse_tabular_file(
    path_arg: Path | str,
    *,
    delimiter: str,
) -> Dict[str, List[Dict[str, str]]]:
    path = Path(path_arg).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp, delimiter=delimiter)
        devices: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in reader:
            if is_header_row(row):
                continue
            device = row.get("DeviceName")
            if device is None:
                raise KeyError("CSV is missing the required 'DeviceName' column")
            devices[device].append(row)

    sorted_devices: Dict[str, List[Dict[str, str]]] = {}
    for device, rows in devices.items():
        timestamped: List[tuple[datetime, Dict[str, str]]] = []
        for idx, row in enumerate(rows, start=1):
            time_value = row.get(TIME_COLUMN)
            if time_value is None:
                raise KeyError(
                    f"CSV is missing the {TIME_COLUMN!r} column required for sorting"
                )
            timestamped.append((parse_timestamp(time_value, idx), row))
        timestamped.sort(key=lambda pair: pair[0])
        sorted_devices[device] = [row for _, row in timestamped]
    return sorted_devices


def collect_group_data(
    rows: List[Dict[str, str]],
    column_groups: List[str],
    columns_map: Dict[str, tuple[str, ...]],
) -> tuple[List[datetime], Dict[str, Dict[str, List[float]]]]:
    """Collect numeric time series for the requested plot groups.

    The `sf0` group performs IMU-only sensor fusion using accelerometer +
    gyroscope (angular speed). Currently it outputs roll only.
    """

    column_data: Dict[str, Dict[str, List[float]]] = {
        group: {col: [] for col in columns}
        for group, columns in columns_map.items()
    }
    timestamps: List[datetime] = []

    # Lazy init of optional deps so `--help` still works without runtime extras.
    needs_ahrs = any(group in {"mad", "mah", "ekf"} for group in columns_map)
    needs_dt = any(
        group in {"mad", "mah", "ekf", "edge", "edge2", "sf0"}
        for group in columns_map
    )

    np = None
    q2euler = None
    EKF = Madgwick = Mahony = None

    ahrs_filters: Dict[str, object] = {}
    ahrs_q: Dict[str, Any] = {}
    last_euler_deg: Dict[str, tuple[float, float, float]] = {}
    last_dt: float = 0.01
    prev_ts: datetime | None = None
    edge_filter = None
    edge_q = None

    # Complementary filter state for IMU-only fusion (`sf0`).
    sf0_roll_rad = 0.0
    sf0_initialized = False
    sf0_alpha = 0.98

    if needs_ahrs:
        try:
            import numpy as np  # type: ignore[no-redef]
            from ahrs.common.orientation import q2euler  # type: ignore[no-redef]
            from ahrs.filters import EKF, Madgwick, Mahony  # type: ignore[no-redef]
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing optional dependency for 'mad'/'mah'/'ekf' groups. "
                "Install with `pip install ahrs`."
            ) from exc

        ahrs_filters = {
            "mad": Madgwick(),
            "mah": Mahony(),
            "ekf": EKF(),
        }
        ahrs_q = {
            "mad": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            "mah": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            "ekf": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        }
        last_euler_deg = {
            "mad": (0.0, 0.0, 0.0),
            "mah": (0.0, 0.0, 0.0),
            "ekf": (0.0, 0.0, 0.0),
        }
        edge_filter = Madgwick()
        edge_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    for idx, row in enumerate(rows, start=1):
        time_value = row.get(TIME_COLUMN)
        if time_value is None:
            raise KeyError(
                f"CSV is missing the {TIME_COLUMN!r} column required for the X axis"
            )
        ts = parse_timestamp(time_value, idx)
        timestamps.append(ts)

        if needs_dt:
            if prev_ts is not None:
                dt_candidate = (ts - prev_ts).total_seconds()
                if dt_candidate > 0.0:
                    last_dt = dt_candidate
            prev_ts = ts

        for group, columns in columns_map.items():
            if group == "q":
                quat = extract_quaternion_wxyz(row)
                if quat is None:
                    if column_data[group]["Q0()"]:
                        last_q0 = column_data[group]["Q0()"][-1]
                        last_q1 = column_data[group]["Q1()"][-1]
                        last_q2 = column_data[group]["Q2()"][-1]
                        last_q3 = column_data[group]["Q3()"][-1]
                        quat = (last_q0, last_q1, last_q2, last_q3)
                    else:
                        quat = (0.0, 0.0, 0.0, 0.0)
                q0, q1, q2, q3 = quat
                column_data[group]["Q0()"].append(q0)
                column_data[group]["Q1()"].append(q1)
                column_data[group]["Q2()"].append(q2)
                column_data[group]["Q3()"].append(q3)
                continue
            if group in {"edge", "edge2"}:
                def append_edge_from_quat(quat: tuple[float, float, float, float]) -> None:
                    edge_deg = quaternion_to_edge_deg(quat)
                    column_data[group]["Edge(°)"].append(edge_deg)
                    if group == "edge2":
                        _roll_deg, pitch_deg, yaw_deg = quaternion_to_euler_deg(quat)
                        column_data[group]["Pitch(°)"].append(pitch_deg)
                        column_data[group]["Yaw(°)"].append(yaw_deg)

                def append_edge_last() -> None:
                    last_edge = (
                        column_data[group]["Edge(°)"][-1]
                        if column_data[group]["Edge(°)"]
                        else 0.0
                    )
                    column_data[group]["Edge(°)"].append(last_edge)
                    if group == "edge2":
                        last_pitch = (
                            column_data[group]["Pitch(°)"][-1]
                            if column_data[group]["Pitch(°)"]
                            else 0.0
                        )
                        last_yaw = (
                            column_data[group]["Yaw(°)"][-1]
                            if column_data[group]["Yaw(°)"]
                            else 0.0
                        )
                        column_data[group]["Pitch(°)"].append(last_pitch)
                        column_data[group]["Yaw(°)"].append(last_yaw)

                quat = extract_quaternion_wxyz(row)
                if quat is not None:
                    append_edge_from_quat(quat)
                    continue

                if column_data[group]["Edge(°)"]:
                    append_edge_last()
                    continue

                # No quaternions in the input (or first samples are missing):
                # estimate them from acc+gyro (IMU) and then compute the
                # edging/roll angle from gravity.
                if edge_filter is None or edge_q is None:
                    try:
                        import numpy as np
                        from ahrs.filters import Madgwick
                    except ModuleNotFoundError as exc:
                        raise RuntimeError(
                            "Missing optional dependency for 'edge'/'edge2' group when quaternions "
                            "are not available. Install with `pip install ahrs`."
                        ) from exc
                    edge_filter = Madgwick()
                    edge_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

                ax_val = row.get("AccX(g)")
                ay_val = row.get("AccY(g)")
                az_val = row.get("AccZ(g)")
                gx_val = row.get("AsX(°/s)")
                gy_val = row.get("AsY(°/s)")
                gz_val = row.get("AsZ(°/s)")
                if (
                    ax_val is None
                    or ay_val is None
                    or az_val is None
                    or gx_val is None
                    or gy_val is None
                    or gz_val is None
                ):
                    raise KeyError(
                        "CSV is missing one of the required IMU columns for 'edge'/'edge2': "
                        "'AccX(g)', 'AccY(g)', 'AccZ(g)', 'AsX(°/s)', 'AsY(°/s)', 'AsZ(°/s)'"
                    )
                try:
                    ax = float(ax_val)
                    ay = float(ay_val)
                    az = float(az_val)
                    gx = float(gx_val)
                    gy = float(gy_val)
                    gz = float(gz_val)
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric entry in IMU columns at row {idx}"
                    ) from exc

                gyr = np.array(
                    [
                        math.radians(gx),
                        math.radians(gy),
                        math.radians(gz),
                    ],
                    dtype=float,
                )
                acc = np.array([ax, ay, az], dtype=float)

                hx_val = row.get("HX(uT)")
                hy_val = row.get("HY(uT)")
                hz_val = row.get("HZ(uT)")

                mag_present = False
                if hx_val is not None and hy_val is not None and hz_val is not None:
                    hx_s = hx_val.strip()
                    hy_s = hy_val.strip()
                    hz_s = hz_val.strip()
                    if (
                        hx_s
                        and hy_s
                        and hz_s
                        and hx_s.lower() != "null"
                        and hy_s.lower() != "null"
                        and hz_s.lower() != "null"
                    ):
                        try:
                            hx = float(hx_s)
                            hy = float(hy_s)
                            hz = float(hz_s)
                            mag_present = True
                        except ValueError:
                            mag_present = False

                # Some logs have duplicate timestamps. Reuse the last observed dt.
                dt = last_dt if last_dt > 0.0 else 0.01

                # If the acceleration magnitude is invalid, keep the last angle.
                if float(np.linalg.norm(acc)) == 0.0:
                    append_edge_last()
                    continue

                try:
                    if mag_present:
                        mag = np.array([hx, hy, hz], dtype=float)
                        edge_q = edge_filter.updateMARG(
                            q=edge_q,
                            gyr=gyr,
                            acc=acc,
                            mag=mag,
                            dt=dt,
                        )
                    else:
                        edge_q = edge_filter.updateIMU(
                            q=edge_q,
                            gyr=gyr,
                            acc=acc,
                            dt=dt,
                        )
                except Exception:
                    append_edge_last()
                    continue

                qw, qx, qy, qz = (
                    float(edge_q[0]),
                    float(edge_q[1]),
                    float(edge_q[2]),
                    float(edge_q[3]),
                )
                append_edge_from_quat((qw, qx, qy, qz))
                continue
            if group == "sf0":
                ax_val = row.get("AccX(g)")
                ay_val = row.get("AccY(g)")
                az_val = row.get("AccZ(g)")
                gx_val = row.get("AsX(°/s)")
                gy_val = row.get("AsY(°/s)")
                gz_val = row.get("AsZ(°/s)")
                if (
                    ax_val is None
                    or ay_val is None
                    or az_val is None
                    or gx_val is None
                    or gy_val is None
                    or gz_val is None
                ):
                    raise KeyError(
                        "CSV is missing one of the required IMU columns for 'sf0': "
                        "'AccX(g)', 'AccY(g)', 'AccZ(g)', 'AsX(°/s)', 'AsY(°/s)', 'AsZ(°/s)'"
                    )
                try:
                    ax = float(ax_val)
                    ay = float(ay_val)
                    az = float(az_val)
                    gx = float(gx_val)
                    gy = float(gy_val)
                    gz = float(gz_val)
                except ValueError as exc:
                    raise ValueError(f"Non-numeric entry in IMU columns at row {idx}") from exc

                # Some logs have duplicate timestamps. Reuse the last observed dt.
                dt = last_dt if last_dt > 0.0 else 0.01

                # Roll observation from gravity (accelerometer direction).
                # This formula assumes the sensor's axes follow the same
                # convention as used elsewhere in this file.
                acc_norm = math.sqrt(ax * ax + ay * ay + az * az)
                if acc_norm > 0.0:
                    roll_acc = math.atan2(ay, az)
                else:
                    roll_acc = sf0_roll_rad

                # Initialize using accelerometer so the first value isn't arbitrary.
                if not sf0_initialized:
                    sf0_roll_rad = roll_acc
                    sf0_initialized = True
                else:
                    # Gyro propagation (deg/s -> rad/s).
                    sf0_roll_rad += math.radians(gx) * dt

                    # Complementary correction using accelerometer.
                    if acc_norm > 0.0:
                        sf0_roll_rad = sf0_alpha * sf0_roll_rad + (1.0 - sf0_alpha) * roll_acc

                column_data[group]["Roll(°)"].append(math.degrees(sf0_roll_rad))
                continue
            if group in {"mad", "mah", "ekf"}:
                ax_val = row.get("AccX(g)")
                ay_val = row.get("AccY(g)")
                az_val = row.get("AccZ(g)")
                gx_val = row.get("AsX(°/s)")
                gy_val = row.get("AsY(°/s)")
                gz_val = row.get("AsZ(°/s)")
                if (
                    ax_val is None
                    or ay_val is None
                    or az_val is None
                    or gx_val is None
                    or gy_val is None
                    or gz_val is None
                ):
                    raise KeyError(
                        "CSV is missing one of the required IMU columns for 'mad'/'mah'/'ekf': "
                        "'AccX(g)', 'AccY(g)', 'AccZ(g)', 'AsX(°/s)', 'AsY(°/s)', 'AsZ(°/s)'"
                    )
                try:
                    ax = float(ax_val)
                    ay = float(ay_val)
                    az = float(az_val)
                    gx = float(gx_val)
                    gy = float(gy_val)
                    gz = float(gz_val)
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric entry in IMU columns at row {idx}"
                    ) from exc

                hx_val = row.get("HX(uT)")
                hy_val = row.get("HY(uT)")
                hz_val = row.get("HZ(uT)")

                mag_present = hx_val is not None and hy_val is not None and hz_val is not None
                if mag_present:
                    try:
                        hx = float(hx_val)
                        hy = float(hy_val)
                        hz = float(hz_val)
                    except ValueError as exc:
                        raise ValueError(
                            f"Non-numeric entry in magnetometer columns at row {idx}"
                        ) from exc

                gyr = np.array(
                    [
                        math.radians(gx),
                        math.radians(gy),
                        math.radians(gz),
                    ],
                    dtype=float,
                )
                acc = np.array([ax, ay, az], dtype=float)
                q = ahrs_q[group]
                filt = ahrs_filters[group]

                # Some logs have duplicate timestamps. Reuse the last observed dt.
                dt = last_dt if last_dt > 0.0 else 0.01

                # Guard against invalid acceleration samples.
                if float(np.linalg.norm(acc)) == 0.0:
                    pitch_deg, roll_deg, yaw_deg = last_euler_deg[group]
                    column_data[group]["Pitch(°)"].append(pitch_deg)
                    column_data[group]["Roll(°)"].append(roll_deg)
                    column_data[group]["Yaw(°)"].append(yaw_deg)
                    continue

                try:
                    if group == "ekf":
                        if mag_present:
                            mag = np.array([hx, hy, hz], dtype=float)
                            q = filt.update(q=q, gyr=gyr, acc=acc, mag=mag, dt=dt)
                        else:
                            q = filt.update(q=q, gyr=gyr, acc=acc, dt=dt)
                    else:
                        if mag_present:
                            mag = np.array([hx, hy, hz], dtype=float)
                            q = filt.updateMARG(q=q, gyr=gyr, acc=acc, mag=mag, dt=dt)
                        else:
                            q = filt.updateIMU(q=q, gyr=gyr, acc=acc, dt=dt)
                except Exception:
                    # If the filter update fails for a sample, keep the last output.
                    pitch_deg, roll_deg, yaw_deg = last_euler_deg[group]
                    column_data[group]["Pitch(°)"].append(pitch_deg)
                    column_data[group]["Roll(°)"].append(roll_deg)
                    column_data[group]["Yaw(°)"].append(yaw_deg)
                    continue

                ahrs_q[group] = q
                roll_rad, pitch_rad, yaw_rad = q2euler(q)
                roll_deg = math.degrees(float(roll_rad))
                pitch_deg = math.degrees(float(pitch_rad))
                yaw_deg = math.degrees(float(yaw_rad))
                last_euler_deg[group] = (pitch_deg, roll_deg, yaw_deg)

                column_data[group]["Pitch(°)"].append(pitch_deg)
                column_data[group]["Roll(°)"].append(roll_deg)
                column_data[group]["Yaw(°)"].append(yaw_deg)
                continue
            if group == "tilt":
                ax_val = row.get("AccX(g)")
                ay_val = row.get("AccY(g)")
                az_val = row.get("AccZ(g)")
                if ax_val is None or ay_val is None or az_val is None:
                    raise KeyError(
                        "CSV is missing one of the required accelerometer columns: 'AccX(g)', 'AccY(g)', 'AccZ(g)'"
                    )
                try:
                    ax = float(ax_val)
                    ay = float(ay_val)
                    az = float(az_val)
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric entry in accelerometer columns at row {idx}"
                    ) from exc

                # Standard pitch/roll from accelerometer (gravity vector).
                roll = math.degrees(math.atan2(ay, az))
                pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
                column_data[group]["Pitch(°)"].append(pitch)
                column_data[group]["Roll(°)"].append(roll)

                # Yaw/heading: tilt-compensated magnetometer heading.
                hx_val = row.get("HX(uT)")
                hy_val = row.get("HY(uT)")
                hz_val = row.get("HZ(uT)")
                if hx_val is None or hy_val is None or hz_val is None:
                    raise KeyError(
                        "CSV is missing one of the required magnetometer columns for yaw: 'HX(uT)', 'HY(uT)', 'HZ(uT)'"
                    )
                try:
                    hx = float(hx_val)
                    hy = float(hy_val)
                    hz = float(hz_val)
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric entry in magnetometer columns at row {idx}"
                    ) from exc

                roll_rad = math.radians(roll)
                pitch_rad = math.radians(pitch)
                mx2 = hx * math.cos(pitch_rad) + hz * math.sin(pitch_rad)
                my2 = (
                    hx * math.sin(roll_rad) * math.sin(pitch_rad)
                    + hy * math.cos(roll_rad)
                    - hz * math.sin(roll_rad) * math.cos(pitch_rad)
                )
                yaw = math.degrees(math.atan2(my2, mx2))
                column_data[group]["Yaw(°)"].append(yaw)
                continue
            for column in columns:
                value = row.get(column)
                if value is None:
                    raise KeyError(
                        f"CSV is missing the {column!r} column required for plotting"
                    )
                try:
                    numeric_value = float(value)
                    column_data[group][column].append(numeric_value)
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric entry {value!r} in column {column} at row {idx}"
                    ) from exc

    for group, columns in columns_map.items():
        for column in columns:
            if column.startswith("Angle") or column in {"Yaw(°)", "Edge(°)"}:
                column_data[group][column] = unwrap_degrees(column_data[group][column])
    return timestamps, column_data


def compute_moving_average(values: List[float], window: int) -> List[float]:
    """Return the simple moving average for the most recent `window` values."""

    if window <= 0:
        raise ValueError("ma_window must be a positive integer")

    averages: List[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        window_size = min(window, index + 1)
        averages.append(running_sum / window_size)
    return averages


def _estimate_sample_interval(timestamps: List[datetime]) -> float:
    """Return the average positive spacing between consecutive timestamps."""

    if len(timestamps) < 2:
        raise ValueError("Need at least two timestamps to estimate the sampling interval")

    intervals: List[float] = []
    for prev, curr in zip(timestamps, timestamps[1:]):
        delta = (curr - prev).total_seconds()
        if delta > 0.0:
            intervals.append(delta)
    if not intervals:
        raise ValueError("Timestamps must be strictly increasing to estimate sampling interval")
    return sum(intervals) / len(intervals)


def _calculate_frequency_spectrum(values: List[float], sample_interval: float) -> tuple[List[float], List[float]]:
    """Return frequency bins and magnitude from the real FFT of `values`."""

    if sample_interval <= 0:
        raise ValueError("Sample interval must be positive for FFT calculations")
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "numpy is required to compute FFT plots. Install it with `pip install numpy`."
        ) from exc

    samples = np.asarray(values, dtype=float)
    if samples.size < 2:
        raise ValueError("At least two samples are required to compute FFT")

    centered = samples - float(np.mean(samples))
    fft_values = np.fft.rfft(centered)
    frequencies = np.fft.rfftfreq(samples.size, d=sample_interval)
    magnitudes = np.abs(fft_values)
    return frequencies.tolist(), magnitudes.tolist()


def apply_iir_bandpass(
    values: List[float],
    sample_interval: float,
    filter_type: str,
    low_hz: float,
    high_hz: float,
) -> List[float]:
    """Return zero-phase IIR band-pass filtered samples using filtfilt.

    Supports filter_type:
      - b  : Butterworth
      - c1 : Chebyshev type I (0.5 dB ripple)
      - c2 : Chebyshev type II (40 dB stopband attenuation)
      - e  : Elliptic (0.5 dB ripple, 40 dB stopband attenuation)
    """

    if sample_interval <= 0.0:
        raise ValueError("Sample interval must be positive for band-pass filtering")
    if low_hz <= 0.0 or high_hz <= 0.0:
        raise ValueError("Band-pass cutoffs must be positive")
    if low_hz >= high_hz:
        raise ValueError("Band-pass low cutoff must be < high cutoff")

    nyquist = 0.5 / sample_interval
    if high_hz >= nyquist:
        raise ValueError(
            f"Band-pass high cutoff must be below Nyquist ({nyquist:.3f} Hz)"
        )

    try:
        from scipy import signal
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "scipy is required for band-pass filtering. Install with `pip install scipy`."
        ) from exc

    wn = [low_hz / nyquist, high_hz / nyquist]
    if filter_type == "b":
        b, a = signal.butter(4, wn, btype="bandpass")
    elif filter_type == "c1":
        b, a = signal.cheby1(4, 0.5, wn, btype="bandpass")
    elif filter_type == "c2":
        b, a = signal.cheby2(4, 40.0, wn, btype="bandpass")
    elif filter_type == "e":
        b, a = signal.ellip(4, 0.5, 40.0, wn, btype="bandpass")
    else:
        raise ValueError(f"Unsupported band-pass filter type: {filter_type}")

    min_len = max(len(a), len(b)) * 3
    if len(values) < min_len:
        raise ValueError(
            f"Not enough samples for band-pass filtering (need >= {min_len})"
        )

    filtered = signal.filtfilt(b, a, values)
    return filtered.tolist()


def plot_frequency_spectrum(
    rows_by_device: Dict[str, List[Dict[str, str]]],
    group: str,
    *,
    block: bool = True,
    plot_backend: PlotBackend = "mp",
) -> None:
    """Render a frequency-domain view for the first requested column group."""

    if plot_backend == "pl":
        _plot_frequency_spectrum_plotly(
            rows_by_device,
            group,
        )
        return

    if plt is None:
        raise RuntimeError(
            "matplotlib is required for FFT plotting. Install dependencies (e.g. `pip install matplotlib`)."
        )
    if group not in COLUMNS:
        raise KeyError(
            f"Unknown column group {group!r}; valid keys: {', '.join(COLUMNS)}"
        )
    if not rows_by_device:
        raise ValueError("No device data available for FFT plotting")

    columns = COLUMNS[group]
    columns_map = {group: columns}
    n_devices = len(rows_by_device)
    fig, axes = plt.subplots(
        n_devices,
        1,
        sharex=True,
        figsize=(12, 2 + n_devices * 1.5),
        squeeze=False,
    )
    axes_list = [axis[0] for axis in axes]

    alias_map = {
        device: device_to_sensor_label(device, fallback=f"sensor{idx + 1}")
        for idx, device in enumerate(rows_by_device)
    }

    for device_idx, (device, rows) in enumerate(rows_by_device.items()):
        ax = axes_list[device_idx]
        timestamps, column_data = collect_group_data(rows, [group], columns_map)
        alias = alias_map[device]
        try:
            sample_interval = _estimate_sample_interval(timestamps)
        except ValueError:
            ax.text(
                0.5,
                0.5,
                "Insufficient samples for FFT",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{alias} FFT")
            ax.set_ylabel(alias)
            ax.grid(True)
            continue

        plotted = False
        for column in columns:
            values = column_data[group][column]
            if len(values) < 2:
                continue
            try:
                frequencies, magnitudes = _calculate_frequency_spectrum(
                    values, sample_interval
                )
            except ValueError:
                continue
            ax.plot(frequencies, magnitudes, label=column)
            plotted = True

        ax.set_title(f"{alias} FFT")
        ax.set_ylabel(alias)
        ax.grid(True)
        if plotted:
            ax.legend(loc="upper right")
        else:
            ax.text(
                0.5,
                0.5,
                "No valid signal samples",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    axes_list[-1].set_xlabel("frequency (Hz)")
    subplot_params = dict(
        left=0.06,
        bottom=0.08,
        right=0.98,
        top=0.94,
        wspace=0.2,
        hspace=0.4,
    )
    fig.subplots_adjust(**subplot_params)
    plt.show(block=block)


def _require_plotly() -> tuple[Any, Any]:
    """Import Plotly lazily so --help works without optional deps."""

    try:
        import importlib

        go = importlib.import_module("plotly.graph_objects")
        subplots = importlib.import_module("plotly.subplots")
        make_subplots = getattr(subplots, "make_subplots")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "plotly is required for Plotly plotting. Install dependencies (e.g. `pip install plotly`)."
        ) from exc
    return go, make_subplots


def _plot_frequency_spectrum_plotly(
    rows_by_device: Dict[str, List[Dict[str, str]]],
    group: str,
) -> None:
    """Render a frequency-domain view using Plotly."""

    go, make_subplots = _require_plotly()

    if group not in COLUMNS:
        raise KeyError(
            f"Unknown column group {group!r}; valid keys: {', '.join(COLUMNS)}"
        )
    if not rows_by_device:
        raise ValueError("No device data available for FFT plotting")

    columns = COLUMNS[group]
    columns_map = {group: columns}
    n_devices = len(rows_by_device)

    alias_map = {
        device: device_to_sensor_label(device, fallback=f"sensor{idx + 1}")
        for idx, device in enumerate(rows_by_device)
    }

    subplot_titles = [f"{alias_map[device]} FFT" for device in rows_by_device]
    fig = make_subplots(
        rows=n_devices,
        cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.06,
    )

    for device_idx, (device, rows) in enumerate(rows_by_device.items(), start=1):
        timestamps, column_data = collect_group_data(rows, [group], columns_map)
        alias = alias_map[device]

        try:
            sample_interval = _estimate_sample_interval(timestamps)
        except ValueError:
            fig.add_annotation(
                x=0.5,
                y=0.5,
                xref=f"x{device_idx} domain" if device_idx > 1 else "x domain",
                yref=f"y{device_idx} domain" if device_idx > 1 else "y domain",
                text="Insufficient samples for FFT",
                showarrow=False,
            )
            fig.update_yaxes(title_text=alias, row=device_idx, col=1)
            continue

        plotted = False
        for column in columns:
            values = column_data[group][column]
            if len(values) < 2:
                continue
            try:
                frequencies, magnitudes = _calculate_frequency_spectrum(
                    values, sample_interval
                )
            except ValueError:
                continue
            plotted = True
            fig.add_trace(
                go.Scatter(
                    x=frequencies,
                    y=magnitudes,
                    mode="lines",
                    name=column,
                    showlegend=(device_idx == 1),
                ),
                row=device_idx,
                col=1,
            )

        fig.update_yaxes(title_text=alias, row=device_idx, col=1)
        if not plotted:
            fig.add_annotation(
                x=0.5,
                y=0.5,
                xref=f"x{device_idx} domain" if device_idx > 1 else "x domain",
                yref=f"y{device_idx} domain" if device_idx > 1 else "y domain",
                text="No valid signal samples",
                showarrow=False,
            )

    fig.update_xaxes(title_text="frequency (Hz)", row=n_devices, col=1)
    fig.update_layout(
        height=250 + n_devices * 220,
        showlegend=True,
        title_text=f"{group.upper()} FFT",
    )
    fig.show()


def plot_altitude(
    timestamps: List[datetime],
    altitudes: List[float],
    *,
    block: bool = True,
    plot_backend: PlotBackend = "mp",
) -> None:
    """Plot GNSS altitude vs time."""

    if not timestamps:
        raise ValueError("No GNSS altitude samples found")
    if len(timestamps) != len(altitudes):
        raise ValueError("GNSS timestamps and altitudes must have the same length")

    if plot_backend == "pl":
        go, _ = _require_plotly()
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=timestamps,
                    y=altitudes,
                    mode="lines",
                    name="altitude",
                )
            ]
        )
        fig.update_layout(
            title_text="ALT (GNSS)",
            xaxis_title="time",
            yaxis_title="altitude (m)",
            height=520,
        )
        fig.show()
        return

    if plt is None or mdates is None:
        raise RuntimeError(
            "matplotlib is required for plotting. Install dependencies (e.g. `pip install matplotlib`)."
        )

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(timestamps, altitudes, label="altitude")
    ax.set_title("ALT (GNSS)")
    ax.set_xlabel("time")
    ax.set_ylabel("altitude (m)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.grid(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show(block=block)


def device_to_sensor_label(device_name: str, *, fallback: str) -> str:
    """Map a DeviceName to a human label (e.g. LEFT/RIGHT).

    The `SENSORS` dict maps display labels (keys) to an identifier substring
    (values) that is expected to appear in the DeviceName column.

    Parameters:
        device_name: Raw DeviceName value from the file.
        fallback: Label to use when no `SENSORS` entry matches.

    Returns:
        A display label such as "LEFT" or "RIGHT".
    """

    normalized = device_name.strip().upper()
    if normalized in SENSORS:
        return normalized

    for label, identifier in SENSORS.items():
        if identifier and identifier in device_name:
            return label
    return fallback


def plot_devices(
    rows_by_device: Dict[str, List[Dict[str, str]]],
    column_groups: List[str],
    title: str | None = None,
    *,
    block: bool = True,
    ma_window: int = 5,
    plot_backend: PlotBackend = "mp",
    bandpass: tuple[str, float, float] | None = None,
) -> None:
    if plot_backend == "pl":
        _plot_devices_plotly(
            rows_by_device,
            column_groups,
            title,
            ma_window=ma_window,
            bandpass=bandpass,
        )
        return

    if plt is None or mdates is None:
        raise RuntimeError(
            "matplotlib is required for plotting. Install dependencies (e.g. `pip install matplotlib`)."
        )
    if not column_groups:
        raise ValueError("At least one column group must be specified for plotting")

    for group in column_groups:
        if group not in COLUMNS:
            raise KeyError(
                f"Unknown column group {group!r}; valid keys: {', '.join(COLUMNS)}"
            )

    if not rows_by_device:
        raise ValueError("No device data available for plotting")

    if ma_window < 0:
        raise ValueError("ma_window must be a positive integer")

    base_groups = column_groups

    if "q" in base_groups and not has_quaternion_values(rows_by_device):
        print("no Q values in this file")
        base_groups = [group for group in base_groups if group != "q"]

    if not base_groups:
        return

    columns_map = {group: COLUMNS[group] for group in base_groups}
    n_devices = len(rows_by_device)

    apply_bandpass = bandpass is not None and any(
        group in base_groups for group in {"edge", "edge2"}
    )
    axes_per_device = len(base_groups) + (1 if apply_bandpass else 0)
    total_axes = n_devices * axes_per_device
    fig, axes = plt.subplots(
        total_axes,
        1,
        sharex=True,
        figsize=(12, 2 + total_axes * 1.5),
        squeeze=False,
    )
    axes_list = [axis[0] for axis in axes]

    alias_map = {
        device: device_to_sensor_label(device, fallback=f"sensor{idx + 1}")
        for idx, device in enumerate(rows_by_device)
    }
    for device_idx, (device, rows) in enumerate(rows_by_device.items()):
        timestamps, column_data = collect_group_data(rows, base_groups, columns_map)
        for group_idx, group in enumerate(base_groups):
            axis_idx = device_idx * axes_per_device + group_idx
            ax = axes_list[axis_idx]
            for column in columns_map[group]:
                column_values = column_data[group][column]
                plot_color = ANGLE_COLORS.get(column) if group == "ang" else None
                ax.plot(timestamps, column_values, label=column, color=plot_color)
                if group != "sf0" and ma_window >= 2 and column_values:
                    ma_values = compute_moving_average(column_values, ma_window)
                    ax.plot(
                        timestamps,
                        ma_values,
                        label=f"{column} MA ({ma_window})",
                        color=plot_color,
                        linestyle="--",
                        linewidth=1.25,
                        alpha=0.85,
                    )
            alias = alias_map[device]
            ax.set_title(group.upper())
            ax.set_ylabel(alias)
            ax.grid(True)
            ax.legend(loc="upper right")

        if apply_bandpass:
            edge_group = "edge" if "edge" in base_groups else "edge2"
            axis_idx = device_idx * axes_per_device + len(base_groups)
            ax = axes_list[axis_idx]
            edge_values = column_data[edge_group]["Edge(°)"]
            try:
                sample_interval = _estimate_sample_interval(timestamps)
                filtered_edge = apply_iir_bandpass(
                    edge_values, sample_interval, bandpass[0], bandpass[1], bandpass[2]
                )
                filt_label = bandpass[0].upper()
                ax.plot(
                    timestamps,
                    filtered_edge,
                    label=f"Edge BP {filt_label} {bandpass[1]}-{bandpass[2]} Hz",
                    color="orange",
                )
                ax.legend(loc="upper right")
            except Exception as exc:
                ax.text(
                    0.5,
                    0.5,
                    str(exc),
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ax.set_title("EDGE band-pass")
            ax.set_ylabel(alias)
            ax.grid(True)

    axes_list[-1].set_xlabel("time")
    axes_list[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    subplot_params = dict(
        left=0.04,
        bottom=0.08,
        right=0.99,
        top=0.96,
        wspace=0.3,
        hspace=0.3,
    )
    fig.subplots_adjust(**subplot_params)
    fig.autofmt_xdate(bottom=subplot_params["bottom"])
    plt.show(block=block)


def _plot_devices_plotly(
    rows_by_device: Dict[str, List[Dict[str, str]]],
    column_groups: List[str],
    title: str | None,
    *,
    ma_window: int,
    bandpass: tuple[str, float, float] | None,
) -> None:
    """Render time-domain traces using Plotly."""

    go, make_subplots = _require_plotly()

    if not column_groups:
        raise ValueError("At least one column group must be specified for plotting")

    for group in column_groups:
        if group not in COLUMNS:
            raise KeyError(
                f"Unknown column group {group!r}; valid keys: {', '.join(COLUMNS)}"
            )

    if not rows_by_device:
        raise ValueError("No device data available for plotting")

    if ma_window < 0:
        raise ValueError("ma_window must be a positive integer")

    base_groups = column_groups

    if "q" in base_groups and not has_quaternion_values(rows_by_device):
        print("no Q values in this file")
        base_groups = [group for group in base_groups if group != "q"]

    if not base_groups:
        return

    columns_map = {group: COLUMNS[group] for group in base_groups}
    n_devices = len(rows_by_device)

    apply_bandpass = bandpass is not None and any(
        group in base_groups for group in {"edge", "edge2"}
    )
    axes_per_device = len(base_groups) + (1 if apply_bandpass else 0)
    total_axes = n_devices * axes_per_device

    alias_map = {
        device: device_to_sensor_label(device, fallback=f"sensor{idx + 1}")
        for idx, device in enumerate(rows_by_device)
    }

    subplot_titles: List[str] = []
    for device in rows_by_device:
        for group in base_groups:
            subplot_titles.append(group.upper())
        if apply_bandpass:
            subplot_titles.append("EDGE BAND-PASS")

    fig = make_subplots(
        rows=total_axes,
        cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.03,
    )

    shown_legend_names: set[str] = set()

    for device_idx, (device, rows) in enumerate(rows_by_device.items()):
        timestamps, column_data = collect_group_data(rows, base_groups, columns_map)
        alias = alias_map[device]
        for group_idx, group in enumerate(base_groups):
            row_idx = device_idx * axes_per_device + group_idx + 1
            fig.update_yaxes(title_text=alias, row=row_idx, col=1)
            for column in columns_map[group]:
                column_values = column_data[group][column]
                plot_color = ANGLE_COLORS.get(column) if group == "ang" else None

                name = column
                showlegend = name not in shown_legend_names
                shown_legend_names.add(name)
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=column_values,
                        mode="lines",
                        name=name,
                        showlegend=showlegend,
                        line={"color": plot_color} if plot_color else None,
                    ),
                    row=row_idx,
                    col=1,
                )

                if group != "sf0" and ma_window >= 2 and column_values:
                    ma_values = compute_moving_average(column_values, ma_window)
                    ma_name = f"{column} MA ({ma_window})"
                    showlegend = ma_name not in shown_legend_names
                    shown_legend_names.add(ma_name)
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=ma_values,
                            mode="lines",
                            name=ma_name,
                            showlegend=showlegend,
                            line={
                                "color": plot_color,
                                "dash": "dash",
                                "width": 2,
                            }
                            if plot_color
                            else {"dash": "dash", "width": 2},
                            opacity=0.85,
                        ),
                        row=row_idx,
                        col=1,
                    )

            if apply_bandpass:
                edge_group = "edge" if "edge" in base_groups else "edge2"
                row_idx = device_idx * axes_per_device + len(base_groups) + 1
                fig.update_yaxes(title_text=alias, row=row_idx, col=1)
                edge_values = column_data[edge_group]["Edge(°)"]
                try:
                    sample_interval = _estimate_sample_interval(timestamps)
                    filtered_edge = apply_iir_bandpass(
                        edge_values, sample_interval, bandpass[0], bandpass[1], bandpass[2]
                    )
                    filt_label = bandpass[0].upper()
                    name = f"Edge BP {filt_label} {bandpass[1]}-{bandpass[2]} Hz"
                    showlegend = name not in shown_legend_names
                    shown_legend_names.add(name)
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=filtered_edge,
                            mode="lines",
                            name=name,
                            showlegend=showlegend,
                            line={"color": "orange"},
                        ),
                        row=row_idx,
                        col=1,
                    )
                except Exception as exc:
                    fig.add_annotation(
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=(row_idx - 0.5) / total_axes,
                        text=str(exc),
                        showarrow=False,
                    )

    fig.update_xaxes(title_text="time", row=total_axes, col=1)
    fig.update_layout(
        height=320 + total_axes * 220,
        showlegend=True,
        title_text=title or "SC traces",
    )
    fig.show()


def parse_group_list(value: str) -> List[str]:
    groups = [group.strip().lower() for group in value.split(",") if group.strip()]
    if not groups:
        raise argparse.ArgumentTypeError("At least one column group must be provided")
    invalid = [group for group in groups if group not in COLUMNS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown column groups: {', '.join(invalid)}; valid keys are: {', '.join(COLUMNS)}"
        )
    return groups


def parse_bandpass(value: str) -> tuple[str, float, float]:
    parts = value.split("|")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "band-pass must be formatted as '<type>|<low>|<high>' (Hz)"
        )

    filter_type = parts[0].strip().lower()
    if filter_type not in {"b", "c1", "c2", "e"}:
        raise argparse.ArgumentTypeError(
            "band-pass type must be one of: b (Butterworth), c1 (Chebyshev I), "
            "c2 (Chebyshev II), e (Elliptic)"
        )

    try:
        low = float(parts[1])
        high = float(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("band-pass cutoffs must be numbers") from exc

    if low <= 0.0 or high <= 0.0:
        raise argparse.ArgumentTypeError("band-pass cutoffs must be positive")
    if low >= high:
        raise argparse.ArgumentTypeError("band-pass low cutoff must be < high cutoff")
    return filter_type, low, high


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SC sensor export (TSV/CSV) traces by column group",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "file",
        help="Path to the TSV/CSV/.ride/.zip file to parse",
    )
    parser.add_argument(
        "-g",
        "--group",
        default="acc",
        help=(
            "Comma-separated column groups to visualize\n"
            f"(valid: {', '.join(COLUMNS.keys())}; default: %(default)s)\n"
            "ang  - angles\n"
            "acc  - accelerations\n"
            "as   - angular speed (gyroscope)\n"
            "h    - magnetic field (magnetometer)\n"
            "q    - quaternion components (Q0..Q3)\n"
            "edge - edging angle around boot forward axis\n"
            "edge2 - edging angle plus pitch/yaw from same fusion\n"
            "alt  - GNSS altitude from *gnss.ride in a .zip\n"
            "tilt - pitch/roll from accelerometer + yaw heading\n"
            "sf0  - IMU-only roll fusion (acc + gyro)\n"
            "mad  - Madgwick filter\n"
            "mah  - Mahony filter\n"
            "ekf  - Extended Kalman Filter"
        ),
    )
    parser.add_argument(
        "-bp",
        "--bandpass",
        type=parse_bandpass,
        help=(
            "Apply IIR band-pass filter to edge traces and plot the filtered "
            "result as an extra subplot. Format: <type>|<low>|<high> (Hz). "
            "Types: b (Butterworth), c1 (Chebyshev I), c2 (Chebyshev II), "
            "e (Elliptic). Ignored unless 'edge' or 'edge2' is requested."
        ),
    )
    parser.add_argument(
        "-p",
        "--plot",
        default="mp",
        choices=("mp", "pl"),
        help=(
            "Select plotting backend (default: %(default)s)\n"
            "mp - matplotlib\n"
            "pl - plotly"
        ),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=PACKAGE_VERSION,
        help="Show the installed package version",
    )
    parser.add_argument(
        "-ma",
        "--ma-window",
        type=int,
        default=5,
        help="Simple moving average window size per signal (default: %(default)s)",
    )
    parser.add_argument(
        "-f",
        "--fft",
        action="store_true",
        help=(
            "Plot the FFT of the first group listed in --group"
        ),
    )
    args = parser.parse_args()
    args.groups = parse_group_list(args.group)
    return args


def main() -> None:
    args = parse_args()
    plot_backend: PlotBackend = args.plot
    if plot_backend == "mp" and plt is None:
        raise RuntimeError(
            "matplotlib is required for plotting. Install dependencies (e.g. `pip install matplotlib`)."
        )

    did_plot = False

    input_path = Path(args.file).expanduser()
    suffix = input_path.suffix.lower()
    groups: List[str] = list(args.groups)
    if "alt" in groups:
        if suffix != ".zip":
            raise ValueError("alt group is only supported for .zip inputs")
        try:
            gnss_timestamps, gnss_altitudes = parse_gnss_ride_zip(input_path)
        except ValueError:
            print("gnss data not found in this zip")
        else:
            plot_altitude(
                gnss_timestamps,
                gnss_altitudes,
                block=False,
                plot_backend=plot_backend,
            )
            did_plot = True
        groups = [group for group in groups if group != "alt"]

    if not groups:
        rows_by_device: Dict[str, List[Dict[str, str]]] = {}
    elif suffix == ".tsv":
        rows_by_device = parse_tsv_file(input_path)
    elif suffix == ".csv":
        rows_by_device = parse_csv_file(input_path)
    elif suffix == ".ride":
        rows_by_device = parse_ride_pair(input_path)
    elif suffix == ".zip":
        rows_by_device = parse_ride_zip(input_path)
    else:
        raise ValueError(
            "Unsupported file extension. Expected one of: .tsv, .csv, .ride, .zip. "
            f"Got: {input_path}"
        )
    # total_rows = sum(len(rows) for rows in rows_by_device.values())
    # print(f"Parsed {total_rows} rows from {args.file}")
    if rows_by_device:
        plot_devices(
            rows_by_device,
            column_groups=groups,
            title=f"{'/'.join(group.upper() for group in groups)} traces",
            block=False,
            ma_window=args.ma_window,
            plot_backend=plot_backend,
            bandpass=args.bandpass,
        )
        did_plot = True
        if args.fft:
            plot_frequency_spectrum(
                rows_by_device,
                groups[0],
                block=False,
                plot_backend=plot_backend,
            )

    if plot_backend == "mp" and did_plot:
        plt.show()


if __name__ == "__main__":
    main()

