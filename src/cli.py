import argparse
import csv
import math
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Dict, List, Optional

from datetime import datetime

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
    "angd": ("AngleX(°)", "AngleY(°)", "AngleZ(°)"),  # sensor1 - sensor2 angle delta
    "accn": ("AccNorm(g)",),  # normalized acceleration magnitude
    "asn": ("AsNorm(°/s)",), # normalized angular speed magnitude
    "hn": ("HNorm(uT)",),  # normalized magnetic field magnitude
    "tilt": ("Pitch(°)", "Roll(°)", "Yaw(°)"),  # pitch/roll from accelerometer + yaw from magnetometer
    "mad": ("Pitch(°)", "Roll(°)", "Yaw(°)"),  # Madgwick filter
    "mah": ("Pitch(°)", "Roll(°)", "Yaw(°)"),  # Mahony filter
    "ekf": ("Pitch(°)", "Roll(°)", "Yaw(°)"),  # Extended Kalman Filter
}

ANGLE_COLORS = {
    "AngleX(°)": "blue",
    "AngleY(°)": "green",
    "AngleZ(°)": "magenta",
}


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
    column_data: Dict[str, Dict[str, List[float]]] = {
        group: {col: [] for col in columns}
        for group, columns in columns_map.items()
    }
    timestamps: List[datetime] = []

    # Lazy init of AHRS filters so `--help` still works without optional deps.
    needs_ahrs = any(group in {"mad", "mah", "ekf"} for group in columns_map)
    if needs_ahrs:
        try:
            import numpy as np
            from ahrs.common.orientation import q2euler
            from ahrs.filters import EKF, Madgwick, Mahony
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing optional dependency for 'mad'/'mah'/'ekf' groups. "
                "Install with `pip install ahrs`."
            ) from exc

        ahrs_filters: Dict[str, object] = {
            "mad": Madgwick(),
            "mah": Mahony(),
            "ekf": EKF(),
        }
        ahrs_q: Dict[str, "np.ndarray"] = {
            "mad": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            "mah": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            "ekf": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        }
        last_euler_deg: Dict[str, tuple[float, float, float]] = {
            "mad": (0.0, 0.0, 0.0),
            "mah": (0.0, 0.0, 0.0),
            "ekf": (0.0, 0.0, 0.0),
        }
        last_dt: float = 0.01
        prev_ts: datetime | None = None
    for idx, row in enumerate(rows, start=1):
        time_value = row.get(TIME_COLUMN)
        if time_value is None:
            raise KeyError(
                f"CSV is missing the {TIME_COLUMN!r} column required for the X axis"
            )
        ts = parse_timestamp(time_value, idx)
        timestamps.append(ts)

        if needs_ahrs:
            if prev_ts is not None:
                dt_candidate = (ts - prev_ts).total_seconds()
                if dt_candidate > 0.0:
                    last_dt = dt_candidate
            prev_ts = ts

        for group, columns in columns_map.items():
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
            if group == "hn":
                hx = row.get("HX(uT)")
                hy = row.get("HY(uT)")
                hz = row.get("HZ(uT)")
                if hx is None or hy is None or hz is None:
                    raise KeyError(
                        "CSV is missing one of the required magnetometer columns: 'HX(uT)', 'HY(uT)', 'HZ(uT)'"
                    )
                try:
                    bx = float(hx)
                    by = float(hy)
                    bz = float(hz)
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric entry in magnetometer columns at row {idx}"
                    ) from exc

                bmag = math.sqrt(bx * bx + by * by + bz * bz)
                column_data[group]["HNorm(uT)"].append(bmag)
                continue
            if group == "accn":
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
                amag = math.sqrt(ax * ax + ay * ay + az * az)
                column_data[group]["AccNorm(g)"].append(amag)
                continue
            if group == "asn":
                gx_val = row.get("AsX(°/s)")
                gy_val = row.get("AsY(°/s)")
                gz_val = row.get("AsZ(°/s)")
                if gx_val is None or gy_val is None or gz_val is None:
                    raise KeyError(
                        "CSV is missing one of the required gyroscope columns: 'AsX(°/s)', 'AsY(°/s)', 'AsZ(°/s)'"
                    )
                try:
                    gx = float(gx_val)
                    gy = float(gy_val)
                    gz = float(gz_val)
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric entry in gyroscope columns at row {idx}"
                    ) from exc
                gmag = math.sqrt(gx * gx + gy * gy + gz * gz)
                column_data[group]["AsNorm(°/s)"].append(gmag)
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
            if column.startswith("Angle") or column == "Yaw(°)":
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
) -> None:
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

    has_angd = "angd" in column_groups
    base_groups = [group for group in column_groups if group != "angd"]
    columns_map = {group: COLUMNS[group] for group in base_groups}
    n_devices = len(rows_by_device)
    total_axes = n_devices * len(base_groups) + (1 if has_angd else 0)
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
            axis_idx = device_idx * len(base_groups) + group_idx
            ax = axes_list[axis_idx]
            for column in columns_map[group]:
                column_values = column_data[group][column]
                plot_color = ANGLE_COLORS.get(column) if group == "ang" else None
                ax.plot(timestamps, column_values, label=column, color=plot_color)
                if ma_window >= 2 and column_values:
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

    if has_angd:
        if len(rows_by_device) < 2:
            raise ValueError("angd requires at least two sensors (LEFT-RIGHT)")

        (device1, rows1), (device2, rows2) = list(rows_by_device.items())[:2]
        timestamps1, data1 = collect_group_data(rows1, ["angd"], {"angd": COLUMNS["angd"]})
        timestamps2, data2 = collect_group_data(rows2, ["angd"], {"angd": COLUMNS["angd"]})

        angle_cols = COLUMNS["angd"]
        min_len = min(len(timestamps1), len(timestamps2))
        if min_len == 0:
            raise ValueError("angd requires angle samples in both sensors")

        ax = axes_list[-1]
        for column in angle_cols:
            values1 = data1["angd"][column][:min_len]
            values2 = data2["angd"][column][:min_len]
            diff_values = [v1 - v2 for v1, v2 in zip(values1, values2)]

            plot_color = ANGLE_COLORS.get(column)

            if column.startswith("Angle"):
                label = f"Δ{column.split('(')[0].strip()}"
            else:
                label = f"Δ{column}"
            ax.plot(timestamps1[:min_len], diff_values, label=label, color=plot_color)
            if ma_window >= 2 and diff_values:
                ma_values = compute_moving_average(diff_values, ma_window)
                ax.plot(
                    timestamps1[:min_len],
                    ma_values,
                    label=f"{label} MA ({ma_window})",
                    color=plot_color,
                    linestyle="--",
                    linewidth=1.25,
                    alpha=0.85,
                )

        ax.set_title("ANGD")
        ax.set_ylabel("Δ angle")
        ax.grid(True)
        ax.legend(loc="upper right")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SC sensor export (TSV/CSV) traces by column group",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "file",
        help="Path to the CSV or TSV file to parse",
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
            "angd - LEFT-RIGHT angle delta\n"
            "accn - acceleration magnitude (norm)\n"
            "asn  - angular speed magnitude (norm)\n"
            "hn   - magnetic field magnitude (norm)\n"
            "tilt - pitch/roll from accelerometer + yaw heading\n"
            "mad  - Madgwick filter\n"
            "mah  - Mahony filter\n"
            "ekf  - Extended Kalman Filter"
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
    args = parser.parse_args()
    args.groups = parse_group_list(args.group)
    return args


def main() -> None:
    args = parse_args()
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plotting. Install dependencies (e.g. `pip install matplotlib`)."
        )
    rows_by_device = parse_tsv_file(args.file)
    # total_rows = sum(len(rows) for rows in rows_by_device.values())
    # print(f"Parsed {total_rows} rows from {args.file}")
    if rows_by_device:
        plot_devices(
            rows_by_device,
            column_groups=args.groups,
            title=f"{'/'.join(group.upper() for group in args.groups)} traces",
            block=False,
            ma_window=args.ma_window,
        )
        plt.show()


if __name__ == "__main__":
    main()

