import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import datetime

CSV_FILE_PATH = "/mnt/c/Users/DSavkovic/Downloads/SC/20260103105034.csv"


# `time` is the axis we use for all plots. Support multiple formats so microseconds remain optional.
TIME_COLUMN = "time"
TIME_FORMATS = (
    "%m/%d/%Y %H:%M:%S.%f",
    "%m/%d/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
)

def is_header_row(row: Dict[str, str]) -> bool:
    time_value = row.get(TIME_COLUMN, "")
    return isinstance(time_value, str) and time_value.strip().lower() == TIME_COLUMN.lower()


COLUMNS = {
    "acc": ("AccX(g)", "AccY(g)", "AccZ(g)"),
    "as": ("AsX(°/s)", "AsY(°/s)", "AsZ(°/s)"),
    "ang": ("AngleX(°)", "AngleY(°)", "AngleZ(°)"),
    "h": ("HX(uT)", "HY(uT)", "HZ(uT)")
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
    for idx, row in enumerate(rows, start=1):
        time_value = row.get(TIME_COLUMN)
        if time_value is None:
            raise KeyError(
                f"CSV is missing the {TIME_COLUMN!r} column required for the X axis"
            )
        timestamps.append(parse_timestamp(time_value, idx))

        for group, columns in columns_map.items():
            for column in columns:
                value = row.get(column)
                if value is None:
                    raise KeyError(
                        f"CSV is missing the {column!r} column required for plotting"
                    )
                try:
                    column_data[group][column].append(float(value))
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric entry {value!r} in column {column} at row {idx}"
                    ) from exc
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


def plot_devices(
    rows_by_device: Dict[str, List[Dict[str, str]]],
    column_groups: List[str],
    title: str | None = None,
    *,
    block: bool = True,
    ma_window: int = 5,
) -> None:
    if not column_groups:
        raise ValueError("At least one column group must be specified for plotting")

    for group in column_groups:
        if group not in COLUMNS:
            raise KeyError(
                f"Unknown column group {group!r}; valid keys: {', '.join(COLUMNS)}"
            )

    if not rows_by_device:
        raise ValueError("No device data available for plotting")

    if ma_window <= 0:
        raise ValueError("ma_window must be a positive integer")

    columns_map = {group: COLUMNS[group] for group in column_groups}
    n_devices = len(rows_by_device)
    total_axes = n_devices * len(column_groups)
    fig, axes = plt.subplots(
        total_axes,
        1,
        sharex=True,
        figsize=(12, 2 + total_axes * 1.5),
        squeeze=False,
    )
    axes_list = [axis[0] for axis in axes]

    alias_map = {device: f"sensor{idx + 1}" for idx, device in enumerate(rows_by_device)}
    for device_idx, (device, rows) in enumerate(rows_by_device.items()):
        timestamps, column_data = collect_group_data(rows, column_groups, columns_map)
        for group_idx, group in enumerate(column_groups):
            axis_idx = device_idx * len(column_groups) + group_idx
            ax = axes_list[axis_idx]
            for column in columns_map[group]:
                column_values = column_data[group][column]
                ax.plot(timestamps, column_values, label=column)
                if ma_window >= 2 and column_values:
                    ma_values = compute_moving_average(column_values, ma_window)
                    ax.plot(
                        timestamps,
                        ma_values,
                        label=f"{column} MA ({ma_window})",
                        linestyle="--",
                        linewidth=1.25,
                        alpha=0.85,
                    )
            alias = alias_map[device]
            ax.set_title(f"{alias} — {group.upper()}")
            ax.set_ylabel("Sensor value")
            ax.grid(True)
            ax.legend(loc="upper right")

    axes_list[-1].set_xlabel("time")
    axes_list[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    subplot_params = dict(
        left=0.03,
        bottom=0.1,
        right=0.97,
        top=0.97,
        wspace=0.2,
        hspace=0.2,
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
    parser = argparse.ArgumentParser(description="Read CSV data for SC script")
    parser.add_argument(
        "file",
        nargs="?",
        default=CSV_FILE_PATH,
        help="Path to the CSV file to parse",
    )
    parser.add_argument(
        "-g",
        "--group",
        default="acc",
        help="Comma-separated column groups to visualize (valid: acc, as, ang, h; default: %(default)s)",
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

