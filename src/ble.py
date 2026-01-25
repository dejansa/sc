"""Helpers for interacting with WIT MOTION BLE 5.0 sensors.

Implements the Bluetooth 5.0 communication protocol documented at:
https://wit-motion.gitbook.io/witmotion-sdk/ble-5.0-protocol/bluetooth-5.0-communication-protocol

The module provides scanning, command building, and packet parsing utilities
so you can read IMU data and configure streaming parameters without needing
to re-implement the low-level protocol every time.
"""

import argparse
import asyncio
import datetime as dt
import inspect
import logging
import struct
import sys
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Iterator, List, Optional, Tuple

from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice
from bleak.exc import BleakError

LOGGER = logging.getLogger(__name__)

# The GitBook protocol description focuses on the byte-level packets; vendors
# sometimes ship different GATT UUIDs per hardware revision. We therefore
# discover the notify/write characteristics at runtime instead of hard-coding.

BLE_PACKET_SIZE = 20

ACC_SCALE = 16.0 / 32768.0
GYRO_SCALE = 2000.0 / 32768.0
ANGLE_SCALE = 180.0 / 32768.0

REGISTER_READ_PREFIX = bytes([0xFF, 0xAA, 0x27])


def build_read_register_command(register: int) -> bytes:
    """Build a BLE 5.0 command to read a register window.

    Protocol: FF AA 27 XX 00
    """

    if not 0 <= register <= 0xFF:
        raise ValueError("register must fit in one byte")
    return REGISTER_READ_PREFIX + bytes([register, 0x00])


def build_set_rate_command(rate_code: int) -> bytes:
    """Build a BLE 5.0 command to set the module return rate.

    Protocol: FF AA 03 RATE 00
    """

    if not 0 <= rate_code <= 0xFF:
        raise ValueError("rate_code must fit in one byte")
    return bytes([0xFF, 0xAA, 0x03, rate_code, 0x00])


def build_save_config_command(mode: int = 0) -> bytes:
    """Build a BLE 5.0 command to save configuration.

    Protocol: FF AA 00 SAVE 00
    SAVE: 0 -> save current configuration
          1 -> restore defaults and save
    """

    if mode not in {0, 1}:
        raise ValueError("mode must be 0 or 1")
    return bytes([0xFF, 0xAA, 0x00, mode, 0x00])


@dataclass
class WitMotionMeasurement:
    """A best-effort decoded sample from WIT MOTION BLE packets."""

    acc: Optional[Tuple[float, float, float]] = None
    gyro: Optional[Tuple[float, float, float]] = None
    angle: Optional[Tuple[float, float, float]] = None
    mag_mg: Optional[Tuple[int, int, int]] = None
    quat: Optional[Tuple[float, float, float, float]] = None
    datetime: Optional[dt.datetime] = None
    power_raw: Optional[int] = None
    start_register: Optional[int] = None
    raw: bytes = b""

    def __str__(self) -> str:  # pragma: no cover - simple helper
        parts: List[str] = []
        if self.acc is not None:
            parts.append(
                f"ACC(g)=({self.acc[0]:.3f},{self.acc[1]:.3f},{self.acc[2]:.3f})"
            )
        if self.gyro is not None:
            parts.append(
                f"GYRO(°/s)=({self.gyro[0]:.1f},{self.gyro[1]:.1f},{self.gyro[2]:.1f})"
            )
        if self.angle is not None:
            parts.append(
                f"ANGLE(°)=({self.angle[0]:.1f},{self.angle[1]:.1f},{self.angle[2]:.1f})"
            )
        if self.mag_mg is not None:
            parts.append(
                f"MAG(mG)=({self.mag_mg[0]},{self.mag_mg[1]},{self.mag_mg[2]})"
            )
        if self.quat is not None:
            parts.append(
                f"Q=({self.quat[0]:.3f},{self.quat[1]:.3f},{self.quat[2]:.3f},{self.quat[3]:.3f})"
            )
        if self.datetime is not None:
            parts.append(self.datetime.isoformat(sep=" ", timespec="seconds"))
        if self.power_raw is not None:
            parts.append(f"POWER(raw)={self.power_raw}")
        if self.start_register is not None:
            parts.append(f"reg=0x{self.start_register:02X}")
        if not parts:
            return f"RAW({len(self.raw)}B): {self.raw.hex(' ')}"
        return " | ".join(parts)


async def discover_witmotion_devices(
    timeout: float = 5.0, name_filter: str = "WT901"
) -> List[BLEDevice]:
    """Scan for nearby WIT MOTION peripherals and return matching BLEDevices."""

    LOGGER.info("Scanning for WIT MOTION devices (timeout=%.1fs)", timeout)
    devices = await BleakScanner.discover(timeout=timeout)
    matches = [
        device
        for device in devices
        if name_filter.lower() in (device.name or "").lower()
    ]
    LOGGER.info("Found %d matching device(s)", len(matches))
    return matches


def _decode_i16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<h", data, offset)[0]


def _decode_u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def iter_sensor_measurements(data: bytes) -> Iterator[WitMotionMeasurement]:
    """Iterate decoded measurements found within a raw notification payload.

    Some firmware versions batch multiple 20-byte frames into one BLE
    notification. This yields each recognized frame rather than only the first.

    Per GitBook:
      - 0x55 0x61: default 20-byte packet with acc + gyro + angle
      - 0x55 0x71: register window (start register + 8 registers)
    """

    if not data:
        yield WitMotionMeasurement(raw=b"")
        return

    offset = 0
    yielded = False
    while offset + 2 <= len(data):
        if data[offset] != 0x55:
            offset += 1
            continue
        if offset + BLE_PACKET_SIZE > len(data):
            break

        flag = data[offset + 1]
        packet = data[offset : offset + BLE_PACKET_SIZE]
        offset += BLE_PACKET_SIZE

        if flag == 0x61:
            ax = _decode_i16(packet, 2) * ACC_SCALE
            ay = _decode_i16(packet, 4) * ACC_SCALE
            az = _decode_i16(packet, 6) * ACC_SCALE
            gx = _decode_i16(packet, 8) * GYRO_SCALE
            gy = _decode_i16(packet, 10) * GYRO_SCALE
            gz = _decode_i16(packet, 12) * GYRO_SCALE
            roll = _decode_i16(packet, 14) * ANGLE_SCALE
            pitch = _decode_i16(packet, 16) * ANGLE_SCALE
            yaw = _decode_i16(packet, 18) * ANGLE_SCALE
            yielded = True
            yield WitMotionMeasurement(
                acc=(ax, ay, az),
                gyro=(gx, gy, gz),
                angle=(roll, pitch, yaw),
                raw=packet,
            )
            continue

        if flag == 0x71:
            reg = packet[2] | (packet[3] << 8)
            regs = packet[4:]
            measurement = WitMotionMeasurement(start_register=reg, raw=packet)
            if reg == 0x3A and len(regs) >= 6:
                hx = _decode_i16(regs, 0)
                hy = _decode_i16(regs, 2)
                hz = _decode_i16(regs, 4)
                measurement.mag_mg = (hx, hy, hz)
            if reg == 0x51 and len(regs) >= 8:
                q0 = _decode_i16(regs, 0) / 32768.0
                q1 = _decode_i16(regs, 2) / 32768.0
                q2 = _decode_i16(regs, 4) / 32768.0
                q3 = _decode_i16(regs, 6) / 32768.0
                measurement.quat = (q0, q1, q2, q3)
            if reg == 0x30 and len(regs) >= 6:
                year = 2000 + (regs[0] & 0xFF)
                month = regs[1] & 0xFF
                day = regs[2] & 0xFF
                hour = regs[3] & 0xFF
                minute = regs[4] & 0xFF
                second = regs[5] & 0xFF
                try:
                    measurement.datetime = dt.datetime(
                        year, month, day, hour, minute, second
                    )
                except ValueError:
                    pass
            if reg == 0x64 and len(regs) >= 2:
                measurement.power_raw = _decode_u16(regs, 0)
            yielded = True
            yield measurement
            continue

    if not yielded:
        yield WitMotionMeasurement(raw=bytes(data))


def parse_sensor_packet(data: bytes) -> WitMotionMeasurement:
    """Parse the first decoded measurement from a raw payload."""

    for measurement in iter_sensor_measurements(data):
        return measurement
    return WitMotionMeasurement(raw=bytes(data))


def _iter_characteristics(client: BleakClient):
    services = getattr(client, "services", None)
    if not services:
        return

    for service in services:
        for char in service.characteristics:
            yield service.uuid, char


def _uuid16(uuid: str) -> Optional[int]:
    """Extract a 16-bit UUID value from a Bluetooth base UUID string."""

    prefix = "0000"
    base = "-0000-1000-8000-00805f9b34fb"
    lower = uuid.lower()
    if not (lower.startswith(prefix) and lower.endswith(base) and len(lower) >= 8):
        return None
    try:
        return int(lower[4:8], 16)
    except ValueError:
        return None


def _is_vendor_uuid(uuid: str) -> bool:
    """Return True for vendor-specific 16-bit UUIDs (0xFF00-0xFFFF)."""

    uuid16 = _uuid16(uuid)
    return uuid16 is not None and uuid16 >= 0xFF00


def _is_standard_uuid(uuid: str) -> bool:
    """Return True for standard 16-bit GATT UUIDs (0x0001-0xFFFF, excluding 0xFF00+).

    Note: This is used only for ranking candidates.
    """

    uuid16 = _uuid16(uuid)
    return uuid16 is not None and uuid16 < 0xFF00


def _debug_log_characteristics(client: BleakClient) -> None:
    """Log all discovered characteristics at DEBUG level for troubleshooting."""

    for service_uuid, char in _iter_characteristics(client):
        props = ",".join(char.properties or [])
        LOGGER.debug(
            "GATT service=%s char=%s props=%s",
            service_uuid,
            char.uuid,
            props,
        )


async def _ensure_services_discovered(client: BleakClient) -> None:
    """Ensure GATT services/characteristics are available on the client.

    Bleak's API differs slightly across versions and backends.
    """

    services = getattr(client, "services", None)
    if services:
        try:
            if len(services) > 0:
                return
        except TypeError:
            return

    get_services = getattr(client, "get_services", None)
    if callable(get_services):
        result = get_services()
        if inspect.isawaitable(result):
            await result
        return

    backend = getattr(client, "_backend", None)
    if backend is not None:
        backend_get_services = getattr(backend, "get_services", None)
        if callable(backend_get_services):
            result = backend_get_services()
            if inspect.isawaitable(result):
                await result
            return

    raise RuntimeError(
        "Cannot enumerate GATT services with this Bleak version/backend. "
        "Try upgrading bleak (and ensure Bluetooth permissions are granted)."
    )


def _pick_io_characteristics(
    client: BleakClient,
) -> tuple[str, str, List[str]]:
    """Return (notify_uuid, write_uuid, write_candidates) for the peripheral.

    We strongly prefer vendor-specific characteristics (0xFF**) over standard
    ones like 0x2A00, since many standard characteristics are read-only and
    will fail with "Access Denied" on writes.
    """

    notify_candidates: List[str] = []
    write_candidates: List[str] = []
    for _, char in _iter_characteristics(client):
        props = {prop.lower() for prop in (char.properties or [])}
        if "notify" in props or "indicate" in props:
            notify_candidates.append(char.uuid)
        if "write" in props or "write-without-response" in props:
            write_candidates.append(char.uuid)

    def _notify_score(uuid: str) -> tuple[int, str]:
        # Prefer vendor UUIDs, then known "FFE4" pattern, then lexical.
        uuid16 = _uuid16(uuid)
        return (
            (0 if _is_vendor_uuid(uuid) else 1),
            (0 if uuid16 == 0xFFE4 else 1),
            uuid,
        )

    def _write_score(uuid: str) -> tuple[int, str]:
        uuid16 = _uuid16(uuid)
        # Best: vendor UUIDs, prefer FFE9/FFE5 patterns commonly used for RX.
        preferred = {0xFFE9, 0xFFE5, 0xFFE1, 0xFFE3, 0xFFE4}
        return (
            (0 if _is_vendor_uuid(uuid) else 1),
            (0 if (uuid16 in preferred) else 1),
            (2 if _is_standard_uuid(uuid) else 0),
            uuid,
        )

    notify_candidates_sorted = sorted(notify_candidates, key=_notify_score)
    write_candidates_sorted = sorted(write_candidates, key=_write_score)

    # Prefer a single vendor characteristic that supports both.
    for uuid in notify_candidates_sorted:
        if uuid in write_candidates_sorted and _is_vendor_uuid(uuid):
            return uuid, uuid, write_candidates_sorted
    # Otherwise any single characteristic that supports both.
    for uuid in notify_candidates_sorted:
        if uuid in write_candidates_sorted:
            return uuid, uuid, write_candidates_sorted

    if notify_candidates_sorted and write_candidates_sorted:
        return (
            notify_candidates_sorted[0],
            write_candidates_sorted[0],
            write_candidates_sorted,
        )
    if notify_candidates_sorted:
        return notify_candidates_sorted[0], notify_candidates_sorted[0], []

    available = sorted({c.uuid for _, c in _iter_characteristics(client)})
    raise RuntimeError(
        "No notify/write GATT characteristic found. "
        f"Available characteristics: {', '.join(available) if available else '(none)'}"
    )


class WitMotionBleClient:
    """Async context manager for talking to a WIT MOTION BLE peripheral."""

    def __init__(self, device: BLEDevice | str) -> None:
        self._device = device
        self._client: Optional[BleakClient] = None
        self._notify_uuid: Optional[str] = None
        self._write_uuid: Optional[str] = None
        self._write_candidates: List[str] = []
        self._notification_queue: asyncio.Queue[bytes] = asyncio.Queue()

    async def __aenter__(self) -> "WitMotionBleClient":
        self._client = BleakClient(self._device)
        await self._client.connect(timeout=10.0)
        await _ensure_services_discovered(self._client)
        _debug_log_characteristics(self._client)
        (
            self._notify_uuid,
            self._write_uuid,
            self._write_candidates,
        ) = _pick_io_characteristics(self._client)
        LOGGER.info("Using notify=%s write=%s", self._notify_uuid, self._write_uuid)
        LOGGER.info("Connected to %s", self._client.address)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._client and self._client.is_connected:
            await self._client.disconnect()
            LOGGER.info("Disconnected from %s", self._client.address)

    async def write_command(
        self,
        payload: bytes,
        response: Optional[bool] = None,
    ) -> None:
        """Send a raw protocol command to the sensor.

        Some Windows stacks report "Access Denied" for characteristics that are
        technically writable but require encryption, pairing, or are simply the
        wrong attribute (e.g. Device Name 0x2A00). We therefore try multiple
        write candidates and both write modes (with/without response).
        """

        if self._client is None:
            raise RuntimeError("Client is not connected")
        if self._write_uuid is None:
            raise RuntimeError("Write characteristic not resolved")

        uuids_to_try = [self._write_uuid] + [
            uuid for uuid in self._write_candidates if uuid != self._write_uuid
        ]
        responses_to_try = [response] if response is not None else [False, True]

        last_error: Exception | None = None
        for uuid in uuids_to_try:
            for resp in responses_to_try:
                try:
                    await self._client.write_gatt_char(uuid, payload, response=resp)
                    if uuid != self._write_uuid:
                        LOGGER.info("Switching write characteristic to %s", uuid)
                        self._write_uuid = uuid
                    return
                except BleakError as exc:
                    last_error = exc
                    message = str(exc)
                    if "access denied" in message.lower():
                        continue
                    raise

        if last_error is not None:
            raise last_error

    async def read_raw_packet(self) -> bytes:
        """Read the latest TX characteristic payload."""

        if self._client is None:
            raise RuntimeError("Client is not connected")
        if self._notify_uuid is None:
            raise RuntimeError("Notify characteristic not resolved")
        return await self._client.read_gatt_char(self._notify_uuid)

    async def read_measurement(self) -> WitMotionMeasurement:
        """Read a complete sensor snapshot and parse it."""

        raw = await self.read_raw_packet()
        return parse_sensor_packet(raw)

    async def start_notifications(self) -> None:
        LOGGER.info("Starting notifications...")
        if self._client is None:
            raise RuntimeError("Client is not connected")
        if self._notify_uuid is None:
            raise RuntimeError("Notify characteristic not resolved")

        def _handler(_: int, data: bytearray) -> None:  # pragma: no cover - tiny helper
            self._notification_queue.put_nowait(bytes(data))

        await self._client.start_notify(self._notify_uuid, _handler)

    async def read_until(
        self,
        predicate: Callable[[WitMotionMeasurement], bool],
        timeout_s: float = 3.0,
    ) -> WitMotionMeasurement:
        """Wait until a notification matches a predicate.

        Parameters:
            predicate: Function that returns True for a desired measurement.
            timeout_s: Maximum seconds to wait.

        Returns:
            The first matching measurement.

        Raises:
            TimeoutError: If no matching measurement arrives in time.
        """

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for measurement")
            raw = await asyncio.wait_for(
                self._notification_queue.get(),
                timeout=remaining,
            )
            for measurement in iter_sensor_measurements(raw):
                if predicate(measurement):
                    return measurement

    async def notification_stream(self) -> AsyncIterator[WitMotionMeasurement]:
        """Yield measurements driven from notification callbacks."""

        if self._client is None:
            raise RuntimeError("Client is not connected")
        while self._client.is_connected:
            raw = await self._notification_queue.get()
            for measurement in iter_sensor_measurements(raw):
                yield measurement


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the blex tool."""

    parser = argparse.ArgumentParser(prog="blex")
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List discovered WIT MOTION devices",
    )
    parser.add_argument(
        "-d",
        "--device",
        metavar="DEVICE",
        help=(
            "Device selector: index from --list, BLE address, "
            "or name substring"
        ),
    )
    parser.add_argument(
        "-m",
        "--measurement",
        action="append",
        metavar="TYPE",
        choices=["acc", "as", "angle", "h", "q", "dt", "power"],
        help=(
            "Measurement type: acc, as (angular speed), angle, "
            "h (magnetometer), q (quaternion), dt (datetime), power"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Scan timeout in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="How many scan attempts to find a device (default: 3)",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=3.0,
        help="Per-measurement wait timeout in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def _format_device_line(index: int, device: BLEDevice) -> str:
    """Format a BLEDevice for human-friendly --list output."""

    name = device.name or "(no name)"
    return f"{index}: {name} [{device.address}]"


def _looks_like_mac_address(value: str) -> bool:
    """Return True if value looks like a Bluetooth MAC address."""

    parts = value.split(":")
    if len(parts) != 6:
        return False
    for part in parts:
        if len(part) != 2:
            return False
        try:
            int(part, 16)
        except ValueError:
            return False
    return True


def _select_device(
    devices: List[BLEDevice],
    selector: Optional[str],
) -> BLEDevice | str:
    """Select a device by index, address, or name substring.

    If selector looks like a MAC address but isn't found in the scan results,
    return the address string and let Bleak attempt a best-effort connect.
    """

    if not devices:
        if selector is not None and _looks_like_mac_address(selector.strip()):
            return selector.strip()
        raise RuntimeError("No devices discovered")
    if selector is None:
        return devices[0]

    value = selector.strip()

    # Index
    try:
        idx = int(value)
        if 0 <= idx < len(devices):
            return devices[idx]
    except ValueError:
        pass

    value_lower = value.lower()
    for device in devices:
        if (device.address or "").lower() == value_lower:
            return device
    for device in devices:
        if value_lower in (device.name or "").lower():
            return device

    if _looks_like_mac_address(value):
        LOGGER.warning(
            "Device %s not discovered in this scan; trying direct connect by address.",
            value,
        )
        return value

    raise RuntimeError(
        f"Device '{selector}' not found. Run with --list to see options."
    )


def _measurement_register(measurement_type: str) -> Optional[int]:
    """Return register to request for a measurement type (or None)."""

    mapping = {
        "h": 0x3A,
        "q": 0x51,
        "dt": 0x30,
        "power": 0x64,
    }
    return mapping.get(measurement_type)


def _format_measurement_value(
    measurement_type: str,
    measurement: WitMotionMeasurement,
) -> str:
    """Format only the requested field from a decoded measurement."""

    if measurement_type == "acc" and measurement.acc is not None:
        return (
            f"ACC(g)=({measurement.acc[0]:.3f},"
            f"{measurement.acc[1]:.3f},{measurement.acc[2]:.3f})"
        )
    if measurement_type == "as" and measurement.gyro is not None:
        return (
            f"GYRO(°/s)=({measurement.gyro[0]:.1f},"
            f"{measurement.gyro[1]:.1f},{measurement.gyro[2]:.1f})"
        )
    if measurement_type == "angle" and measurement.angle is not None:
        return (
            f"ANGLE(°)=({measurement.angle[0]:.1f},"
            f"{measurement.angle[1]:.1f},{measurement.angle[2]:.1f})"
        )
    if measurement_type == "h" and measurement.mag_mg is not None:
        return (
            f"MAG(mG)=({measurement.mag_mg[0]},"
            f"{measurement.mag_mg[1]},{measurement.mag_mg[2]})"
        )
    if measurement_type == "q" and measurement.quat is not None:
        return (
            f"Q=({measurement.quat[0]:.3f},{measurement.quat[1]:.3f},"
            f"{measurement.quat[2]:.3f},{measurement.quat[3]:.3f})"
        )
    if measurement_type == "dt" and measurement.datetime is not None:
        return measurement.datetime.isoformat(sep=" ", timespec="seconds")
    if measurement_type == "power" and measurement.power_raw is not None:
        return f"POWER(raw)={measurement.power_raw}"
    return str(measurement)


async def _find_device_by_address(
    address: str,
    timeout: float,
    retries: int,
) -> Optional[BLEDevice]:
    """Try to resolve a BLEDevice by address with scan retries."""

    find_fn = getattr(BleakScanner, "find_device_by_address", None)
    for _ in range(max(1, retries)):
        if callable(find_fn):
            found = await find_fn(address, timeout=timeout)
            if found is not None:
                return found
        else:
            devices = await BleakScanner.discover(timeout=timeout)
            for device in devices:
                if (device.address or "").lower() == address.lower():
                    return device
    return None


async def _read_measurement_type(
    client: WitMotionBleClient,
    measurement_type: str,
    wait_s: float,
) -> WitMotionMeasurement:
    """Read a single measurement of a given type."""

    reg = _measurement_register(measurement_type)

    def _matches(measurement: WitMotionMeasurement) -> bool:
        if measurement_type == "acc":
            return measurement.acc is not None
        if measurement_type == "as":
            return measurement.gyro is not None
        if measurement_type == "angle":
            return measurement.angle is not None
        if measurement_type == "h":
            return measurement.mag_mg is not None
        if measurement_type == "q":
            return measurement.quat is not None
        if measurement_type == "dt":
            return measurement.datetime is not None
        if measurement_type == "power":
            return measurement.power_raw is not None
        return False

    if reg is None:
        return await client.read_until(_matches, timeout_s=wait_s)

    # Register-based measurements are often returned as a separate 0x71 frame,
    # and some firmware sends them only intermittently. Retry a few times.
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            await client.write_command(build_read_register_command(reg))
        except Exception as exc:
            LOGGER.warning(
                "Failed to request %s window (reg=0x%02X) attempt %d: %s",
                measurement_type,
                reg,
                attempt,
                exc,
            )
        try:
            return await client.read_until(_matches, timeout_s=wait_s)
        except TimeoutError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise TimeoutError("Timed out waiting for measurement")


async def _run_cli(argv: Optional[List[str]] = None) -> int:
    """Entry point for the blex console script."""

    args = _parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            LOGGER.info("Using bleak %s", version("bleak"))
        except PackageNotFoundError:
            LOGGER.info("Using bleak (version unknown)")
    except Exception:
        pass

    devices = await discover_witmotion_devices(timeout=args.timeout)

    if args.list:
        if not devices:
            print("No WIT MOTION devices discovered.")
            return 1
        for idx, device in enumerate(devices):
            print(_format_device_line(idx, device))
        return 0

    if not args.measurement:
        if not devices:
            print("No WIT MOTION devices discovered.")
            return 1
        for idx, device in enumerate(devices):
            print(_format_device_line(idx, device))
        print("\nProvide --measurement TYPE to read values.")
        return 0

    if not devices:
        print("No WIT MOTION devices discovered.")
        return 1

    try:
        selected = _select_device(devices, args.device)
    except RuntimeError as exc:
        if devices:
            print(str(exc))
            print("Discovered devices:")
            for idx, device in enumerate(devices):
                print(_format_device_line(idx, device))
        raise

    if isinstance(selected, str):
        if sys.platform.startswith("win"):
            found = await _find_device_by_address(
                selected,
                timeout=args.timeout,
                retries=args.retries,
            )
            if found is None:
                raise RuntimeError(
                    f"Device with address {selected} was not found in scans. "
                    "Wake the sensor and retry (or use --list and pass an index)."
                )
            selected = found
        else:
            print(f"Using address {selected}")

    if not isinstance(selected, str):
        print(f"Using {selected.name or 'unnamed device'} ({selected.address})")

    async with WitMotionBleClient(selected) as client:
        await client.start_notifications()
        for measurement_type in args.measurement:
            measurement = await _read_measurement_type(
                client,
                measurement_type,
                args.wait,
            )
            print(
                f"{measurement_type}: "
                f"{_format_measurement_value(measurement_type, measurement)}"
            )

    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_run_cli(sys.argv[1:])))


if __name__ == "__main__":
    main()
