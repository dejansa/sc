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
from typing import Any, AsyncIterator, Callable, Iterator, List, Optional, Tuple

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


def build_set_baud_command(baud_code: int) -> bytes:
    """Build a command to set the serial baud rate.

    Protocol: FF AA 04 BAUD 00

    Note: This configures the module's serial baud register; some BLE-only
    devices may ignore it.
    """

    if not 0 <= baud_code <= 0xFF:
        raise ValueError("baud_code must fit in one byte")
    return bytes([0xFF, 0xAA, 0x04, baud_code, 0x00])


def build_calibration_command(code: int) -> bytes:
    """Build a calibration/zero-offset command.

    Protocol: FF AA 01 CAL 00

    Known values from the BLE 5.0 protocol documentation:
      - 0x01: acceleration calibration
      - 0x05: acceleration calibration L
      - 0x06: acceleration calibration R
      - 0x07: magnetic field calibration (start)
      - 0x00: complete magnetic field calibration

    Some devices also support a gyro zero-offset command, but it is not listed
    in the docs we reference; this tool may be best-effort.
    """

    if not 0 <= code <= 0xFF:
        raise ValueError("code must fit in one byte")
    return bytes([0xFF, 0xAA, 0x01, code, 0x00])


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
    temperature_c: Optional[float] = None
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
        if self.temperature_c is not None:
            parts.append(f"TEMP(°C)={self.temperature_c:.2f}")
            if self.power_raw is not None:
                percent = battery_percent_from_raw(self.power_raw)
                if percent is None:
                    parts.append(f"BAT(raw)={self.power_raw}")
                else:
                    parts.append(f"BAT={percent}% (raw={self.power_raw})")
        if self.start_register is not None:
            parts.append(f"reg=0x{self.start_register:02X}")
        if not parts:
            return f"RAW({len(self.raw)}B): {self.raw.hex(' ')}"
        return " | ".join(parts)


def battery_percent_from_raw(raw_value: int) -> Optional[int]:
    """Convert a power register raw value into battery percent.

    The BLE 5.0 protocol doc provides a piecewise mapping from the decimal
    register value to an approximate battery percentage.
    """

    if raw_value < 0:
        return None

    # Thresholds from the WITMOTION BLE 5.0 documentation.
    if raw_value > 830:
        return 100
    if 393 <= raw_value <= 396:
        return 90
    if 387 <= raw_value < 393:
        return 75
    if 382 <= raw_value < 387:
        return 60
    if 379 <= raw_value < 382:
        return 50
    if 377 <= raw_value < 379:
        return 40
    if 373 <= raw_value < 377:
        return 30
    if 370 <= raw_value < 373:
        return 20
    if 368 <= raw_value < 370:
        return 15
    if 350 <= raw_value < 368:
        return 10
    if 340 <= raw_value < 350:
        return 5
    if raw_value < 340:
        return 0

    # Unknown region between 396 and 830.
    return None


def _make_bleak_scanner(detection_callback: Callable[[BLEDevice, Any], None]) -> Any:
    """Create a BleakScanner, preferring active scanning when supported."""

    kwargs: dict[str, Any] = {}
    try:
        sig = inspect.signature(BleakScanner)
        if "scanning_mode" in sig.parameters:
            kwargs["scanning_mode"] = "active"
    except Exception:
        pass

    try:
        return BleakScanner(detection_callback=detection_callback, **kwargs)
    except TypeError:
        return BleakScanner(detection_callback=detection_callback)


async def discover_witmotion_devices(
    timeout: float = 5.0,
    name_filter: str = "WT901",
    retries: int = 1,
) -> List[BLEDevice]:
    """Scan for nearby WIT MOTION peripherals and return matching BLEDevices.

    Notes on reliability:
    - On Windows/WinRT, a peripheral can be seen with a name in one advertisement
      and without it in another. A single BleakScanner.discover() snapshot can
      therefore miss devices if you filter solely on the final device.name.
    - Even when names are stable, a single short scan window can miss devices
      that are advertising slowly or that the radio doesn't catch in time.

    This helper mitigates both issues by collecting detection callbacks over the
    scan window and optionally repeating scans, merging results by address.
    """

    needle = (name_filter or "").strip().lower()
    scan_attempts = max(1, retries)
    LOGGER.info(
        "Scanning for WIT MOTION devices (timeout=%.1fs, attempts=%d)",
        timeout,
        scan_attempts,
    )

    best_name_by_address: dict[str, str] = {}
    device_by_address: dict[str, BLEDevice] = {}
    matched_addresses: set[str] = set()

    def _on_detect(device: BLEDevice, adv_data: Any) -> None:
        address = device.address
        device_by_address[address] = device
        local_name = ""
        if adv_data is not None:
            local_name = (getattr(adv_data, "local_name", None) or "").strip()

        candidate = local_name or (device.name or "").strip()
        if candidate and not best_name_by_address.get(address):
            best_name_by_address[address] = candidate
        if needle and needle in candidate.lower():
            matched_addresses.add(address)
        if not needle:
            matched_addresses.add(address)

    last_total = 0
    for attempt in range(1, scan_attempts + 1):
        try:
            scanner = _make_bleak_scanner(_on_detect)
            await scanner.start()
            await asyncio.sleep(timeout)
            await scanner.stop()
        except Exception as exc:
            LOGGER.debug(
                "Scan attempt %d: falling back to discover() due to error: %s",
                attempt,
                exc,
            )
            devices = await BleakScanner.discover(timeout=timeout)
            for device in devices:
                _on_detect(device, None)

        total_now = len(device_by_address)
        LOGGER.debug(
            "Scan attempt %d/%d: saw %d unique device(s)",
            attempt,
            scan_attempts,
            total_now,
        )

        # If we are no longer discovering new devices and we already have at
        # least one match, stop early.
        if total_now == last_total and matched_addresses:
            break
        last_total = total_now

    matches: List[BLEDevice] = []
    for address in matched_addresses:
        device = device_by_address.get(address)
        if device is None:
            continue
        best_name = best_name_by_address.get(address, "").strip()
        if best_name and not (device.name or "").strip():
            try:
                device.name = best_name
            except Exception:
                pass
        matches.append(device)

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
            if reg == 0x40 and len(regs) >= 2:
                measurement.temperature_c = _decode_i16(regs, 0) / 100.0
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
        # Keep this bounded: in REPL a user may read sporadically while the
        # sensor streams at 50-200 Hz, which would otherwise accumulate stale
        # measurements and make reads appear "stuck".
        self._notification_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)

    def clear_pending_notifications(self) -> int:
        """Drop queued notifications and return how many were removed."""

        removed = 0
        while True:
            try:
                self._notification_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                removed += 1
        return removed

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
            payload = bytes(data)
            try:
                self._notification_queue.put_nowait(payload)
            except asyncio.QueueFull:
                # Drop the oldest queued item so we keep the newest samples.
                try:
                    self._notification_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self._notification_queue.put_nowait(payload)
                except asyncio.QueueFull:
                    # Extremely unlikely: ignore if still full.
                    pass

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
        choices=[
            "acc",
            "as",
            "angle",
            "h",
            "q",
            "dt",
            "temp",
            "bat",
            "power",
            "bias",
        ],
        help=(
            "Measurement type: acc, as (angular speed), angle, "
            "h (magnetometer), q (quaternion), dt (datetime), temp, bat, bias"
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
        "temp": 0x40,
        "bat": 0x64,
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
    if measurement_type == "temp" and measurement.temperature_c is not None:
        return f"TEMP(°C)={measurement.temperature_c:.2f}"
    if measurement_type in {"bat", "power"} and measurement.power_raw is not None:
        percent = battery_percent_from_raw(measurement.power_raw)
        if percent is None:
            return f"BAT(raw)={measurement.power_raw}"
        return f"BAT={percent}% (raw={measurement.power_raw})"
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
        if measurement_type == "temp":
            return measurement.temperature_c is not None
        if measurement_type in {"bat", "power"}:
            return measurement.power_raw is not None
        return False

    if reg is None:
        # Drop any backlog so we read the freshest sample after this command.
        client.clear_pending_notifications()
        return await client.read_until(_matches, timeout_s=wait_s)

    # Register-based measurements are often returned as a separate 0x71 frame,
    # and some firmware sends them only intermittently. Retry a few times.
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            client.clear_pending_notifications()
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

    devices = await discover_witmotion_devices(
        timeout=args.timeout,
        retries=args.retries,
    )

    # No-args behavior: enter interactive mode.
    if argv is not None and len(argv) == 0:
        return await _run_repl(
            scan_timeout=args.timeout,
            retries=args.retries,
            wait_s=args.wait,
            debug=args.debug,
        )

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
            if measurement_type == "bias":
                w1 = await _read_register_window_i16(
                    client, start_register=0x05, wait_s=args.wait
                )
                w2 = await _read_register_window_i16(
                    client, start_register=0x0D, wait_s=args.wait
                )
                print("bias:")
                print(f"  AXOFFSET: {w1[0]}")
                print(f"  AYOFFSET: {w1[1]}")
                print(f"  AZOFFSET: {w1[2]}")
                print(f"  GXOFFSET: {w1[3]}")
                print(f"  GYOFFSET: {w1[4]}")
                print(f"  GZOFFSET: {w1[5]}")
                print(f"  HXOFFSET: {w1[6]}")
                print(f"  HYOFFSET: {w1[7]}")
                print(f"  HZOFFSET: {w2[0]}")
                continue
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


async def _async_input(prompt: str) -> str:
    """Read input without blocking the event loop."""

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, input, prompt)


def _print_repl_help() -> None:
    print(
        "Commands:\n"
        "  h, ?, help   Show this help\n"
        "  exit, quit   Disconnect and exit\n"
        "  save         Save current configuration\n"
        "  rate [hz]    Read or set return rate (0.1,0.5,1,2,5,10,20,50,100,200)\n"
        "  baud <bps>   Set baud rate (best-effort; device may ignore)\n"
        "  bias         Read zero-offset registers (AX..HZ)\n"
        "  acc0 [l|r]   Acceleration zero-offset calibration\n"
        "  gyro0        Angular velocity zero-offset (best-effort)\n"
        "  mag0         Start magnetic field calibration\n"
        "  mag0 done    Complete magnetic field calibration\n"
        "\n"
        "Measurements (same as -m):\n"
        "  acc          Acceleration\n"
        "  as           Angular speed (gyro)\n"
        "  ang          Euler angles\n"
        "  mag          Magnetometer\n"
        "  q            Quaternion\n"
        "  dt           Datetime\n"
        "  temp         Temperature\n"
        "  bat          Battery level\n"
    )


def _parse_rate_hz(value: str) -> int:
    """Map a user-specified Hz value to the protocol RATE code."""

    hz_to_code = {
        0.1: 0x01,
        0.5: 0x02,
        1.0: 0x03,
        2.0: 0x04,
        5.0: 0x05,
        10.0: 0x06,
        20.0: 0x07,
        50.0: 0x08,
        100.0: 0x09,
        200.0: 0x0A,
    }

    try:
        hz = float(value)
    except ValueError as exc:
        raise ValueError("rate must be a number") from exc

    if hz in hz_to_code:
        return hz_to_code[hz]
    raise ValueError(
        "Unsupported rate. Use one of: 0.1 0.5 1 2 5 10 20 50 100 200"
    )


def _rate_code_to_hz(code: int) -> Optional[float]:
    """Map a protocol RATE code to Hz."""

    code_to_hz = {
        0x01: 0.1,
        0x02: 0.5,
        0x03: 1.0,
        0x04: 2.0,
        0x05: 5.0,
        0x06: 10.0,
        0x07: 20.0,
        0x08: 50.0,
        0x09: 100.0,
        0x0A: 200.0,
    }
    return code_to_hz.get(code)


async def _read_register_first_u16(
    client: WitMotionBleClient,
    start_register: int,
    wait_s: float,
) -> int:
    """Read the first 16-bit value from a register-window response."""

    last_error: Exception | None = None
    for _ in range(3):
        try:
            await client.write_command(build_read_register_command(start_register))
            measurement = await client.read_until(
                lambda m: m.start_register == start_register,
                timeout_s=wait_s,
            )
            if len(measurement.raw) < 6:
                raise RuntimeError("Short register response")
            # First register value is at bytes 4..5 of the 20-byte 0x71 frame.
            return _decode_u16(measurement.raw, 4)
        except Exception as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to read register")


def _decode_register_window_i16(raw_packet: bytes) -> List[int]:
    """Decode the 8x int16 register values from a 0x55 0x71 20-byte frame."""

    if len(raw_packet) < BLE_PACKET_SIZE or raw_packet[0] != 0x55 or raw_packet[1] != 0x71:
        raise ValueError("Not a register-window packet")
    values: List[int] = []
    # Values start at byte 4, little-endian signed 16-bit.
    for offset in range(4, BLE_PACKET_SIZE, 2):
        values.append(_decode_i16(raw_packet, offset))
    return values


async def _read_register_window_i16(
    client: WitMotionBleClient,
    start_register: int,
    wait_s: float,
) -> List[int]:
    """Read a register window and decode 8 signed 16-bit values."""

    last_error: Exception | None = None
    for _ in range(3):
        try:
            client.clear_pending_notifications()
            await client.write_command(build_read_register_command(start_register))
            measurement = await client.read_until(
                lambda m: m.start_register == start_register,
                timeout_s=wait_s,
            )
            return _decode_register_window_i16(measurement.raw)
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to read register window")


def _parse_baud(value: str) -> int:
    """Map a user-specified baud rate to a BAUD code.

    The BLE 5.0 doc shows BAUD as a register but doesn't list codes here.
    This mapping follows common WITMOTION conventions; if your device uses a
    different mapping, pass a raw code as hex (e.g. 0x04).
    """

    value = value.strip().lower()
    if value.startswith("0x"):
        return int(value, 16)

    bps = int(value)
    bps_to_code = {
        9600: 0,
        19200: 1,
        38400: 2,
        57600: 3,
        115200: 4,
        230400: 5,
        460800: 6,
        921600: 7,
    }
    if bps in bps_to_code:
        return bps_to_code[bps]
    raise ValueError(
        "Unsupported baud. Use 9600/19200/38400/57600/115200 or a raw code like 0x04"
    )


async def _prompt_select_device(
    devices: List[BLEDevice],
) -> BLEDevice:
    """Prompt the user to select a device from a list."""

    for idx, device in enumerate(devices):
        print(_format_device_line(idx, device))

    while True:
        choice = (await _async_input("Select device index (or 'exit'): ")).strip()
        if choice.lower() in {"exit", "quit"}:
            raise SystemExit(0)
        try:
            idx = int(choice)
        except ValueError:
            print("Please enter a number from the list.")
            continue
        if 0 <= idx < len(devices):
            return devices[idx]
        print("Index out of range.")


async def _run_repl(
    scan_timeout: float,
    retries: int,
    wait_s: float,
    debug: bool,
) -> int:
    """Interactive mode: scan, select device, connect, then command loop."""

    log_level = logging.DEBUG if debug else logging.INFO
    logging.getLogger().setLevel(log_level)

    print(f"Scanning for devices (timeout={scan_timeout:.1f}s)...")
    devices = await discover_witmotion_devices(
        timeout=scan_timeout,
        retries=retries,
    )
    if not devices:
        print("No WIT MOTION devices discovered.")
        return 1

    selected = await _prompt_select_device(devices)
    print(f"Using {selected.name or 'unnamed device'} ({selected.address})")

    async with WitMotionBleClient(selected) as client:
        await client.start_notifications()
        _print_repl_help()

        while True:
            try:
                command = (await _async_input("blex> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                return 0

            if not command:
                continue

            parts = command.split()
            cmd = parts[0].lower()
            cmd_args = parts[1:]
            if cmd in {"exit", "quit"}:
                return 0
            if cmd in {"help", "?", "h"}:
                _print_repl_help()
                continue

            if cmd == "save":
                try:
                    await client.write_command(build_save_config_command(0))
                    print("Saved configuration.")
                except Exception as exc:
                    print(f"Error: {exc}")
                continue

            if cmd == "bias":
                try:
                    # 0x05 window returns 0x05..0x0C (AX..HY). 0x0D window is HZ.
                    w1 = await _read_register_window_i16(
                        client, start_register=0x05, wait_s=wait_s
                    )
                    w2 = await _read_register_window_i16(
                        client, start_register=0x0D, wait_s=wait_s
                    )
                    labels = [
                        ("AXOFFSET", w1[0]),
                        ("AYOFFSET", w1[1]),
                        ("AZOFFSET", w1[2]),
                        ("GXOFFSET", w1[3]),
                        ("GYOFFSET", w1[4]),
                        ("GZOFFSET", w1[5]),
                        ("HXOFFSET", w1[6]),
                        ("HYOFFSET", w1[7]),
                        ("HZOFFSET", w2[0]),
                    ]
                    for name, value in labels:
                        print(f"{name}: {value}")
                except Exception as exc:
                    print(f"Error: {exc}")
                continue

            if cmd == "rate":
                if len(cmd_args) == 0:
                    try:
                        raw_value = await _read_register_first_u16(
                            client,
                            start_register=0x03,
                            wait_s=wait_s,
                        )
                        rate_code = raw_value & 0xFF
                        hz = _rate_code_to_hz(rate_code)
                        if hz is None:
                            print(f"RATE(code)=0x{rate_code:02X}")
                        else:
                            print(f"RATE={hz:g}Hz (code=0x{rate_code:02X})")
                    except Exception as exc:
                        print(f"Error: {exc}")
                    continue

                if len(cmd_args) == 1:
                    try:
                        rate_code = _parse_rate_hz(cmd_args[0])
                        await client.write_command(build_set_rate_command(rate_code))
                        await client.write_command(build_save_config_command(0))
                        print(
                            f"Set rate to {cmd_args[0]} Hz (code=0x{rate_code:02X})."
                        )
                    except Exception as exc:
                        print(f"Error: {exc}")
                    continue

                print("Usage: rate [hz]")
                continue

            if cmd == "baud":
                if len(cmd_args) != 1:
                    print("Usage: baud <bps|0xCODE>")
                    continue
                try:
                    baud_code = _parse_baud(cmd_args[0])
                    await client.write_command(build_set_baud_command(baud_code))
                    await client.write_command(build_save_config_command(0))
                    print(
                        f"Set baud to {cmd_args[0]} (code=0x{baud_code:02X})."
                    )
                except Exception as exc:
                    print(f"Error: {exc}")
                continue

            if cmd == "acc0":
                cal_code = 0x01
                if cmd_args:
                    arg = cmd_args[0].lower()
                    if arg == "l":
                        cal_code = 0x05
                    elif arg == "r":
                        cal_code = 0x06
                    else:
                        print("Usage: acc0 [l|r]")
                        continue
                try:
                    await client.write_command(build_calibration_command(cal_code))
                    await client.write_command(build_save_config_command(0))
                    print(f"Acceleration calibration sent (code=0x{cal_code:02X}).")
                except Exception as exc:
                    print(f"Error: {exc}")
                continue

            if cmd == "gyro0":
                # Best-effort: not listed in the BLE 5.0 page we reference.
                # Many devices use 0x02 for gyro zero-offset.
                try:
                    await client.write_command(build_calibration_command(0x02))
                    await client.write_command(build_save_config_command(0))
                    print("Gyro zero-offset command sent (code=0x02, best-effort).")
                except Exception as exc:
                    print(f"Error: {exc}")
                continue

            if cmd == "mag0":
                cal_code = 0x07
                if cmd_args and cmd_args[0].lower() in {"done", "finish", "end"}:
                    cal_code = 0x00
                try:
                    await client.write_command(build_calibration_command(cal_code))
                    await client.write_command(build_save_config_command(0))
                    if cal_code == 0x07:
                        print("Mag calibration started. Rotate the sensor, then run: mag0 done")
                    else:
                        print("Mag calibration completed.")
                except Exception as exc:
                    print(f"Error: {exc}")
                continue

            # Aliases
            if cmd in {"mag", "m"}:
                cmd = "h"
            if cmd in {"gyro"}:
                cmd = "as"
            if cmd in {"ang"}:
                cmd = "angle"
            if cmd in {"battery", "bat", "power"}:
                cmd = "bat"

            if cmd not in {"acc", "as", "angle", "h", "q", "dt", "temp", "bat"}:
                print("Unknown command. Type 'help' for options.")
                continue

            # For Windows devices that go quiet, try to re-request by register a
            # few times (handled inside _read_measurement_type).
            try:
                measurement = await _read_measurement_type(client, cmd, wait_s)
                print(_format_measurement_value(cmd, measurement))
            except TimeoutError:
                print(f"Timed out waiting for {cmd}. Try again.")
            except Exception as exc:
                print(f"Error: {exc}")


def main() -> None:
    raise SystemExit(asyncio.run(_run_cli(sys.argv[1:])))


if __name__ == "__main__":
    main()
