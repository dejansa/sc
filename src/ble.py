"""Helpers for interacting with WIT MOTION BLE 5.0 sensors.

Implements the Bluetooth 5.0 communication protocol documented at:
https://wit-motion.gitbook.io/witmotion-sdk/ble-5.0-protocol/bluetooth-5.0-communication-protocol

The module provides scanning, command building, and packet parsing utilities
so you can read IMU data and configure streaming parameters without needing
to re-implement the low-level protocol every time.
"""

import asyncio
import logging
import struct
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional, Tuple

from bleak import BleakClient, BleakScanner
from bleak.exc import BleakCharacteristicNotFoundError
from bleak.backends.device import BLEDevice

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
    matches = [device for device in devices if name_filter.lower() in (device.name or "").lower()]
    LOGGER.info("Found %d matching device(s)", len(matches))
    return matches


def _decode_i16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<h", data, offset)[0]


def parse_sensor_packet(data: bytes) -> WitMotionMeasurement:
    """Parse a single BLE 5.0 packet (up to 20 bytes).

    Per GitBook:
      - 0x55 0x61: default 20-byte packet with acc + gyro + angle (18 bytes payload)
      - 0x55 0x71: register window (start register + 8 registers = 20 bytes total)
    """

    if not data:
        return WitMotionMeasurement(raw=b"")
    # Notifications can deliver more than one 20B frame at once.
    # Parse the first valid one we find.
    for start in range(0, max(1, len(data) - 1)):
        if start + 2 > len(data) or data[start] != 0x55:
            continue
        flag = data[start + 1]
        if start + BLE_PACKET_SIZE > len(data):
            continue
        packet = data[start : start + BLE_PACKET_SIZE]

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
            return WitMotionMeasurement(
                acc=(ax, ay, az),
                gyro=(gx, gy, gz),
                angle=(roll, pitch, yaw),
                raw=packet,
            )

        if flag == 0x71:
            reg = packet[2] | (packet[3] << 8)
            # 8 registers, 2 bytes each.
            regs = packet[4:]
            measurement = WitMotionMeasurement(start_register=reg, raw=packet)
            # Known windows
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
            return measurement

    return WitMotionMeasurement(raw=bytes(data))


def _iter_characteristics(client: BleakClient):
    for service in client.services:
        for char in service.characteristics:
            yield service.uuid, char


def _pick_io_characteristics(client: BleakClient) -> tuple[str, str]:
    """Return (notify_uuid, write_uuid) for the connected peripheral."""

    notify_candidates: List[str] = []
    write_candidates: List[str] = []
    for _, char in _iter_characteristics(client):
        props = {prop.lower() for prop in (char.properties or [])}
        if "notify" in props or "indicate" in props:
            notify_candidates.append(char.uuid)
        if "write" in props or "write-without-response" in props:
            write_candidates.append(char.uuid)

    # Prefer a single characteristic that supports both.
    for uuid in notify_candidates:
        if uuid in write_candidates:
            return uuid, uuid
    if notify_candidates and write_candidates:
        return notify_candidates[0], write_candidates[0]
    if notify_candidates:
        return notify_candidates[0], notify_candidates[0]
    raise BleakCharacteristicNotFoundError(
        "No notify/write GATT characteristic found. Available characteristics: "
        + ", ".join(sorted({c.uuid for _, c in _iter_characteristics(client)}))
    )


class WitMotionBleClient:
    """Async context manager for talking to a WIT MOTION BLE peripheral."""

    def __init__(self, device: BLEDevice | str) -> None:
        self._device = device
        self._client: Optional[BleakClient] = None
        self._notify_uuid: Optional[str] = None
        self._write_uuid: Optional[str] = None
        self._notification_queue: asyncio.Queue[bytes] = asyncio.Queue()

    async def __aenter__(self) -> "WitMotionBleClient":
        self._client = BleakClient(self._device)
        await self._client.connect(timeout=10.0)
        await self._client.get_services()
        self._notify_uuid, self._write_uuid = _pick_io_characteristics(self._client)
        LOGGER.info("Using notify=%s write=%s", self._notify_uuid, self._write_uuid)
        LOGGER.info("Connected to %s", self._client.address)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._client and self._client.is_connected:
            await self._client.disconnect()
            LOGGER.info("Disconnected from %s", self._client.address)

    async def write_command(self, payload: bytes, response: bool = False) -> None:
        """Send a raw protocol command to the sensor."""

        if self._client is None:
            raise RuntimeError("Client is not connected")
        if self._write_uuid is None:
            raise RuntimeError("Write characteristic not resolved")
        await self._client.write_gatt_char(self._write_uuid, payload, response=response)

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
        print("Starting notifications...")
        if self._client is None:
            raise RuntimeError("Client is not connected")
        if self._notify_uuid is None:
            raise RuntimeError("Notify characteristic not resolved")

        def _handler(_: int, data: bytearray) -> None:  # pragma: no cover - tiny helper
            self._notification_queue.put_nowait(bytes(data))

        await self._client.start_notify(self._notify_uuid, _handler)

    async def notification_stream(self) -> AsyncIterator[WitMotionMeasurement]:
        """Yield measurements driven from notification callbacks."""

        if self._client is None:
            raise RuntimeError("Client is not connected")
        while self._client.is_connected:
            raw = await self._notification_queue.get()
            yield parse_sensor_packet(raw)


async def _run_main() -> None:
    logging.basicConfig(level=logging.INFO)
    devices = await discover_witmotion_devices()
    print(f"{devices=}")
    if not devices:
        print("No WIT MOTION devices discovered. Make sure the sensor is broadcasting BLE signals.")
        return

    # On Windows, connecting by MAC address can be unreliable. Prefer passing the
    # BLEDevice object that came from discovery, and try each candidate.
    last_error: Exception | None = None
    for selected in devices:
        print(f"Using {selected.name or 'unnamed device'} ({selected.address})")
        try:
            async with WitMotionBleClient(selected) as client:
                await client.start_notifications()

                # Ask the sensor to return extra register windows (mag/quaternion)
                # per the BLE 5.0 protocol docs.
                await client.write_command(build_read_register_command(0x3A))  # magnetometer
                await client.write_command(build_read_register_command(0x51))  # quaternion

                async for idx, measurement in enumerate(client.notification_stream()):
                    print(f"Notification {idx + 1}: {measurement}")
                    if idx >= 2:
                        return
        except Exception as exc:
            last_error = exc
            print(f"Failed to connect to {selected.address}: {exc}")
            continue

    if last_error is not None:
        raise last_error


def main() -> None:
    asyncio.run(_run_main())


if __name__ == "__main__":
    main()
