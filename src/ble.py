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
from bleak.backends.device import BLEDevice

LOGGER = logging.getLogger(__name__)

WITMOTION_SERVICE_UUID = "0000ffe0-0000-1000-8000-00805f9b34fb"
WITMOTION_TX_CHAR_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"

FRAME_LENGTH = 11
ACC_SCALE = 16.0 / 32768
GYRO_SCALE = 2000.0 / 32768
MAG_SCALE = 2000.0 / 32768

_FRAME_SCALES = {
    0x51: ACC_SCALE,
    0x52: GYRO_SCALE,
    0x53: MAG_SCALE,
}


@dataclass
class WitMotionMeasurement:
    acc: Tuple[float, float, float]
    gyro: Tuple[float, float, float]
    mag: Tuple[float, float, float]

    def __str__(self) -> str:  # pragma: no cover - simple helper
        return (
            f"ACC(g): ({self.acc[0]:.3f}, {self.acc[1]:.3f}, {self.acc[2]:.3f}) | "
            f"GYRO(°/s): ({self.gyro[0]:.3f}, {self.gyro[1]:.3f}, {self.gyro[2]:.3f}) | "
            f"MAG(uT): ({self.mag[0]:.1f}, {self.mag[1]:.1f}, {self.mag[2]:.1f})"
        )


async def discover_witmotion_devices(
    timeout: float = 5.0, name_filter: str = "WT901"
) -> List[BLEDevice]:
    """Scan for nearby WIT MOTION peripherals and return matching BLEDevices."""

    LOGGER.info("Scanning for WIT MOTION devices (timeout=%.1fs)", timeout)
    devices = await BleakScanner.discover(timeout=timeout)
    matches = [device for device in devices if name_filter.lower() in (device.name or "").lower()]
    LOGGER.info("Found %d matching device(s)", len(matches))
    return matches


def parse_sensor_packet(data: bytes) -> WitMotionMeasurement:
    """Parse a stream of WIT MOTION sensor frames into one measurement set."""

    acc: Optional[Tuple[float, float, float]] = None
    gyro: Optional[Tuple[float, float, float]] = None
    mag: Optional[Tuple[float, float, float]] = None
    idx = 0
    while idx + FRAME_LENGTH <= len(data):
        if data[idx] != 0x55:
            idx += 1
            continue
        frame_id = data[idx + 1]
        payload = data[idx + 2 : idx + 8]
        if len(payload) < 6:
            break

        checksum = data[idx + 10]
        computed = sum(data[idx : idx + 10]) & 0xFF
        if checksum != computed:
            LOGGER.debug("Checksum mismatch on frame at %d", idx)
            idx += FRAME_LENGTH
            continue

        if frame_id not in _FRAME_SCALES:
            idx += FRAME_LENGTH
            continue

        raw_vector = struct.unpack("<hhh", payload)
        scale = _FRAME_SCALES[frame_id]
        scaled = tuple(value * scale for value in raw_vector)
        if frame_id == 0x51:
            acc = scaled
        elif frame_id == 0x52:
            gyro = scaled
        elif frame_id == 0x53:
            mag = scaled
        idx += FRAME_LENGTH

    return WitMotionMeasurement(
        acc=acc or (0.0, 0.0, 0.0),
        gyro=gyro or (0.0, 0.0, 0.0),
        mag=mag or (0.0, 0.0, 0.0),
    )


class WitMotionBleClient:
    """Async context manager for talking to a WIT MOTION BLE peripheral."""

    def __init__(self, address: str) -> None:
        self._address = address
        self._client: Optional[BleakClient] = None
        self._notification_queue: asyncio.Queue[bytes] = asyncio.Queue()

    async def __aenter__(self) -> "WitMotionBleClient":
        self._client = BleakClient(self._address)
        await self._client.connect(timeout=10.0)
        LOGGER.info("Connected to %s", self._address)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._client and self._client.is_connected:
            await self._client.disconnect()
            LOGGER.info("Disconnected from %s", self._address)

    async def write_command(self, payload: bytes, response: bool = False) -> None:
        """Send a raw protocol command to the sensor."""

        if self._client is None:
            raise RuntimeError("Client is not connected")
        await self._client.write_gatt_char(WITMOTION_TX_CHAR_UUID, payload, response=response)

    async def read_raw_packet(self) -> bytes:
        """Read the latest TX characteristic payload."""

        if self._client is None:
            raise RuntimeError("Client is not connected")
        return await self._client.read_gatt_char(WITMOTION_TX_CHAR_UUID)

    async def read_measurement(self) -> WitMotionMeasurement:
        """Read a complete sensor snapshot and parse it."""

        raw = await self.read_raw_packet()
        return parse_sensor_packet(raw)

    async def start_notifications(self) -> None:
        if self._client is None:
            raise RuntimeError("Client is not connected")

        def _handler(_: int, data: bytearray) -> None:  # pragma: no cover - tiny helper
            self._notification_queue.put_nowait(bytes(data))

        await self._client.start_notify(WITMOTION_TX_CHAR_UUID, _handler)

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
    if not devices:
        print("No WIT MOTION devices discovered. Make sure the sensor is broadcasting BLE signals.")
        return

    selected = devices[0]
    print(f"Using {selected.name or 'unnamed device'} ({selected.address})")
    async with WitMotionBleClient(selected.address) as client:
        measurement = await client.read_measurement()
        print("One-shot measurement:", measurement)

        await client.start_notifications()
        async for idx, measurement in enumerate(client.notification_stream()):
            print(f"Notification {idx + 1}: {measurement}")
            if idx >= 2:
                break


def main() -> None:
    asyncio.run(_run_main())


if __name__ == "__main__":
    main()
