"""Custom types for integration_blueprint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.helpers.device_registry import DeviceInfo
    from homeassistant.loader import Integration

    from .coordinator import HAPredictionUpdateCoordinator


type HAPredictionConfigEntry = ConfigEntry[HAPredictionData]


@dataclass
class HAPredictionData:
    """Data for the Blueprint integration."""

    coordinator: HAPredictionUpdateCoordinator
    integration: Integration
    datafile: Path
    device_info: DeviceInfo
    unsubscribe: list[Callable]
    # target_entity: str
    # feature_entities: list[str]
