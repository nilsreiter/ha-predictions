"""Base Entity class."""

from __future__ import annotations

from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .coordinator import HAPredictionUpdateCoordinator


class HAPredictionEntity(CoordinatorEntity[HAPredictionUpdateCoordinator]):
    """Base Entity class."""

    def __init__(self, coordinator: HAPredictionUpdateCoordinator) -> None:
        """Initialize."""
        super().__init__(coordinator)
        self._attr_device_info = coordinator.config_entry.runtime_data.device_info
        self.coordinator.register(self)

    def notify(self, msg: str) -> None:
        """Handle notifications from the coordinator."""
