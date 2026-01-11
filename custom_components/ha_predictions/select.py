from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.components.select import SelectEntity, SelectEntityDescription
from propcache import cached_property

from .const import OP_MODE_PROD, OP_MODE_TRAIN
from .entity import HAPredictionEntity

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddEntitiesCallback

    from .coordinator import HAPredictionUpdateCoordinator
    from .data import HAPredictionConfigEntry


async def async_setup_entry(
    hass: HomeAssistant,  # noqa: ARG001 Unused function argument: `hass`
    entry: HAPredictionConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the select platform."""
    async_add_entities(
        [
            SelectModeEntity(
                coordinator=entry.runtime_data.coordinator,
                entity_description=SelectEntityDescription(
                    key="operation_mode",
                    name="Mode",
                    icon="mdi:form-dropdown",
                ),
            ),
        ]
    )


class SelectModeEntity(HAPredictionEntity, SelectEntity):
    _attr_options: list[str] = [OP_MODE_TRAIN, OP_MODE_PROD]

    def __init__(
        self,
        coordinator: HAPredictionUpdateCoordinator,
        entity_description: SelectEntityDescription,
    ):
        super().__init__(coordinator)
        self.entity_description = entity_description
        self.coordinator.register(self)

    @cached_property
    def unique_id(self) -> str | None:
        return self.coordinator.config_entry.entry_id + "-select-operation-mode"

    @property
    def current_option(self) -> str:
        """Return the current option."""
        return self.coordinator.operation_mode

    @property
    def available(self) -> bool:
        return True

    async def async_select_option(self, option: str) -> None:
        """Change the selected option."""
        self.coordinator.set_operation_mode(option)
        self.schedule_update_ha_state()
